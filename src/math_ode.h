/** \file    math_ode.h
    \brief   ODE integration classes
    \author  Eugene Vasiliev
    \date    2008-2025

    This module implements classes for integration of ordinary differential equation systems.

    OdeStepperDOP853 is a modification of the 8th order Runge-Kutta Solver from
    Hairer, Norsett & Wanner, "Solving ordinary differential equations", 1987, Berlin:Springer.
    Based on the C version (by J.Colinge) of the original Fortran code by E.Hairer & G.Wanner.
*/

#pragma once
#include "math_base.h"
#include <cmath>
#include <vector>

namespace math{

/** Prototype of a function that is used in integration of ordinary differential equation systems:
    dx/dt = f(t, x), where y is an N-dimensional vector. */
class IOdeSystem {
public:
    IOdeSystem() {};
    virtual ~IOdeSystem() {};

    /** Compute the r.h.s. of the differential equation: 
        \param[in]  t  is the integration variable (time), measured from the beginning of the timestep.
        \param[in]  x  is the vector of values of dependent variables.
        \param[out] dxdt  should return the time derivatives of these variables.
        \param[in,out] af  if not NULL, can store the optional multiplicative accuracy factor;
        its purpose is to increase the accuracy (use a tighter tolerance) when needed,
        for instance, when the potential energy of the system is much larger than the total energy.
        On input, this parameter is set to 1, and if no adjustment is needed, it can be left
        unchanged, otherwise it could be set to a lower value, which is multiplied by accRel
        in the ODE stepper (not every method will use this parameter, though).
    */
    virtual void eval(const double t, const double x[], double dxdt[], double* af=0) const = 0;

    /** Return the size of ODE system (number of variables N) */
    virtual unsigned int size() const = 0;
};

/** Prototype of a function that is used in integration of second-order ODEs of the following form:
    d2x(t) / dt2 = a(t, x),
    where x is an N-dimensional vector, and the acceleration may depend only on position x and time t,
    but not on velocity dx/dt.
    It may also provide the third time derivative:  d3x(t) / dt3 = j(t, x).
    Note: the size of the system reported by size() is 2N, i.e. vectors x and dx/dt.
*/
class IOdeSystem2ndOrder: public IOdeSystem {
public:
    /** Compute 2nd and optionally 3rd time derivatives of x(t): a(t,x)=d2x/dt2 and j(t,x)=d3x/dt3.
        \param[in]  t is the integration variable (time), measured from the beginning of the timestep.
        \param[in]  x is the vector of values of dependent variables: x(t).
        \param[out] d2xdt2 should return the second time derivative of x(t).
        \param[out] d3xdt3 if not NULL, should return the third  time derivative of x(t).
        \param[in,out] af  if not NULL, can store the optional multiplicative accuracy factor.
    */
    virtual void eval2(const double t, const double xv[], double d2xdt2[], double d3xdt3[]=0,
        double* af=0) const = 0;

    // represent the second-order ODE system as a first-order one for the vector w = {x,v=dx/dt}
    virtual void eval(const double t, const double w[], double dwdt[], double* af=0) const;
};

/** Prototype of a function that is used in integration of second-order
    linear ordinary differential equation systems with variable coefficients:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is an N-dimensional vector, and A, B are NxN matrices;
    the acceleration depends on time t and both the position x and velocity dx/dt, but only linearly
    (thus is is not a subclass of IOdeSystem2ndOrder)
    Note: the size of the system reported by size() is 2N, i.e. vectors x and dx/dt.
*/
class IOdeSystem2ndOrderLinear: public IOdeSystem {
public:
    /** Compute the matrices A and B in the r.h.s. of the differential equation at the given time:
        \param[in]  t  is the integration variable (time), measured from the beginning of the timestep.
        \param[out] a  should point to an existing array of length N^2,
        which will be filled with the flattened (row-major) matrix A: mat[i*N+j] = A_{ij}.
        \param[out] b  same for the matrix B.
    */
    virtual void evalMat(const double t, double a[], double b[]) const = 0;

    /** Represent the system as a generic first-order ODE for the vector w = {x,v=dx/dt} */
    virtual void eval(const double t, const double w[], double dwdt[], double* af=0) const;
};


/* -------- header-only integrator cores (CPU/GPU unification, Tier 3) --------
   The adaptive-timestep math of the ODE steppers lives here as templated
   AGAMA_DEVICE_INLINE free functions, shared verbatim between the CPU stepper
   classes below and the GPU persistent orbit kernels (one thread per orbit).
   The Force template parameter is any callable with the signature
       void force(T t, const T x[], T dxdt[], T* accuracyFactor)
   mirroring IOdeSystem::eval() (accuracyFactor may be NULL).
   All scratch storage is caller-provided (the CPU classes use alloca,
   device kernels use fixed-size local arrays); no heap allocation here.
   For T=double the arithmetic is bit-for-bit identical to the original
   implementation that previously lived in math_ode.cpp; numeric literals
   occurring inside expressions are wrapped in T(...) so that the T=float
   instantiation performs no accidental double-precision arithmetic. */

/** Estimate the initial timestep for an ODE integration, using the r.h.s. of the ODE
    evaluated at the initial point and two trial points (an Euler and a Heun step).
    \param[in]  force  is the r.h.s. of the ODE (see the signature above);
    \param[in]  NDIM   is the size of the ODE system;
    \param[in]  x      is the initial state vector (NDIM elements);
    \param[in]  accAbs, accRel  are the absolute and relative tolerance parameters;
    \param[in,out] scratch  is a caller-provided working array of 4*NDIM elements.
    \return  the estimated length of the initial timestep (always positive).
*/
template<typename T, typename Force>
AGAMA_DEVICE_INLINE T ode_init_timestep(const Force& force, int NDIM,
    const T x[], T accAbs, T accRel, T* scratch)
{
    // min/max limits on the initial timestep
    const T HMIN = T(1e-12), HMAX = T(1e12);

    T *xt = scratch,
    *k1 = xt + NDIM,  // dx/dt(t) at the initial point
    *k2 = k1 + NDIM,  // local temporary storage
    *k3 = k2 + NDIM;

    // compute the derivatives at the initial point
    force(/*time offset*/ T(0), x, k1, (T*)NULL);

    // compute the L2-norm of x and dx/dt (use the sum of all components for the crude estimate)
    T normx0 = 0, normd0 = 0;
    for(int i=0; i<NDIM; i++) {
        normx0 += pow_2(x [i]);
        normd0 += pow_2(k1[i]);
    }

    // estimate a reasonable timestep as |x| / |dx/dt|, with appropriate safety cutoffs
    T h1 = std::fmax(HMIN, std::fmin(HMAX, std::sqrt(normx0 / normd0) * T(0.01)));

    // perform an explicit Euler step with a length h1 estimated from the 1st derivative
    for(int i=0; i<NDIM; i++)
        xt[i] = x[i] + h1 * k1[i];
    force(/*time offset*/ h1, xt, k2, (T*)NULL);

    // Heun's method (corrector step using xt estimated during the predictor step)
    for(int i=0; i<NDIM; i++)
        xt[i] = x[i] + T(0.5) * h1 * (k1[i] + k2[i]);
    force(/*time offset*/ h1, xt, k3, (T*)NULL);

    // estimate the second and third derivatives of the solution (separately for each component),
    // and use them to choose the timestep (take the shortest one among all components);
    // this is a modification of the original algorithm aimed at improving robustness:
    // it uses a combination of higher derivatives in both the numerator and the denominator,
    // which is less likely to be identically zero, even if some of them are;
    // moreover, the estimate is performed for each component separately, better adapted to
    // their possibly very different scales
    T h2 = T(INFINITY);
    for(int i=0; i<NDIM; i++) {
        T d0 = std::fmax(accAbs, std::fmax(std::fabs(x[i]), std::fabs(xt[i]))); // |x|
        T d1 = std::fmax(std::fabs(k1[i]), std::fabs(k3[i]));            // |dx/dt|
        T d2 = std::fabs((3 * k2[i] - k1[i] - 2 * k3[i]) / h1 );         // |d2x/dt2|
        T d3 = std::fabs(6 * (k3[i] - k2[i]) / (h1*h1) );                // |d3x/dt3|
        T hh = (d0 * d2 + d1 * d1) / (d1 * d3 + d2 * d2);                // Aarseth-like criterion
        h2 = std::fmin(h2, hh);
    }

    T h = std::fmax(HMIN, std::fmin(HMAX, T(std::pow(accRel, 1./8)) * std::sqrt(h2)));
    return h;
}

/** (Re-)initialize the persistent state of the DOP853 integrator from the given state vector.
    \param[in]  force  is the r.h.s. of the ODE;
    \param[in]  NDIM   is the size of the ODE system;
    \param[in]  stateNew  is the new state vector (NDIM elements);
    \param[in]  accRel, accAbs  are the tolerance parameters;
    \param[out] state  is the persistent 10*NDIM storage of the integrator
    (x, dx/dt, and 8 dense-output coefficient blocks); the first two blocks are filled here;
    \param[in,out] nextTimeStep  is the prediction for the next timestep:
    if zero on input (the very first call), it is estimated by ode_init_timestep;
    \param[in,out] scratch  is a caller-provided working array of 4*NDIM elements.
*/
template<typename T, typename Force>
AGAMA_DEVICE_INLINE void dop853_init(const Force& force, int NDIM, const T stateNew[],
    T accRel, T accAbs, T* state, T& nextTimeStep, T* scratch)
{
    // copy the vector x
    for(int d=0; d<NDIM; d++)
        state[d] = stateNew[d];
    // obtain the derivatives dx/dt and the accuracy factor
    T accFac = 1;
    force(T(0), stateNew, /*where to store the derivs*/ state + NDIM,
        nextTimeStep==0 ? &accFac : (T*)NULL);
    if(nextTimeStep == 0)  // first call to init() at the beginning of integration, step is not known
        nextTimeStep = ode_init_timestep(force, NDIM, stateNew, accAbs, accRel, scratch) * accFac;
}

/** Perform a single adaptive timestep of the DOP853 integrator
    (8th order Runge-Kutta with error control and 7th order dense output).
    \param[in]  force  is the r.h.s. of the ODE;
    \param[in]  NDIM   is the size of the ODE system;
    \param[in]  accRel, accAbs  are the relative and absolute tolerance parameters;
    \param[in,out] state  is the persistent 10*NDIM storage: on input, blocks 0 and 1 hold
    x and dx/dt at the beginning of the timestep (as filled by dop853_init or the previous
    call to this function); on successful return, they hold the values at the end of the
    accepted timestep, and blocks 2..9 hold the dense-output interpolation coefficients
    (rcont1..rcont8) for the just-completed timestep;
    \param[in,out] xt  is a caller-provided scratch array of 10*NDIM elements;
    \param[in,out] nextTimeStep  is the prediction for the length of the next timestep,
    updated on return;
    \param[in]  maxTimeStep  is the upper limit on the length of the timestep
    (of either sign, defining the direction of integration).
    \return  the (signed) length of the timestep taken, or 0 on error
    (in which case the state is not advanced and the caller must terminate the integration).
*/
template<typename T, typename Force>
AGAMA_DEVICE_INLINE T dop853_step(const Force& force, int NDIM, T accRel, T accAbs,
    T* state, T* xt, T& nextTimeStep, T maxTimeStep)
{
    if(maxTimeStep == 0)
        return 0;  // something must be wrong, no progress made
    const T
    // fractions of timestep at each RK stage
    c2   =  0.05260015195876773187856,
    c3   =  0.07890022793815159781784,
    c4   =  0.11835034190722739672676,
    c5   =  0.28164965809277260327324,
    c6   =  0.33333333333333333333333,
    c7   =  0.25000000000000000000000,
    c8   =  0.30769230769230769230769,
    c9   =  0.65128205128205128205128,
    c10  =  0.60000000000000000000000,
    c11  =  0.85714285714285714285714,
    c12  =  1.00000000000000000000000,
    // coefficients for Runge-Kutta stages
    a21  =  0.05260015195876773187856,
    a31  =  0.01972505698453789945446,
    a32  =  0.05917517095361369836338,
    a41  =  0.02958758547680684918169,
    a43  =  0.08876275643042054754507,
    a51  =  0.24136513415926668550237,
    a53  = -0.88454947932828608534486,
    a54  =  0.92483400326179200311574,
    a61  =  0.03703703703703703703704,
    a64  =  0.17082860872947387127960,
    a65  =  0.12546768756682242501669,
    a71  =  0.03710937500000000000000,
    a74  =  0.17025221101954403931498,
    a75  =  0.06021653898045596068502,
    a76  = -0.01757812500000000000000,
    a81  =  0.03709200011850479271088,
    a84  =  0.17038392571223999381021,
    a85  =  0.10726203044637328465181,
    a86  = -0.01531943774862440175279,
    a87  =  0.00827378916381402288758,
    a91  =  0.62411095871607571711443,
    a94  = -3.36089262944694129406857,
    a95  = -0.86821934684172600681819,
    a96  =  27.5920996994467083049416,
    a97  =  20.1540675504778934086187,
    a98  = -43.4898841810699588477366,
    a101 =  0.47766253643826436589043,
    a104 = -2.48811461997166764192642,
    a105 = -0.59029082683684299637145,
    a106 =  21.2300514481811942347289,
    a107 =  15.2792336328824235832597,
    a108 = -33.2882109689848629194453,
    a109 = -0.02033120170850862613582,
    a111 = -0.93714243008598732571704,
    a114 =  5.18637242884406370830024,
    a115 =  1.09143734899672957818500,
    a116 = -8.14978701074692612513997,
    a117 = -18.5200656599969598641566,
    a118 =  22.7394870993505042818970,
    a119 =  2.49360555267965238987089,
    a1110= -3.04676447189821950038237,
    a121 =  2.27331014751653820792360,
    a124 = -10.5344954667372501984067,
    a125 = -2.00087205822486249909676,
    a126 = -17.9589318631187989172766,
    a127 =  27.9488845294199600508500,
    a128 = -2.85899827713502369474066,
    a129 = -8.87285693353062954433549,
    a1210=  12.3605671757943030647266,
    a1211=  0.64339274601576353035597,
    // Runge-Kutta coefficients for the final stage
    b1   =  0.05429373411656876223805,
    b6   =  4.45031289275240888144114,
    b7   =  1.89151789931450038304282,
    b8   = -5.80120396001058478146721,
    b9   =  0.31116436695781989440892,
    b10  = -0.15216094966251607855618,
    b11  =  0.20136540080403034837478,
    b12  =  0.04471061572777259051769,
    // coefficients for error estimates (3rd and 5th order)
    bhh1 =  0.24409448818897637795276,
    bhh2 =  0.73384668828161185734136,
    bhh3 =  0.02205882352941176470588,
    er1  =  0.01312004499419488073250,
    er6  = -1.22515644637620444072057,
    er7  = -0.49575894965725019152141,
    er8  =  1.66437718245498653696153,
    er9  = -0.35032884874997368168865,
    er10 =  0.33417911871301747902973,
    er11 =  0.08192320648511571246571,
    er12 = -0.02235530786388629525884,
    // coefficients for 7th order interpolation instead of the original 8th order
    d41  = -5.40685903845352664250302,
    d46  =  367.268892700041893590281,
    d47  =  154.609958204083905482676,
    d48  = -505.920283865412564024766,
    d49  =  15.5975154819608130688200,
    d410 = -26.1936204184402805956691,
    d411 = -0.74003512364122230844721,
    d412 =  1.11776539319431476294221,
    d413 = -0.33333333333333333333333,
    d51  =  6.51987095363079615048119,
    d56  = -1066.34956011730205278592,
    d57  = -351.864047514639508625601,
    d58  =  1363.51955696662884408368,
    d59  = -112.727669432657582669864,
    d510 =  159.796191868560289612921,
    d511 = -2.13865100308788816220259,
    d512 = -3.75569172113289760348584,
    d513 =  7.00000000000000000000000,
    d61  =  10.4698004763293477204238,
    d66  = -1380.01473607038123167155,
    d67  = -531.219827862514074379012,
    d68  =  1866.98964341870892451324,
    d69  = -53.3302605020547902574560,
    d610 =  82.4147560258671369782481,
    d611 =  7.38443654502992069572676,
    d612 =  0.41729908012587751149843,
    d613 = -3.11111111111111111111111,
    d71  = -16.6338582677165354330709,
    d76  =  4516.16568914956011730205,
    d77  =  1393.85185384057776465219,
    d78  = -5687.52042419481539670071,
    d79  =  473.965563750151263163661,
    d710 = -661.810776942355889724311,
    d711 = -18.0180473354013232598119,
    d712 = 0,
    d713 = 0,
    // parameters for step size selection
    fdec = 0.333, // maximum instantaneous decrease factor
    finc = 6.0,   // maximum increase factor
    safe = 0.9;   // safety factor in timestep

    T
    // persistent data (conserved between timesteps)
    *x  = state,       // x     at the beginning of the timestep
    *k1 = x  + NDIM,   // dx/dt at the beginning of the timestep
    // temporary data (used only inside this routine)
    *k2 = xt + NDIM,
    *k3 = k2 + NDIM,
    *k4 = k3 + NDIM,
    *k5 = k4 + NDIM,
    *k6 = k5 + NDIM,
    *k7 = k6 + NDIM,
    *k8 = k7 + NDIM,
    *k9 = k8 + NDIM,
    *k10= k9 + NDIM,
    *k11= k2,  // last stages reuse the memory from earlier stages
    *k12= k3,
    *k13= k4,
    *k1b= k5;
    // use the previously estimated timestep or the requested max step, whichever is shorter
    T sign = maxTimeStep >= 0 ? T(+1) : T(-1);
    T timeStep = std::fmin(maxTimeStep * sign, nextTimeStep) * sign;
    // track the number of rejected attempts and the progress in error reduction
    int nbad = 0;
    T preverr = T(INFINITY);

    // repeat until the step is accepted
    while(1) {
        if(timeStep==0 || !isFinite(timeStep))
            return 0;   // error, integration must be terminated

        // the twelve Runge-Kutta stages
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                a21 * k1[i];
        force(c2*timeStep, xt, k2, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a31*k1[i] + a32*k2[i]);
        force(c3*timeStep, xt, k3, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a41*k1[i] + a43*k3[i]);
        force(c4*timeStep, xt, k4, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a51*k1[i] + a53*k3[i] + a54*k4[i]);
        force(c5*timeStep, xt, k5, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a61*k1[i] + a64*k4[i] + a65*k5[i]);
        force(c6*timeStep, xt, k6, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a71*k1[i] + a74*k4[i] + a75*k5[i] + a76*k6[i]);
        force(c7*timeStep, xt, k7, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a81*k1[i] + a84*k4[i] + a85*k5[i] + a86*k6[i] + a87*k7[i]);
        force(c8*timeStep, xt, k8, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a91*k1[i] + a94*k4[i] + a95*k5[i] + a96*k6[i] + a97*k7[i] + a98*k8[i]);
        force(c9*timeStep, xt, k9, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a101*k1[i] + a104*k4[i] + a105*k5[i] + a106*k6[i] + a107*k7[i] +
                 a108*k8[i] + a109*k9[i]);
        force(c10*timeStep, xt, k10, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a111*k1[i] + a114*k4[i] + a115 *k5 [i] + a116*k6[i] + a117*k7[i] +
                 a118*k8[i] + a119*k9[i] + a1110*k10[i]);
        force(c11*timeStep, xt, k11, (T*)NULL);
        for(int i=0; i<NDIM; i++)
            xt[i] = x[i] + timeStep *
                (a121*k1[i] + a124*k4[i] + a125 *k5 [i] + a126 *k6 [i] + a127*k7[i] +
                 a128*k8[i] + a129*k9[i] + a1210*k10[i] + a1211*k11[i]);
        force(c12*timeStep, xt, k12, (T*)NULL);
        for(int i=0; i<NDIM; i++) {
            k1b[i] = b1*k1 [i] + b6 *k6 [i] + b7 *k7 [i] + b8*k8[i] + b9*k9[i] +
                    b10*k10[i] + b11*k11[i] + b12*k12[i];
            xt[i] = x[i] + timeStep * k1b[i];
        }

        // compute the solution at the end of the timestep and the accuracy factor
        T accFac = T(1.0);
        force(timeStep, xt, k13, &accFac);

        // error estimation
        T err5 = T(0.0), err3 = T(0.0);
        for(int i=0; i<NDIM; i++) {
            T sk = (accAbs + accRel * std::fmax(std::fabs(x[i]), std::fabs(xt[i]))) * accFac;
            if(sk==0) continue;
            err3 += pow_2( (   k1b[i] - bhh1*k1 [i] - bhh2*k9 [i] - bhh3*k12[i]) / sk);
            err5 += pow_2( (er1*k1[i] + er6 *k6 [i] + er7 *k7 [i] + er8 *k8 [i]  +
                            er9*k9[i] + er10*k10[i] + er11*k11[i] + er12*k12[i]) / sk);
        }
        T den = std::sqrt(NDIM * (err5 + T(0.01) * err3));
        T err = den==0 ? T(0) : err5 * std::fabs(timeStep) / den;

        if(!isFinite(err))
            return 0;   // error, integration must be terminated

        // computation of nextTimeStep
        T fac = std::sqrt(std::sqrt(std::sqrt(err)));  // = pow(err, 1./8);

        // step accepted if the internal error estimate is small enough
        if(err <= 1) {
            // adjust the prediction for the next timestep
            nextTimeStep = std::fabs(timeStep) * std::fmin(finc, safe/fac);
            break;
        }
        else {
            if(err > T(0.5)*preverr) {
                // the error is supposed to improve as the timestep is reduced;
                // if that's not happening, something might be wrong
                nbad++;
                if(nbad >= 2)
                    fac = T(1)/fdec;  // apply maximum reduction in timestep
                if(nbad >= 4) {
                    // no improvement likely means that something is wrong with the error estimate
                    // (e.g. one of the variables is nearly zero so that the relative error is huge);
                    // just accept the current step and try to proceed further
                    nextTimeStep = std::fabs(timeStep);
                    break;
                }
            } else
                nbad = 0;  // reset the counter of badly failed steps which didn't reduce the error
            preverr = err;
            // if the step is rejected, make it smaller
            timeStep *= std::fmax(fdec, safe/fac);
        }
    }

    // preparation of interpolation coefficients for dense output
    T   // the interpolation coefficients are stored in 'state' at different offsets
    *rcont1 = k1     + NDIM,
    *rcont2 = rcont1 + NDIM,
    *rcont3 = rcont2 + NDIM,
    *rcont4 = rcont3 + NDIM,
    *rcont5 = rcont4 + NDIM,
    *rcont6 = rcont5 + NDIM,
    *rcont7 = rcont6 + NDIM,
    *rcont8 = rcont7 + NDIM;
    for(int i=0; i<NDIM; i++) {
        rcont1[i] = x[i];
        T xd = xt[i] - x[i];   // x(t+dt) - x(t)
        rcont2[i] = xd;
        T xc = timeStep * k1[i] - xd;
        rcont3[i] = xc;
        rcont4[i] = xd - timeStep*k13[i] - xc;
        rcont5[i] = timeStep * (d41 *k1 [i] + d46 *k6 [i] + d47 *k7 [i] + d48 *k8 [i] +
                    d49*k9[i] + d410*k10[i] + d411*k11[i] + d412*k12[i] + d413*k13[i]);
        rcont6[i] = timeStep * (d51 *k1 [i] + d56 *k6 [i] + d57 *k7 [i] + d58 *k8 [i] +
                    d59*k9[i] + d510*k10[i] + d511*k11[i] + d512*k12[i] + d513*k13[i]);
        rcont7[i] = timeStep * (d61 *k1 [i] + d66 *k6 [i] + d67 *k7 [i] + d68 *k8 [i] +
                    d69*k9[i] + d610*k10[i] + d611*k11[i] + d612*k12[i] + d613*k13[i]);
        rcont8[i] = timeStep * (d71 *k1 [i] + d76 *k6 [i] + d77 *k7 [i] + d78 *k8 [i] +
                    d79*k9[i] + d710*k10[i] + d711*k11[i] + d712*k12[i] + d713*k13[i]);
        // store the new values and derivatives of x at the end of the current timestep
        x [i] = xt [i];
        k1[i] = k13[i];
    }
    return timeStep;
}

/** Dense-output interpolation of the solution within the last completed DOP853 timestep.
    \param[in]  state  is the persistent 10*NDIM storage of the integrator,
    with the dense-output coefficients filled by the last successful dop853_step;
    \param[in]  NDIM   is the size of the ODE system;
    \param[in]  prevTimeStep  is the length of the last completed timestep;
    \param[in]  timeOffset  is the time offset from the beginning of that timestep
    (must lie within the timestep, taking into account its sign - not checked here);
    \param[in]  i  is the index of the requested component of the solution vector.
    \return  the interpolated solution.
*/
template<typename T>
AGAMA_DEVICE_INLINE T dop853_dense(const T* state, int NDIM, T prevTimeStep, T timeOffset, int i)
{
    if(timeOffset == prevTimeStep)
        return state[i];  // state vector at the end of the last timestep
    if(timeOffset == 0)
        return state[i+2*NDIM];  // = rcont1[i], state vector at the beginning of the last timestep
    T p = timeOffset / prevTimeStep, q = T(1.0) - p;  // both p & q should be between 0 and 1
    // interpolation coefs rcont1..rcont8 are stored in the 'state' array at various offsets
    return   state[i+2*NDIM] + p * (state[i+3*NDIM] + q * (state[i+4*NDIM] + p * (state[i+5*NDIM] +
        q * (state[i+6*NDIM] + p * (state[i+7*NDIM] + q * (state[i+8*NDIM] + p *  state[i+9*NDIM]))))));
}


/** (Re-)initialize the persistent state of the DPRKN8 integrator from the given state vector.
    \param[in]  force1  is the first-order r.h.s. of the ODE (Force callable, see above),
    used only for the initial timestep estimate via ode_init_timestep;
    \param[in]  force2  is the second-order r.h.s. (Force2 callable:
    void force2(T t, const T x[], T d2xdt2[], T* d3xdt3, T* af));
    \param[in]  NDIM    is the full size of the ODE system (2 * numVar);
    \param[in]  stateNew  is the new state vector (NDIM elements: positions then velocities);
    \param[in]  accRel  is the tolerance parameter;
    \param[out] state   is the persistent 3*NDIM storage (x, dx/dt, d2x/dt2, and the jerk/snap/crackle
    coefficients share the remaining slots - see dprkn8_step for the layout);
    \param[in,out] nextTimeStep  is the prediction for the next timestep:
    if zero on input (the very first call), it is estimated by ode_init_timestep;
    \param[out] qold    is the persistent PI-controller state; set to 0.0001 on first call;
    \param[in,out] scratch  is a caller-provided working array of 4*NDIM elements.
*/
template<typename T, typename Force1, typename Force2>
AGAMA_DEVICE_INLINE void dprkn8_init(const Force1& force1, const Force2& force2, int NDIM,
    const T stateNew[], T accRel, T* state, T& nextTimeStep, T& qold, T* scratch)
{
    int numVar = NDIM / 2;
    for(int d=0; d<NDIM; d++)  // copy the vector x (all NDIM: positions + velocities)
        state[d] = stateNew[d];
    // obtain the derivative d2x/dt2 and the accuracy factor
    T accFac = T(1);
    force2(T(0), stateNew, /*output*/ state + 2*numVar, (T*)NULL,
        nextTimeStep==T(0) ? &accFac : (T*)NULL);
    if(nextTimeStep == T(0))  // initial timestep assignment
        nextTimeStep = ode_init_timestep(force1, NDIM, stateNew, T(0), accRel, scratch) * accFac;
    qold = T(0.0001);
}

/** Perform a single adaptive timestep of the DPRKN8 integrator
    (8th order Runge-Kutta-Nystrom with 9 force evaluations per timestep,
    6th order position / 5th order velocity dense output).
    \param[in]  force2  is the second-order r.h.s. of the ODE (Force2 callable);
    \param[in]  NDIM    is the full size of the ODE system (2 * numVar);
    \param[in]  accRel  is the relative tolerance parameter;
    \param[in,out] state  is the persistent 3*NDIM storage: on input blocks 0..2 hold
    x, dx/dt, d2x/dt2 at the beginning of the timestep (as filled by dprkn8_init or
    the previous call); on successful return they hold the values at the end of the
    accepted step, and blocks 3..5 hold d3x/dt3, d4x/dt4, d5x/dt5 (the dense-output
    Taylor coefficients for the completed step);
    \param[in,out] scratch  is a caller-provided working array of 13*numVar elements
    (where numVar = NDIM/2);
    \param[in,out] nextTimeStep  is the prediction for the next timestep, updated on return;
    \param[in,out] qold   is the persistent PI-controller state from the previous accepted step;
    \param[in]  maxTimeStep  is the upper limit on the length of the timestep.
    \return  the (signed) length of the timestep taken, or 0 on error.
*/
template<typename T, typename Force2>
AGAMA_DEVICE_INLINE T dprkn8_step(const Force2& force2, int NDIM, T accRel,
    T* state, T* scratch, T& nextTimeStep, T& qold, T maxTimeStep)
{
    if(maxTimeStep == T(0))
        return T(0);
    int numVar = NDIM / 2;
    const T
    c1  = T(1.0 / 20),
    c2  = T(1.0 / 10),
    c3  = T(3.0 / 10),
    c4  = T(1.0 / 2),
    c5  = T(7.0 / 10),
    c6  = T(9.0 / 10),
    c7  = T(1.0),
    c8  = T(1.0),
    a21 = T(1.0 / 800),
    a31 = T(1.0 / 600),
    a32 = T(1.0 / 300),
    a41 = T(9.0 / 200),
    a42 = T(-9.0 / 100),
    a43 = T(9.0 / 100),
    a51 = T(-66701.0 / 197352),
    a52 = T(28325.0 / 32892),
    a53 = T(-2665.0  / 5482),
    a54 = T(2170.0  / 24669),
    a61 = T(227015747.0 / 304251000),
    a62 = T(-54897451.0 / 30425100),
    a63 = T(12942349.0 / 10141700),
    a64 = T(-9499.0 / 304251),
    a65 = T(539.0 / 9250),
    a71 = T(-1131891597.0 / 901789000),
    a72 = T(41964921.0 / 12882700),
    a73 = T(-6663147.0 / 3220675),
    a74 = T(270954.0 / 644135),
    a75 = T(-108.0 / 5875),
    a76 = T(114.0 / 1645),
    a81 = T(13836959.0 / 3667458),
    a82 = T(-17731450.0 / 1833729),
    a83 = T(1063919505.0 / 156478208),
    a84 = T(-33213845.0 / 39119552),
    a85 = T(13335.0 / 28544),
    a86 = T(-705.0  / 14272),
    a87 = T(1645.0 / 57088),
    a91 = T(223.0  / 7938),
    a93 = T(1175.0 / 8064),
    a94 = T(925.0  / 6048),
    a95 = T(41.0   / 448),
    a96 = T(925.0  / 14112),
    a97 = T(1175.0 / 72576),
    b1  = T(223.0  / 7938),
    b3  = T(1175.0 / 8064),
    b4  = T(925.0  / 6048),
    b5  = T(41.0   / 448),
    b6  = T(925.0  / 14112),
    b7  = T(1175.0 / 72576),
    bp1 = T(223.0  / 7938),
    bp3 = T(5875.0 / 36288),
    bp4 = T(4625.0 / 21168),
    bp5 = T(41.0   / 224),
    bp6 = T(4625.0 / 21168),
    bp7 = T(5875.0 / 36288),
    bp8 = T(223.0  / 7938),
    btilde1  = T(223.0  / 7938  - 7987313.0  / 109941300),
    btilde3  = T(1175.0 / 8064  - 1610737.0  / 44674560),
    btilde4  = T(925.0  / 6048  - 10023263.0 / 33505920),
    btilde5  = T(41.0   / 448   + 497221.0   / 12409600),
    btilde6  = T(925.0  / 14112 - 10023263.0 / 78180480),
    btilde7  = T(1175.0 / 72576 - 1610737.0  / 402071040),
    bptilde1 = T(223.0  / 7938  - 7987313.0  / 109941300),
    bptilde3 = T(5875.0 / 36288 - 1610737.0  / 40207104),
    bptilde4 = T(4625.0 / 21168 - 10023263.0 / 23454144),
    bptilde5 = T(41.0   / 224   + 497221.0   / 6204800),
    bptilde6 = T(4625.0 / 21168 - 10023263.0 / 23454144),
    bptilde7 = T(5875.0 / 36288 - 1610737.0  / 40207104),
    bptilde8 = T(223.0  / 7938  + 4251941.0  / 54970650),
    bptilde9 = T(-3.0   /  20);

    T
    // persistent data (conserved between timesteps)
    *x  = state,           // x       at the beginning of the timestep
    *v  = x  + numVar,     // dx/dt   at the beginning of the timestep
    *k1 = v  + numVar,     // d2x/dt2 at the beginning of the step
    *j1 = k1 + numVar,     // d3x/dt3 at the beginning of the timestep
    // temporary data (used only inside this routine)
    *xt = scratch,
    *k2 = xt + numVar,
    *k3 = k2 + numVar,
    *k4 = k3 + numVar,
    *k5 = k4 + numVar,
    *k6 = k5 + numVar,
    *k7 = k6 + numVar,
    *k8 = k7 + numVar,
    *k9 = k8 + numVar,
    // final x and v at the end of the timestep
    *xn = k9  + numVar,
    *vn = xn  + numVar,
    *kn = vn  + numVar,
    *jn = kn  + numVar;
    // use the previously estimated timestep or the requested max step, whichever is shorter
    T sign = maxTimeStep >= T(0) ? T(+1) : T(-1);
    T timeStep = std::fmin(maxTimeStep * sign, nextTimeStep) * sign;

    // track the number of rejected attempts and the progress in error reduction
    int nbad = 0, niter = 0;
    T preverr = T(INFINITY);

    // repeat until the step is accepted
    while(true) {
        if(timeStep==T(0) || !isFinite(timeStep))
            return T(0);   // error, integration must be terminated

        // nine Runge-Kutta stages
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c1 * v[i] + timeStep *
                (a21 * k1[i]));
        force2(c1 * timeStep, xt, k2, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c2 * v[i] + timeStep *
                (a31 * k1[i] + a32 * k2[i]));
        force2(c2 * timeStep, xt, k3, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c3 * v[i] + timeStep *
                (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]));
        force2(c3 * timeStep, xt, k4, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c4 * v[i] + timeStep *
                (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]));
        force2(c4 * timeStep, xt, k5, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c5 * v[i] + timeStep *
                (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]));
        force2(c5 * timeStep, xt, k6, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c6 * v[i] + timeStep *
                (a71 * k1[i] + a72 * k2[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]));
        force2(c6 * timeStep, xt, k7, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c7 * v[i] + timeStep *
                (a81 * k1[i] + a82 * k2[i] + a83 * k3[i] + a84 * k4[i] + a85 * k5[i] + a86 * k6[i] +
                 a87 * k7[i]));
        force2(c7 * timeStep, xt, k8, (T*)NULL, (T*)NULL);
        for(int i=0; i<numVar; i++)
            xt[i] = x[i] + timeStep * (c8 * v[i] + timeStep *  // no a92 and a98
                (a91 * k1[i] + a93 * k3[i] + a94 * k4[i] + a95 * k5[i] + a96 * k6[i] + a97 * k7[i]));
        force2(c8 * timeStep, xt, k9, (T*)NULL, (T*)NULL);

        // compute variables at the end of the timestep
        for(int i=0; i<numVar; i++) {
            xn[i] = x[i] + timeStep * (v[i] + timeStep *  // no b2 and b8
                (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i] + b7 * k7[i]));
            vn[i] = v[i] + timeStep *  // no bp2
                (bp1* k1[i] + bp3* k3[i] + bp4* k4[i] + bp5* k5[i] + bp6* k6[i] + bp7* k7[i] + bp8* k8[i]);
        }

        // final evaluation of ODE at the end of the timestep and the computation of accuracy factor
        T accFac = T(1.0);
        force2(timeStep, xn, /*output*/kn, /*no 3rd deriv*/(T*)NULL, &accFac);

        // error estimation
        T err = T(0.0);
        for(int i=0; i<numVar; i++) {
            T denom = (/*accAbs +*/ accRel * std::fmax(std::fabs(x[i]), std::fabs(xn[i]))) * accFac;
            if(denom!=T(0)) {
                T xtilde = pow_2(timeStep) * (  // no btilde2,8,9
                    btilde1 * k1[i] + btilde3 * k3[i] + btilde4 * k4[i] + btilde5 * k5[i] +
                    btilde6 * k6[i] + btilde7 * k7[i]);
                err += pow_2(xtilde / denom);
            }
            denom = (/*accAbs +*/ accRel * std::fmax(std::fabs(v[i]), std::fabs(vn[i]))) * accFac;
            if(denom!=T(0)) {
                T vtilde = timeStep * (  // no bptilde2
                    bptilde1* k1[i] + bptilde3* k3[i] + bptilde4* k4[i] + bptilde5* k5[i] +
                    bptilde6* k6[i] + bptilde7* k7[i] + bptilde8* k8[i] + bptilde9* k9[i]);
                err += pow_2(vtilde / denom);
            }
        }
        err = std::sqrt(err / NDIM);

        // step estimation either by a proportional-integral (PI) controller
        const T gamma = T(0.9), qmax = T(10.0), qmin = T(0.2), beta1 = T(7./80), beta2 = T(4./80);
        T q1 = std::pow(err, beta1);
        if(err <= T(1) || ++niter >= 12) {  // step accepted
            T q = accRel * accFac==INFINITY /*ignore accuracy control*/ ? T(0) :
                std::fmin(T(1) / qmin, std::fmax(T(1) / qmax, q1 / std::pow(qold, beta2) / gamma));
            // adjust the prediction for the next timestep
            nextTimeStep = std::fabs(timeStep) / q;
            qold = err;
            break;
        }

        // same precautions as in dop853; more aggressive step reduction in case of persistent troubles
        if(err > T(0.5)*preverr) {
            // the error is supposed to improve as the timestep is reduced;
            // if that's not happening, something might be wrong
            if(++nbad >= 2)
                q1 = T(1)/qmin;  // apply maximum reduction in timestep
        } else
            nbad = 0;  // reset the counter of badly failed steps which didn't reduce the error

        // step rejected
        preverr = err;
        timeStep /= std::fmin(T(1) / qmin, q1 / gamma);
    }

    // preparation of interpolation coefficients for dense output
    for(int i=0; i<numVar; i++) {
        // compute coefficients of Taylor expansion d3x/dt3 .. d5x/dt5
        jn[i]             = ((
            -60 * (x[i] -       xn[i]) / timeStep +
            -24 *  v[i] -  36 * vn[i]) / timeStep +
            -3  * k1[i] +   9 * kn[i]) / timeStep;
        state[4*numVar+i] = ((
            -360 * (x[i] -       xn[i]) / timeStep +
            -168 *  v[i] - 192 * vn[i]) / timeStep +
            -24  * k1[i] +  36 * kn[i]) / pow_2(timeStep);
        state[5*numVar+i] = ((
            -720 * (x[i] -       xn[i]) / timeStep +
            -360 *  v[i] - 360 * vn[i]) / timeStep +
            -60  * k1[i] +  60 * kn[i]) / pow_3(timeStep);
        x [i] = xn[i];
        v [i] = vn[i];
        k1[i] = kn[i];
        j1[i] = jn[i];
    }
    return timeStep;
}

/** Dense-output interpolation of the solution within the last completed DPRKN8 timestep.
    \param[in]  state   is the persistent 3*NDIM storage of the integrator (numVar = NDIM/2),
    with blocks 0..5 holding x, dx/dt, d2x/dt2, d3x/dt3, d4x/dt4, d5x/dt5 at the end of
    the step;
    \param[in]  NDIM    is the full size of the ODE system (2 * numVar);
    \param[in]  prevTimeStep  is the signed length of the last completed timestep;
    \param[in]  timeOffset  is the time offset from the beginning of that timestep
    (must lie within the timestep);
    \param[in]  ind  is the index of the requested component (0..NDIM-1).
    \return  the interpolated solution.
*/
template<typename T>
AGAMA_DEVICE_INLINE T dprkn8_dense(const T* state, int NDIM, T prevTimeStep, T timeOffset, int ind)
{
    int numVar = NDIM / 2;
    T deltat = timeOffset - prevTimeStep;  // expected to be between -prevTimeStep and 0
    if(deltat == T(0))
        return state[ind];
    int i = ind % numVar;
    if(ind < numVar) {  // interpolate position
        return                 state[i+0*numVar] +
            deltat *          (state[i+1*numVar] +
            deltat * T(1./2) * (state[i+2*numVar] +
            deltat * T(1./3) * (state[i+3*numVar] +
            deltat * T(1./4) * (state[i+4*numVar] +
            deltat * T(1./5) *  state[i+5*numVar] ))));
    } else {  // interpolate velocity
        return                 state[i+1*numVar] +
            deltat *          (state[i+2*numVar] +
            deltat * T(1./2) * (state[i+3*numVar] +
            deltat * T(1./3) * (state[i+4*numVar] +
            deltat * T(1./4) *  state[i+5*numVar] )));
    }
}


/** Base class for numerical integrators of ODE systems.
    The task of this class is to advance the solution by one timestep at a time,
    evaluating the r.h.s. of the ODE at some intermediate times within the current timestep,
    and possibly adjusting the timestep to satisfy the accuracy requirements.
    The calling code is responsible for the overall process of integrating the ODE
    on a given time interval, keeping track of the total time, etc.
*/
class BaseOdeStepper {
public:
    virtual ~BaseOdeStepper() {};

    /** (re-)initialize the internal state from the given ODE system state */
    virtual void init(const double stateNew[]) = 0;

    /** advance the solution by one timestep.
        \param[in]  maxTimeStep is the upper limit on the length of the timestep (can have any sign);
        the actual timestep is controlled by the accuracy requirements and may be shorter.
        \return the length of the timestep taken, or zero on error
    */
    virtual double doStep(double maxTimeStep) = 0;

    /** return the interpolated solution within the last completed timestep.
        \param[in]  timeOffset  is the time offset from the beginning of the last completed step,
            and it should not exceed the length of this step (taking into account its sign).
        \param[in]  ind  is the index of the component of the solution vector;
        \return  the interpolated solution at the given time.
        \throw  std::out_of_range if the index is not in the range (0 .. N-1),
            or if the requested time offset falls outside the last completed timestep.
    */
    virtual double getSol(double timeOffset, unsigned int ind) const = 0;
};


/** 8th order Runge-Kutta integrator with 7th order interpolation for the dense output
    (modification of the original algorithm from Hairer,Norsett&Wanner, reducing the order of
    interpolation from 8 to 7 and saving 3 function evaluations per timestep) */
class OdeStepperDOP853: public BaseOdeStepper {
public:
    OdeStepperDOP853(const IOdeSystem& _odeSystem, double _accRel=1e-8, double _accAbs=0) :
        odeSystem(_odeSystem), NDIM(odeSystem.size()),
        accRel(_accRel), accAbs(_accAbs),
        prevTimeStep(0), nextTimeStep(0),
        state(NDIM * 10)  // storage for the current values and derivs of x and for 8 interpolation coefs
    {}
    virtual void init(const double stateNew[]);
    virtual double doStep(double maxTimeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    const IOdeSystem& odeSystem; ///< the interface providing the r.h.s. of the ODE
    const int NDIM;              ///< number of equations
    const double accRel, accAbs; ///< relative and absolute tolerance parameters
    double prevTimeStep;         ///< length of the last completed time step
    double nextTimeStep;         ///< predicted length of the next timestep (not the one just completed)
    std::vector<double> state;   ///< 10*NDIM values: x, dx/dt, and 8 interpolation coefs for dense output
};


/** 8th order Runge-Kutta-Nystrom scheme with nine function evaluations per timestep,
    requires a special type of ODE system that provides second time derivatives of x.
    The order of solution is 8, the order of interpolation is 6 for x, 5 for dx/dt.
*/
class OdeStepperDPRKN8: public BaseOdeStepper {
public:
    OdeStepperDPRKN8(const IOdeSystem2ndOrder& _odeSystem, double _accRel=1e-8);
    virtual void init(const double stateNew[]);
    virtual double doStep(double maxTimeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    /// the object providing the r.h.s. of the ODE
    const IOdeSystem2ndOrder& odeSystem; ///< interface for evaluating the acceleration
    const int NDIM;              ///< number of equations
    const double accRel;         ///< relative tolerance parameter
    double prevTimeStep;         ///< length of the last completed time step
    double nextTimeStep;         ///< predicted length of the next timestep (not the one just completed)
    std::vector<double> state;   ///< 3*NDIM values
    double qold;                 ///< adaptive timestep change in the previous step
};




/** Implicit Gauss-Legendre method with 3 collocation points for second-order linear ODE systems:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is a N-dimensional vector, and A, C are NxN matrices.
    It has no built-in error control, i.e. no adaptive timestepping,
    is intended for solving the variational equation during orbit integration,
    and may evolve K >= 1 independent vectors x_k simultaneously.
    The order of solution is 6, the order of interpolation is 5 for x, 4 for dx/dt.
    \tparam NDIM is the size of vector x (hence the size of the entire ODE system is
    2 NDIM * numVectors); only the cases NDIM=1,2,3 are compiled.
*/
template<int NDIM>
class Ode2StepperGL3: public BaseOdeStepper {
public:
    Ode2StepperGL3(const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int numVectors=1);
    virtual void init(const double stateNew[]);
    virtual double doStep(double timeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;
private:
    const IOdeSystem2ndOrderLinear& odeSystem;   ///< interface providing the r.h.s. of the ODE
    const unsigned int numVectors;  ///< number of independent vectors being evolved
    double prevTimeStep;  ///< length of the just completed timestep
    bool newstep;         ///< whether the extrapolation coefs are known from the previous step
    std::vector<double> state;  ///< internal state, including interpolation coefficients
};


/** Implicit Gauss-Legendre method with 4 collocation points for second-order linear ODE systems:
    d2x(t) / dt2 = A(t) x(t) + B(t) dx(t)/dt,
    where x is a N-dimensional vector, and A, C are NxN matrices.
    It has no built-in error control, i.e. no adaptive timestepping,
    is intended for solving the variational equation during orbit integration,
    and may evolve K >= 1 independent vectors x_k simultaneously.
    The order of solution is 8, the order of interpolation is 6 for x, 5 for dx/dt.
    \tparam NDIM is the size of vector x (hence the size of the entire ODE system is
    2 NDIM * numVectors); only the cases NDIM=1,2,3 are compiled.
*/
template<int NDIM>
class Ode2StepperGL4: public BaseOdeStepper {
public:
    Ode2StepperGL4(const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int numVectors=1);
    virtual void init(const double stateNew[]);
    virtual double doStep(double timeStep);
    virtual double getSol(double timeOffset, unsigned int ind) const;

private:
    const IOdeSystem2ndOrderLinear& odeSystem;   ///< interface providing the r.h.s. of the ODE
    const unsigned int numVectors;  ///< number of independent vectors being evolved
    double prevTimeStep;  ///< length of the just completed timestep
    bool newstep;         ///< whether the extrapolation coefs are known from the previous step
    std::vector<double> state;  ///< internal state, including interpolation coefficients
};

}  // namespace

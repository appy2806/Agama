#include "math_ode.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace math{

void IOdeSystem2ndOrder::eval(const double t, const double w[], double dwdt[], double* af) const
{
    const unsigned int N = size() / 2;  // dimension of position or velocity
    for(unsigned int i=0; i<N; i++)
        dwdt[i] = w[i + N];
    eval2(t, w, dwdt + N, NULL, af);
}

void IOdeSystem2ndOrderLinear::eval(const double t, const double w[], double dwdt[], double*) const
{
    // temp storage for matrices a, b
    unsigned int N = size() / 2;  // length of x and dx/dt vectors
    double *A = static_cast<double*>(alloca(N*N * sizeof(double) * 2)), *B = A + N*N;
    evalMat(t, A, B);
    for(unsigned int i=0; i<N; i++) {
        dwdt[i] = w[i+N];
        dwdt[i+N] = 0;
        for(unsigned int k=0; k<N; k++)
            dwdt[i+N] += A[i*N+k] * w[k] + B[i*N+k] * w[k+N];
    }
}

namespace {
/// adapter presenting IOdeSystem::eval as the Force callable expected by the header-only cores
struct OdeRhs {
    const IOdeSystem& sys;
    explicit OdeRhs(const IOdeSystem& _sys) : sys(_sys) {}
    void operator()(double t, const double x[], double dxdt[], double* af) const {
        sys.eval(t, x, dxdt, af);
    }
};

/// adapter presenting IOdeSystem2ndOrder::eval2 as the Force2 callable expected by the
/// header-only cores (dprkn8_step, hermite_step):
///   void force2(T t, const T x[], T d2xdt2[], T* d3xdt3, T* af)
struct OdeRhs2 {
    const IOdeSystem2ndOrder& sys;
    explicit OdeRhs2(const IOdeSystem2ndOrder& _sys) : sys(_sys) {}
    void operator()(double t, const double x[], double d2xdt2[], double* d3xdt3, double* af) const {
        sys.eval2(t, x, d2xdt2, d3xdt3, af);
    }
};
}

double initTimeStep(const IOdeSystem& odeSystem, const double x[], double accAbs, double accRel)
{
    const int NDIM = odeSystem.size();
    // temporary storage allocated on the stack
    double *scratch = static_cast<double*>(alloca(NDIM*4 * sizeof(double)));
    return ode_init_timestep(OdeRhs(odeSystem), NDIM, x, accAbs, accRel, scratch);
}

/** --- DOP853 high-accuracy Runge-Kutta integrator --- **/
// The actual math lives in the header-only cores dop853_init / dop853_step / dop853_dense
// (math_ode.h), shared with the GPU persistent orbit kernel; the class methods below are
// thin wrappers providing the storage and the IOdeSystem-based force callable.

void OdeStepperDOP853::init(const double stateNew[])
{
    double *scratch = static_cast<double*>(alloca(NDIM*4 * sizeof(double)));
    dop853_init(OdeRhs(odeSystem), NDIM, stateNew, accRel, accAbs,
        /*persistent state*/ &state[0], nextTimeStep, scratch);
}

double OdeStepperDOP853::doStep(double maxTimeStep)
{
    // temporary storage for intermediate Runge-Kutta steps
    double *xt = static_cast<double*>(alloca(NDIM*10 * sizeof(double)));
    double timeStep = dop853_step(OdeRhs(odeSystem), NDIM, accRel, accAbs,
        &state[0], xt, nextTimeStep, maxTimeStep);
    if(timeStep != 0)
        prevTimeStep = timeStep;
    return timeStep;
}

// dense output function
double OdeStepperDOP853::getSol(double timeOffset, unsigned int i) const
{
    if(i >= (unsigned int)NDIM)
        throw std::out_of_range("OdeStepperDOP853: element index out of range");
    double p = timeOffset / prevTimeStep;
    if(!(timeOffset == prevTimeStep || timeOffset == 0 || (p>=0 && 1.0-p>=0)))
        throw std::out_of_range("OdeStepperDOP853: requested time is outside the last completed timestep");
    return dop853_dense(&state[0], NDIM, prevTimeStep, timeOffset, (int)i);
}


/** --- 8th order Runge-Kutta-Nystrom integrator for second-order ODEs --- **/
// The actual math lives in the header-only cores dprkn8_init / dprkn8_step / dprkn8_dense
// (math_ode.h), shared with the GPU persistent orbit kernel; the class methods below are
// thin wrappers providing the storage and the IOdeSystem2ndOrder-based force callables.

OdeStepperDPRKN8::OdeStepperDPRKN8(const IOdeSystem2ndOrder& _odeSystem, double _accRel) :
    odeSystem(_odeSystem),
    NDIM(odeSystem.size()),
    accRel(10 * pow(_accRel, 0.9)),  // empirical approximate match to dop853's accuracy parameter
    nextTimeStep(0),
    state(NDIM * 3)
{}

void OdeStepperDPRKN8::init(const double stateNew[])
{
    double *scratch = static_cast<double*>(alloca(NDIM*4 * sizeof(double)));
    dprkn8_init(OdeRhs(odeSystem), OdeRhs2(odeSystem), NDIM, stateNew, accRel,
        &state[0], nextTimeStep, qold, scratch);
}

double OdeStepperDPRKN8::doStep(double maxTimeStep)
{
    int numVar = NDIM / 2;
    // temporary storage for intermediate Runge-Kutta steps (13*numVar elements)
    double *scratch = static_cast<double*>(alloca(numVar * 13 * sizeof(double)));
    double timeStep = dprkn8_step(OdeRhs2(odeSystem), NDIM, accRel,
        &state[0], scratch, nextTimeStep, qold, maxTimeStep);
    if(timeStep != 0)
        prevTimeStep = timeStep;
    return timeStep;
}

double OdeStepperDPRKN8::getSol(double timeOffset, unsigned int ind) const
{
    if(ind >= (unsigned int)NDIM)
        throw std::out_of_range("OdeStepperDPRKN8: element index out of range");
    return dprkn8_dense(&state[0], NDIM, prevTimeStep, timeOffset, (int)ind);
}


/** --- 4th order Hermite scheme --- **/
// The actual math lives in the header-only cores hermite_init / hermite_step / hermite_dense
// (math_ode.h), shared with the GPU persistent orbit kernel; the class methods below are
// thin wrappers providing the storage and the IOdeSystem2ndOrder-based force callables.

OdeStepperHermite::OdeStepperHermite(const IOdeSystem2ndOrder& _odeSystem, double _accRel) :
    odeSystem(_odeSystem),
    NDIM(odeSystem.size()),
    accRel(1.5 * pow(_accRel, 0.2)),  // empirical approximate match to dop853's accuracy parameter
    prevTimeStep(0),
    nextTimeStep(0),
    state(NDIM * 5)
{}

void OdeStepperHermite::init(const double stateNew[])
{
    double *scratch = static_cast<double*>(alloca(NDIM*4 * sizeof(double)));
    hermite_init(OdeRhs(odeSystem), NDIM, stateNew, accRel,
        &state[0], nextTimeStep, scratch);
}

double OdeStepperHermite::doStep(double maxTimeStep)
{
    double timeStep = hermite_step(OdeRhs2(odeSystem), NDIM, accRel,
        &state[0], nextTimeStep, maxTimeStep);
    if(timeStep != 0)
        prevTimeStep = timeStep;
    return timeStep;
}

double OdeStepperHermite::getSol(double timeOffset, unsigned int ind) const
{
    unsigned int numVar = NDIM/2;  // dimension of either coordinate or momentum vector
    int i = ind % numVar;
    if(i >= NDIM)
        throw std::out_of_range("OdeStepperHermite: element index out of range");
    double h = timeOffset / prevTimeStep;
    if(h<0 || h>1 || (prevTimeStep==0 && h!=0))
        throw std::out_of_range("OdeStepperHermite: requested time is outside the last completed timestep");
    return hermite_dense(&state[0], NDIM, prevTimeStep, timeOffset, (int)ind);
}


/** --- Integrators for second-order linear ODE systems --- */

template<int NDIM>
Ode2StepperGL3<NDIM>::Ode2StepperGL3(
    const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int _numVectors) :
    odeSystem(_odeSystem),
    numVectors(_numVectors),
    prevTimeStep(0),
    newstep(true),
    state(5 * NDIM * numVectors)
{
    if(numVectors <= 0)
        throw std::invalid_argument("Ode2StepperGL3: invalid number of vectors");
    if(odeSystem.size() != NDIM * 2)
        throw std::invalid_argument("Ode2StepperGL3: invalid size of the ODE system");
}

template<int NDIM>
void Ode2StepperGL3<NDIM>::init(const double stateNew[])
{
    state.assign(5*NDIM * numVectors, 0);
    for(unsigned int v=0; v<numVectors; v++)
        std::copy(stateNew + v * 2*NDIM, stateNew + (v+1) * 2*NDIM, state.begin() + v * 5*NDIM);
    newstep = true;
}

template<int NDIM>
double Ode2StepperGL3<NDIM>::getSol(double timeOffset, unsigned int i) const
{
    double T = timeOffset - prevTimeStep;
    // T is expected to be between -prevTimeStep and 0, i.e. interpolating on the previous timestep
    if(! ( (prevTimeStep>=0 && timeOffset>=0 && timeOffset<=prevTimeStep)
        || (prevTimeStep<=0 && timeOffset<=0 && timeOffset>=prevTimeStep) ) )
        throw std::out_of_range("Ode2StepperGL3: requested time is outside the last completed timestep");
    unsigned int vec = i / (2*NDIM);  // index of the requested vector
    if(vec >= numVectors)
        throw std::out_of_range("Ode2StepperGL3: element index out of range");
    i -= vec * 2*NDIM;          // index of the requested element in the given vector
    if(T==0)  // fast track: no interpolation needed, just copy the element of the state vector
        return state[vec * 5*NDIM + i];
    unsigned int d = i % NDIM;  // index of the requested dimension in either position or velocity
    const double *x = &state[vec * 5*NDIM + d], *xdot = x+NDIM,  // state vector (pos, vel)
        *p = xdot+NDIM, *q = p+NDIM, *r = q+NDIM;  // higher-order interpolation coefficients
    if(i == d)  // d-th component of x(t)
        return (*x) + T * ((*xdot) + T * ((*p) + T * ((*q) + T * (*r))));
    else        // d-th component of dx(t)/dt
        return (*xdot) + T * (2 * (*p) + T * (3 * (*q) + T * 4 * (*r)));
}

template<int NDIM>
double Ode2StepperGL3<NDIM>::doStep(double dt)
{
    double dt2 = dt*dt, idt = 1./dt, idt2 = idt*idt;

    // collocation points are the nodes of Gauss-Legendre quadrature of degree 3:
    // this gives the highest possible accuracy of the whole scheme (it has order 6)
    static const double h0 = 0.1127016653792583, h1 = 0.5, h2 = 1-h0,
    // various combinations of collocation points
    su0 = 0.5 *  h0 * h0,
    sv0 = su0 *  h0 * -2 / 3,
    sw0 = sv0 * (0.25 * h0 - h1),
    su1 = 0.5 *  h1 * h1,
    sv1 = su1 * (h1 / 3 - h0),
    sw1 = su1 *  h1 / 6 * (4 * h0 - h1),
    su2 = 0.5 *  h2 * h2,
    sv2 = su2 * (h2 / 3 - h0),
    sw2 = su2 * (h2 / 3 * (0.5 * h2 - h0 - h1) + h0 * h1),
    du0 = h0,
    dv0 = h0  * -h0 / 2,
    dw0 = h0  * h0 * (h1 / 2 - h0 / 6),
    du1 = h1,
    dv1 = h1  * (h1 / 2 - h0),
    dw1 = h1  * h1 * (h0 / 2 - h1 / 6),
    du2 = h2,
    dv2 = h2  * (h2 / 2 - h0),
    dw2 = h2  * (h2 * (h2 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    mvu = 1 / (h1 - h0),
    mwu = 1 / (h2 - h0) / (h2 - h1),
    mwv = 1 / (h2 - h1),
    mpu = 1./2,
    mpv = 1./2 * (1 - h0),
    mpw = 1./2 * (1 - h0) * (1 - h1),
    mqv = 1./6,
    mqw = 1./6 * (2 - h0 - h1),
    mrw = 1./12;

    // collect the values of the matrices A and B in the RHS of the ODE at the collocation points h_k,
    // corresponding to times  t_0 + h_k * dt, where 0<=h<=1 is the time normalized to timestep;
    // values of A(h_k) and B(h_k) are stored as flattened arrays in row-major order
    double a0[NDIM * NDIM], a1[NDIM * NDIM], a2[NDIM * NDIM];
    double b0[NDIM * NDIM], b1[NDIM * NDIM], b2[NDIM * NDIM];
    odeSystem.evalMat(dt * h0, a0, b0);
    odeSystem.evalMat(dt * h1, a1, b1);
    odeSystem.evalMat(dt * h2, a2, b2);

    // The second derivative of d-th component x_d is approximated by a quadratic polynomial in h,
    // with coefficients u_d, v_d, w_d to be determined:
    // x_d''(h) = u_d + (h-h0) * (v_d + (h-h1) * w_d)                                       [*]
    // accordingly, the first derivative and the value of x_d itself are
    // x_d'(h)  = x_d'(0) + dt * (pu(h) * u_d + pv(h) * v_d + pw(h) * w_d),
    // x_d(h)   = x_d(0)  + dt * h * x_d'(0) + dt^2 * (su(h) * u_d + sv(h) * v_d + sw(h) * w_d),
    // where pu, pv, pw, su, sv, sw are known polynomials of h
    // (pre-computed for all collocation points h_k and for the final point h=1).
    // The RHS of our ODE prescribes that
    // x_d''(h_k) = \sum_{j=1}^{NDIM} A_{dj}(h_k) x_j(h_k) + B_{dj}(h_k) x_j'(h_j).
    // We iteratively recompute the coefficients u,v,w:
    // first evaluate x_d(h_k) with the current values of these coefs at the given point h_k,
    // then compute the rhs x_d''(h_k),
    // then use the relation [*] to update u, v, and w - it is written in such a way that
    // for each point h_k we update only one of these coefs (h0 -> u, h1 -> v, h2 -> w);
    // the updated coefs are then used for the next point h_{k+1}, and then the whole iteration
    // cycle is repeated a few times (three seems to be enough).

    // storage for temporary arrays defined below
    const int tempSize = NDIM * 8;
    double *temp = static_cast<double*>(alloca(tempSize * sizeof(double)));
    // approximate values of x and xdot at the current iteration and the current collocation point h_k
    double *xh = temp, *dh = temp + NDIM;
    // pre-computed values of x_prev[d] + x_prev'[d] * dt * h_{0,1,2}
    double *x0 = temp + 2*NDIM, *x1 = temp + 3*NDIM, *x2 = temp + 4*NDIM;
    // polynomial coefficients to be calculated
    double *u = temp + 5*NDIM, *v = temp + 6*NDIM, *w = temp + 7*NDIM;

    // process each vector independently
    for(unsigned int vec=0; vec<numVectors; vec++) {
        // values of x and xdot at the beginning of the current timestep
        double *x = &state[vec * 5*NDIM], *xdot = x + NDIM;
        // values of higher-order interpolation coefficients at the beginning of the timestep
        double *p = xdot + NDIM, *q = p + NDIM, *r = q + NDIM;
        for(int d=0; d<NDIM; d++) {
            x0[d] = x[d] + xdot[d] * dt * h0;
            x1[d] = x[d] + xdot[d] * dt * h1;
            x2[d] = x[d] + xdot[d] * dt * h2;
            // predict the polynomial coefs in x'' from the previous timestep
            double Q = dt*q[d], R = dt2*r[d];
            u[d] = 2  * p[d] + 6*h0 * Q + 12*h0*h0 * R;
            v[d] = 6  * Q + 12*(h0+h1) * R;
            w[d] = 12 * R;
        }

        // iteratively find the coefficients u_d, v_d, w_d for each component of vector x
        const int NUMITER = newstep ? 6 : 3;
        for(int i=0; i<NUMITER; i++) {
            // consider each collocation point h_k in turn, and use it to update u, v, w (one by one)

            // h_0 is used for calculating u
            // first predict the values of x and xdot at time h_0
            for(int d=0; d<NDIM; d++) {
                xh[d] = x0  [d] + dt2 * (su0 * u[d] + sv0 * v[d] + sw0 * w[d]);
                dh[d] = xdot[d] + dt  * (du0 * u[d] + dv0 * v[d] + dw0 * w[d]);
            }
            // then compute the RHS at time h_0 and calculate u
            for(int d=0; d<NDIM; d++) {
                // second derivative of d'th component (RHS of the ODE) at the current point
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a0[d * NDIM + j] * xh[j] + b0[d * NDIM + j] * dh[j];
                u[d] = rhs;
            }

            // h_1 is used for calculating v (third derivative of d'th component)
            for(int d=0; d<NDIM; d++) {
                xh[d] = x1  [d] + dt2 * (su1 * u[d] + sv1 * v[d] + sw1 * w[d]);
                dh[d] = xdot[d] + dt  * (du1 * u[d] + dv1 * v[d] + dw1 * w[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a1[d * NDIM + j] * xh[j] + b1[d * NDIM + j] * dh[j];
                v[d] = (rhs - u[d]) * mvu;
            }

            // finally, h_2 is used for calculating w (fourth derivative)
            for(int d=0; d<NDIM; d++) {
                xh[d] = x2  [d] + dt2 * (su2 * u[d] + sv2 * v[d] + sw2 * w[d]);
                dh[d] = xdot[d] + dt  * (du2 * u[d] + dv2 * v[d] + dw2 * w[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a2[d * NDIM + j] * xh[j] + b2[d * NDIM + j] * dh[j];
                w[d] = (rhs - u[d]) * mwu - v[d] * mwv;
            }
        }

        // now that the coefficients u, v, w are known,
        // compute the new values of x_d, xdot_d at the end of timestep (h=1)
        // and the coefficients p, q, r for interpolating the solution at any moment of time
        // inside the completed timestep
        for(int d=0; d<NDIM; d++) {
            double
            P = mpw * w[d] + mpv * v[d] + mpu * u[d],
            Q = mqw * w[d] + mqv * v[d],
            R = mrw * w[d];
            p[d] = P;
            q[d] = Q * idt;
            r[d] = R * idt2;
            x[d]    += (    P - 2 * Q + 3 * R) * dt2 + xdot[d] * dt;
            xdot[d] += (2 * P - 3 * Q + 4 * R) * dt;
        }
    }

    prevTimeStep = dt;
    newstep = false;
    return dt;
}

template<int NDIM>
Ode2StepperGL4<NDIM>::Ode2StepperGL4(
    const IOdeSystem2ndOrderLinear& _odeSystem, unsigned int _numVectors) :
    odeSystem(_odeSystem),
    numVectors(_numVectors),
    prevTimeStep(0),
    newstep(true),
    state(6*NDIM * numVectors)
{
    if(numVectors <= 0)
        throw std::invalid_argument("Ode2StepperGL4: invalid number of vectors");
    if(odeSystem.size() != NDIM * 2)
        throw std::invalid_argument("Ode2StepperGL4: invalid size of the ODE system");
}

template<int NDIM>
void Ode2StepperGL4<NDIM>::init(const double stateNew[])
{
    state.assign(6*NDIM * numVectors, 0);
    for(unsigned int v=0; v<numVectors; v++)
        std::copy(stateNew + v * 2*NDIM, stateNew + (v+1) * 2*NDIM, state.begin() + v * 6*NDIM);
    newstep = true;
}

template<int NDIM>
double Ode2StepperGL4<NDIM>::getSol(double timeOffset, unsigned int i) const
{
    double T = timeOffset - prevTimeStep;
    // T is expected to be between -prevTimeStep and 0, i.e. interpolating on the previous timestep
    if(! ( (prevTimeStep>=0 && timeOffset>=0 && timeOffset<=prevTimeStep)
        || (prevTimeStep<=0 && timeOffset<=0 && timeOffset>=prevTimeStep) ) )
        throw std::out_of_range("Ode2StepperGL4: requested time is outside the last completed timestep");
    unsigned int vec = i / (2*NDIM);  // index of the requested vector
    if(vec >= numVectors)
        throw std::out_of_range("Ode2StepperGL4: element index out of range");
    i -= vec * 2*NDIM;          // index of the requested element in the given vector
    if(T==0)  // fast track: no interpolation needed, just copy the element of the state vector
        return state[vec * 6*NDIM + i];
    unsigned int d = i % NDIM;  // index of the requested dimension in either position or velocity
    const double *x = &state[vec * 6*NDIM + d], *xdot = x+NDIM,  // state vector (pos, vel)
        *p = xdot+NDIM, *q = p+NDIM, *r = q+NDIM, *s = r+NDIM;   // higher-order interpolation coefs
    if(i == d)  // d-th component of x(t)
        return (*x) + T * ((*xdot) + T * ((*p) + T * ((*q) + T * ((*r) + T * (*s)))));
    else        // d-th component of dx(t)/dt
        return (*xdot) + T * (2 * (*p) + T * (3 * (*q) + T * 4 * ((*r) + T * 5 * (*s))));
}

template<int NDIM>
double Ode2StepperGL4<NDIM>::doStep(double dt)
{
    double dt2 = dt*dt, dt3 = dt2*dt, idt = 1./dt, idt2 = idt*idt, idt3 = idt2*idt;

    // collocation points are the nodes of Gauss-Legendre quadrature of degree 4:
    // this gives the highest possible accuracy of the whole scheme (it has order 8)
    static const double h0 = 0.069431844202973713, h1 = 0.33000947820757187, h2 = 1-h1, h3 = 1-h0,
    // various combinations of collocation points
    su0 = 0.5 * h0 * h0,
    sv0 = su0 * h0 * -2/3,
    sw0 = su0 * h0 * (h1*2/3 - h0/6),
    sz0 = su0 * h0 * (h0 * (h1+h2)/6 - h0*h0/15 - h1*h2*2/3),
    su1 = 0.5 * h1 * h1,
    sv1 = su1 *(h1/3 - h0),
    sw1 = su1 * h1 * (h0*2/3 - h1/6),
    sz1 = su1 * h1 * (h1 * (h0+h2)/6 - h1*h1/15 - h0*h2*2/3),
    su2 = 0.5 * h2 * h2,
    sv2 = su2 *(h2/3 - h0),
    sw2 = su2 *(h2/3 * (h2/2 - h0 - h1) + h0*h1),
    sz2 = su2 * h2 * (h2 * (h0+h1)/6 - h2*h2/15 - h0*h1*2/3),
    su3 = 0.5 * h3 * h3,
    sv3 = su3 *(h3/3 - h0),
    sw3 = su3 *(h3/3 * (h3/2 - h0 - h1) + h0*h1),
    sz3 = su3 *(h3 * (h3 * (h3*0.6-h0-h1-h2)/6 + (h0*h1+h1*h2+h0*h2)/3) - h0*h1*h2),
    du0 = h0,
    dv0 = h0  * -h0 / 2,
    dw0 = h0  * h0 * (h1 / 2 - h0 / 6),
    dz0 = h0  * h0 * (h0 * (-h0 / 12 + h1 / 6 + h2 / 6) - h1 * h2 / 2),
    du1 = h1,
    dv1 = h1  * (h1 / 2 - h0),
    dw1 = h1  * h1 * (h0 / 2 - h1 / 6),
    dz1 = h1  * h1 * (h1 * (-h1 / 12 + h0 / 6 + h2 / 6) - h0 * h2 / 2),
    du2 = h2,
    dv2 = h2  * (h2 / 2 - h0),
    dw2 = h2  * (h2 * (h2 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    dz2 = h2  *  h2 * (h2 * (-h2 / 12 + h0 / 6 + h1 / 6) - h0 * h1 / 2),
    du3 = h3,
    dv3 = h3  * (h3 / 2 - h0),
    dw3 = h3  * (h3 * (h3 / 3 - h1 / 2 - h0 / 2) + h0 * h1),
    dz3 = h3  * (h3 * (h0*h1/2 + h0*h2/2 + h1*h2/2 - h3 * (h0/3 + h1/3 + h2/3 - h3/4)) - h0*h1*h2),
    mvu = 1 / (h1 - h0),
    mwu = 1 / (h2 - h0) / (h2 - h1),
    mwv = 1 / (h2 - h1),
    mzu = 1 / (h3 - h0) / (h3 - h1) / (h3 - h2),
    mzv = 1 / (h3 - h1) / (h3 - h2),
    mzw = 1 / (h3 - h2),
    mpu = 1./2,
    mpv = 1./2 * (1 - h0),
    mpw = 1./2 * (1 - h0) * (1 - h1),
    mpz = 1./2 * (1 - h0) * (1 - h1) * (1 - h2),
    mqv = 1./6,
    mqw = 1./6 * (2 - h0 - h1),
    mqz = 1./6 * ((1-h0) * (1-h1) + (1-h1) * (1-h2) + (1-h2) * (1-h0)),
    mrw = 1./12,
    mrz = 1./12 * (3 - h0 - h1 - h2),
    msz = 1./20;

    // collect the values of the matrices A and B in the RHS of the ODE at the collocation points h_k,
    // corresponding to times  t_0 + h_k * dt, where 0<=h<=1 is the time normalized to timestep;
    // values of A(h_k) and B(h_k) are stored as flattened arrays in row-major order
    double a0[NDIM * NDIM], a1[NDIM * NDIM], a2[NDIM * NDIM], a3[NDIM * NDIM];
    double b0[NDIM * NDIM], b1[NDIM * NDIM], b2[NDIM * NDIM], b3[NDIM * NDIM];
    odeSystem.evalMat(dt * h0, a0, b0);
    odeSystem.evalMat(dt * h1, a1, b1);
    odeSystem.evalMat(dt * h2, a2, b2);
    odeSystem.evalMat(dt * h3, a3, b3);

    // The second derivative of d-th component x_d is approximated by a cubic polynomial in h,
    // with coefficients u_d, v_d, w_d, z_d to be determined:
    // x_d''(h) = u_d + (h-h0) * (v_d + (h-h1) * (w_d + (h-h2) * z_d))
    // the procedure is analogous to the 6-th order method.

    // storage for temporary arrays defined below
    const int tempSize = NDIM * 10;
    double *temp = static_cast<double*>(alloca(tempSize * sizeof(double)));
    // approximate values of x and xdot at the current iteration and the current collocation point h_k
    double *xh = temp, *dh = temp + NDIM;
    // pre-computed values of x_prev[d] + x_prev'[d] * dt * h_{0,1,2,3}
    double *x0 = temp + 2*NDIM, *x1 = temp + 3*NDIM, *x2 = temp + 4*NDIM, *x3 = temp + 5*NDIM;
    // polynomial coefficients to be calculated
    double *u = temp + 6*NDIM, *v = temp + 7*NDIM, *w = temp + 8*NDIM, *z = temp + 9*NDIM;

    // process each vector independently
    for(unsigned int vec=0; vec<numVectors; vec++) {
        // values of x and xdot at the beginning of the current timestep
        double *x = &state[vec * 6*NDIM], *xdot = x + NDIM;
        // values of higher-order interpolation coefficients at the beginning of the timestep
        double *p = xdot + NDIM, *q = p + NDIM, *r = q + NDIM, *s = r + NDIM;
        for(int d=0; d<NDIM; d++) {
            x0[d] = x[d] + xdot[d] * dt * h0;
            x1[d] = x[d] + xdot[d] * dt * h1;
            x2[d] = x[d] + xdot[d] * dt * h2;
            x3[d] = x[d] + xdot[d] * dt * h3;
            // predict the polynomial coefs in x'' from the previous timestep
            double Q = dt*q[d], R = dt2*r[d], S = dt3*s[d];
            u[d] = 2  * p[d] + 6*h0 * Q + 12*h0*h0 * R + 20*h0*h0*h0 * S;
            v[d] = 6  * Q + 12*(h0+h1) * R + 20*(h0*h0+h0*h1+h1*h1) * S;
            w[d] = 12 * R + 20*(h0+h1+h2) * S,
            z[d] = 20 * S;
        }

        // iteratively find the coefficients u_d, v_d, w_d, z_d for each component of vector x
        const int NUMITER = newstep ? 8 : 4;
        for(int iter=0; iter<NUMITER; iter++) {
            // consider each collocation point h_k in turn, and use it to update u, v, w, z (one by one)

            // h_0 is used for calculating u
            // first predict the values of x and xdot at time h_0
            for(int d=0; d<NDIM; d++) {
                xh[d] = x0  [d] + dt2 * (su0 * u[d] + sv0 * v[d] + sw0 * w[d] + sz0 * z[d]);
                dh[d] = xdot[d] + dt  * (du0 * u[d] + dv0 * v[d] + dw0 * w[d] + dz0 * z[d]);
            }
            // then compute the RHS at time h_0 and calculate u
            for(int d=0; d<NDIM; d++) {
                // second derivative of d-th component (RHS of the ODE) at the current point
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a0[d * NDIM + j] * xh[j] + b0[d * NDIM + j] * dh[j];
                u[d] = rhs;
            }

            // h_1 is used for calculating v
            for(int d=0; d<NDIM; d++) {
                xh[d] = x1  [d] + dt2 * (su1 * u[d] + sv1 * v[d] + sw1 * w[d] + sz1 * z[d]);
                dh[d] = xdot[d] + dt  * (du1 * u[d] + dv1 * v[d] + dw1 * w[d] + dz1 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a1[d * NDIM + j] * xh[j] + b1[d * NDIM + j] * dh[j];
                v[d] = (rhs - u[d]) * mvu;
            }

            // h_2 is used for calculating w
            for(int d=0; d<NDIM; d++) {
                xh[d] = x2  [d] + dt2 * (su2 * u[d] + sv2 * v[d] + sw2 * w[d] + sz2 * z[d]);
                dh[d] = xdot[d] + dt  * (du2 * u[d] + dv2 * v[d] + dw2 * w[d] + dz2 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a2[d * NDIM + j] * xh[j] + b2[d * NDIM + j] * dh[j];
                w[d] = (rhs - u[d]) * mwu - v[d] * mwv;
            }

            // finally, h_3 is used for calculating z
            for(int d=0; d<NDIM; d++) {
                xh[d] = x3  [d] + dt2 * (su3 * u[d] + sv3 * v[d] + sw3 * w[d] + sz3 * z[d]);
                dh[d] = xdot[d] + dt  * (du3 * u[d] + dv3 * v[d] + dw3 * w[d] + dz3 * z[d]);
            }
            for(int d=0; d<NDIM; d++) {
                double rhs = 0;
                for(int j=0; j<NDIM; j++)
                    rhs += a3[d * NDIM + j] * xh[j] + b3[d * NDIM + j] * dh[j];
                z[d] = (rhs - u[d]) * mzu - v[d] * mzv - w[d] * mzw;
            }
        }

        // now that the coefficients u, v, w, z are known,
        // compute the new values of x_d, x_d' at the end of timestep (h=1)
        // and the coefficients p, q, r, s for interpolating the solution at any moment of time
        // inside the completed timestep
        for(int d=0; d<NDIM; d++) {
            double
            P = mpz * z[d] + mpw * w[d] + mpv * v[d] + mpu * u[d],
            Q = mqz * z[d] + mqw * w[d] + mqv * v[d],
            R = mrz * z[d] + mrw * w[d],
            S = msz * z[d];
            p[d] = P;
            q[d] = Q * idt;
            r[d] = R * idt2;
            s[d] = S * idt3;
            x[d]    += (    P - 2 * Q + 3 * R - 4 * S) * dt2 + xdot[d] * dt;
            xdot[d] += (2 * P - 3 * Q + 4 * R - 5 * S) * dt;
        }
    }

    prevTimeStep = dt;
    newstep = false;
    return dt;
}

// compile the template instantiation for 1d,2d and 3d systems only
template class Ode2StepperGL3<1>;
template class Ode2StepperGL3<2>;
template class Ode2StepperGL3<3>;
template class Ode2StepperGL4<1>;
template class Ode2StepperGL4<2>;
template class Ode2StepperGL4<3>;

}  // namespace

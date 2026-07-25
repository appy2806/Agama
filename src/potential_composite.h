/** \file    potential_composite.h
    \brief   Composite density and potential classes and various modifiers
    \author  Eugene Vasiliev
    \date    2014-2022
*/
#pragma once
#include "potential_base.h"
#include "smart.h"
#include "math_spline.h"
#include "gpu_policy.h"   // agama::forall for templated batch evaluators (Tier 1)

namespace potential{

// =====================================================================
// Tier 1 leaf math for UniformAcceleration (single source: the same body is
// called by the CPU evalCar below and by evalmanyCarT/evalmanyPhiAccCarT).
// Unlike every other Tier 1 leaf in this codebase, UniformAcceleration is
// genuinely time-dependent (the acceleration components are time-dependent
// CubicSpline evaluations) -- but the spline evaluation itself is NOT part of
// the leaf: the caller evaluates accx(time)/accy(time)/accz(time) ONCE on the
// host (CubicSpline::operator() is not AGAMA_DEVICE-callable; math_spline.h is
// out of scope for this migration) and passes the resulting three scalars in.
// The leaf itself is then an ordinary POD-in/POD-out function like any other.
// =====================================================================

/** UniformAcceleration:  Phi(x,y,z) = x*dx + y*dy + z*dz  for a given (already
    time-evaluated) acceleration-derived gradient (dx,dy,dz) = (-ax(t),-ay(t),-az(t)).
    grad (length 3, nullable) is just (dx,dy,dz) -- constant in space. */
template<typename T>
AGAMA_DEVICE_INLINE void uniform_acceleration_eval(T dx, T dy, T dz, T x, T y, T z,
    T* potential, T* grad /*[3]*/)
{
    if(potential)
        *potential = x*dx + y*dy + z*dz;
    if(grad) {
        grad[0] = dx;
        grad[1] = dy;
        grad[2] = dz;
    }
}

/** Interface for composite Density or Potential classes:
    these may contain multiple individual components, as in CompositeDensity or Composite potential,
    or wrap a single object and modify its behaviour, as in Shifted/Rotating/etc.
*/
template<class BaseDensityOrPotential> class BaseComposite {
public:
    virtual ~BaseComposite() {}
    virtual unsigned int size() const = 0;
    virtual shared_ptr<const BaseDensityOrPotential> component(unsigned int index) const = 0;
};

/** A collection of several density objects */
class CompositeDensity: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /** construct from the provided array of components */
    CompositeDensity(const std::vector<PtrDensity>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    /** sum up masses of all components */
    virtual double totalMass() const;

    /** joins the names of all components */
    virtual std::string name() const;

    virtual unsigned int size() const { return components.size(); }
    virtual PtrDensity component(unsigned int index) const { return components.at(index); }

private:
    std::vector<PtrDensity> components;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** A collection of several potential objects */
class Composite: public BasePotential, public BaseComposite<BasePotential> {
public:
    /** construct from the provided array of components */
    Composite(const std::vector<PtrPotential>& _components);

    /** provides the 'least common denominator' for the symmetry degree */
    virtual coord::SymmetryType symmetry() const;

    /** sum up masses of all components */
    virtual double totalMass() const;

    /** joins the names of all components */
    virtual std::string name() const;

    virtual unsigned int size() const { return components.size(); }
    virtual PtrPotential component(unsigned int index) const { return components.at(index); }

private:
    std::vector<PtrPotential> components;
    std::vector<char> componentTypes;
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Time-dependent potential represented by a collection of other potentials,
    either piecewise-constant or linearly interpolated in time */
class Evolving: public BasePotentialCar, public BaseComposite<BasePotential> {
public:
    Evolving(const std::vector<double> timestamps,
        const std::vector<PtrPotential> instances,
        bool interpLinear=false);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Evolving"; }
    /// provide access to underlying potential instances (but not their associated timestamps)
    virtual unsigned int size() const { return instances.size(); }
    virtual PtrPotential component(unsigned int index) const { return instances.at(index); }
private:
    /// array of time stamps for a time-dependent potential
    std::vector<double> timestamps;
    /// array of potentials corresponding to each moment of time (possibly just one)
    std::vector<PtrPotential> instances;
    /// use linear or nearest-point interpolation of a time-dependent potential
    bool interpLinear;
    /// highest symmetry level shared by all instances of the potential
    coord::SymmetryType sym;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &pos, double time) const;
};


/** Time-dependent but spatially uniform acceleration arising from a non-inertial reference frame */
class UniformAcceleration: public BasePotentialCar {
public:
    /// initialize from a triplet of splines representing time-dependent acceleration
    UniformAcceleration(
        const math::CubicSpline& _accx,
        const math::CubicSpline& _accy,
        const math::CubicSpline& _accz)
    :
        accx(_accx), accy(_accy), accz(_accz) {}

    virtual coord::SymmetryType symmetry() const { return coord::ST_NONE; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "UniformAcceleration"; }

    /** Tier 1 batch evaluator: Phi at N Cartesian positions via the
        uniform_acceleration_eval leaf, for a single given `time` shared by the
        whole batch (mirrors BaseDensity::evalmanyDensityCar's single-scalar-time
        convention). The three CubicSpline lookups happen ONCE here, host-side,
        before the forall; NOT wired into the GPU dispatch tables in
        potential_gpu.cpp / potential_descriptor.h -- see the class-level note
        below for why. add=true accumulates into phi[] (composite support). */
    template<typename T, class Policy>
    inline void evalmanyCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* phi, double time = 0, bool add = false) const
    {
        const T dx = static_cast<T>(-accx(time)),
                dy = static_cast<T>(-accy(time)),
                dz = static_cast<T>(-accz(time));
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            T pot;
            uniform_acceleration_eval(dx, dy, dz, x, y, z, &pot, (T*)NULL);
            phi[i] = add ? phi[i] + pot : pot;
        });
    }

    /** Fused batch evaluator: Phi (optional) + Cartesian acceleration a = -grad
        Phi = (ax(t),ay(t),az(t)) -- spatially uniform, so every point gets the
        same acceleration; the leaf is still called per-point for a single
        source of the Phi expression. add=true accumulates. */
    template<typename T, class Policy>
    inline void evalmanyPhiAccCarT(Policy pol, std::size_t N,
        const T* xyz, /*out, nullable*/ T* phi, /*out length 3N*/ T* acc,
        double time = 0, bool add = false) const
    {
        const T dx = static_cast<T>(-accx(time)),
                dy = static_cast<T>(-accy(time)),
                dz = static_cast<T>(-accz(time));
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            const T x = xyz[i*3+0], y = xyz[i*3+1], z = xyz[i*3+2];
            T pot, grad[3];
            uniform_acceleration_eval(dx, dy, dz, x, y, z, phi ? &pot : (T*)NULL, grad);
            if(phi) phi[i] = add ? phi[i] + pot : pot;
            acc[i*3+0] = add ? acc[i*3+0] - grad[0] : -grad[0];
            acc[i*3+1] = add ? acc[i*3+1] - grad[1] : -grad[1];
            acc[i*3+2] = add ? acc[i*3+2] - grad[2] : -grad[2];
        });
    }

    /** Batch density evaluator, present so that this class satisfies the same
        three-method contract as every other GPU-dispatchable potential (the
        AGAMA_GPU_POT_LIST X-macro in potential_gpu.cpp instantiates all three).

        Phi = x*dx + y*dy + z*dz is linear, so its Hessian -- and hence its
        Laplacian -- vanishes identically, and the density is exactly zero
        everywhere and at every time. That is not an approximation or a
        placeholder: it is the same answer the CPU path gives, since evalCar
        clears deriv2 and BasePotential::densityCar reads the density off the
        Hessian trace. A uniform acceleration represents an external tidal field
        whose source lies outside the model, so it carries no local mass. */
    template<typename T, class Policy>
    inline void evalmanyDensCarT(Policy pol, std::size_t N,
        const T* xyz, /*out*/ T* rho, double /*time*/ = 0, bool add = false) const
    {
        (void)xyz;
        agama::forall(pol, N, [=] AGAMA_DEVICE (std::size_t i) {
            rho[i] = add ? rho[i] : T(0);
        });
    }

private:
    const math::CubicSpline accx, accy, accz;

    // NOTE on GPU dispatch status. This class IS registered in
    // potential_gpu.cpp's AGAMA_GPU_POT_LIST, so the batch entry points
    // (evalPotentialGPU / evalForceGPU / evalDensityGPU, and their Python
    // device=/dtype= surface) handle it correctly at any time: those carry a
    // `time` argument that is shared by the whole batch, and the three splines
    // are evaluated ONCE host-side above before the forall -- so the answer is
    // exact, not a t=0 approximation.
    //
    // It is deliberately still ABSENT from potential_descriptor.h's
    // GpuPotTag/GpuPotTerm, which is what the orbit kernel evaluates. That
    // kernel sees a different t at every RK stage, so representing this class
    // there needs the three CubicSplines resident on the device rather than a
    // host-side snapshot of scalars -- the same missing piece that keeps
    // time-VARYING modifier chains off the orbit path (constant ones are
    // supported; see buildGpuPotDesc's requireTimeIndependent flag). Until that
    // lands, an orbit in a potential containing a UniformAcceleration component
    // fails closed with NotImplementedError rather than silently freezing time.

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        double dx = -accx(time), dy = -accy(time), dz = -accz(time);
        double grad[3];
        uniform_acceleration_eval(dx, dy, dz, pos.x, pos.y, pos.z,
            potential, deriv ? grad : (double*)NULL);
        if(deriv) {
            deriv->dx = grad[0];
            deriv->dy = grad[1];
            deriv->dz = grad[2];
        }
        if(deriv2)
            coord::clear(*deriv2);
    }
};


// =====================================================================
// GPU representation of a modifier chain (Tier 1 modifiers)
// =====================================================================

/** The complete GPU-side representation of an arbitrarily deep chain of
    Shifted / Tilted / Rotating / Scaled modifiers wrapped around one concrete
    potential: a similarity transform of the query position plus a scalar
    rescaling of the outputs. Thirteen numbers, independent of chain depth.

    Each of the four modifiers maps a query at position x, in its own external
    frame, to a query at A*x + b in the frame of the object it wraps, and
    rescales the result by a constant k:

      Shifted (center c) : A = I,      b = -c,  k = 1,      s = 1
      Tilted  (Euler)    : A = R,      b = 0,   k = 1,      s = 1
      Rotating (angle q) : A = Rz(q),  b = 0,   k = 1,      s = 1
      Scaled  (a, L)     : A = (1/L)I, b = 0,   k = a * 1/L, s = 1/L

    where every A is a uniform scale times a rotation. Similarity transforms are
    closed under composition, so a whole chain collapses exactly to a single
    (M, off, kphi, sc) with M = sc * (orthogonal), and for the innermost object:

      Phi_outer (x) = kphi * Phi_inner(M x + off)
      grad_outer(x) = kphi * M^T * grad_inner(M x + off)      [chain rule]
      rho_outer (x) = kphi * sc^2 * rho_inner(M x + off)      [Laplacian of the above]

    This is what makes modifiers cheap on GPU: no per-modifier kernel and no
    scratch buffers -- the transform is folded into the by-value descriptor term
    and applied by the same thread that evaluates the leaf. `sc` is carried
    explicitly rather than recovered as cbrt(det M) so the density factor costs
    no extra device arithmetic.

    Parameters are stored fp64 and cast once per kernel instantiation, exactly
    like the rest of GpuPotTerm (hard constraint #4). */
template<typename T>
struct GpuPotXform {
    T mat[9];  ///< linear part M, row-major: x_inner[i] = sum_j mat[3*i+j] * x[j] + off[i]
    T off[3];  ///< translation part of the position map
    T kphi;    ///< output factor: Phi_outer = kphi * Phi_inner
    T sc;      ///< the uniform scale s in M = s*R; density picks up a further s^2
};

/** Reset a transform to the identity (no modifiers). */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_xform_identity(GpuPotXform<T>& f)
{
    f.mat[0] = 1; f.mat[1] = 0; f.mat[2] = 0;
    f.mat[3] = 0; f.mat[4] = 1; f.mat[5] = 0;
    f.mat[6] = 0; f.mat[7] = 0; f.mat[8] = 1;
    f.off[0] = f.off[1] = f.off[2] = 0;
    f.kphi = 1;
    f.sc   = 1;
}

/** Map an external position to the innermost object's frame: out = M*x + off. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_xform_pos(const GpuPotXform<T>& f, T x, T y, T z, /*out*/ T out[3])
{
    out[0] = f.mat[0] * x + f.mat[1] * y + f.mat[2] * z + f.off[0];
    out[1] = f.mat[3] * x + f.mat[4] * y + f.mat[5] * z + f.off[1];
    out[2] = f.mat[6] * x + f.mat[7] * y + f.mat[8] * z + f.off[2];
}

/** Map a gradient-like vector back to the external frame: out = kphi * M^T * v.
    Used for the acceleration, which transforms with the transpose of M (chain
    rule) rather than with M itself -- getting this backwards is silent and
    wrong for any chain containing a rotation, so it lives in one function. */
template<typename T>
AGAMA_DEVICE_INLINE void gpu_xform_vec(const GpuPotXform<T>& f, const T v[3], /*out*/ T out[3])
{
    out[0] = f.kphi * (f.mat[0] * v[0] + f.mat[3] * v[1] + f.mat[6] * v[2]);
    out[1] = f.kphi * (f.mat[1] * v[0] + f.mat[4] * v[1] + f.mat[7] * v[2]);
    out[2] = f.kphi * (f.mat[2] * v[0] + f.mat[5] * v[1] + f.mat[8] * v[2]);
}

/** Host-side composition: fold one more modifier stage, which sits INSIDE
    everything already accumulated in `f`, into `f`.

    Call order therefore runs from the outermost modifier to the innermost, i.e.
    in the same direction as descending the wrapper chain from the object the
    user holds down to the concrete potential. Given the accumulated map
    x -> M x + off and the new stage's x -> A x + b, the composite is
    A(M x + off) + b, hence M <- A*M and off <- A*off + b. */
inline void gpuXformComposeInner(GpuPotXform<double>& f,
    const double A[9], const double b[3], double k, double s)
{
    double M[9], o[3];
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++)
            M[3*i+j] = A[3*i+0] * f.mat[0*3+j] + A[3*i+1] * f.mat[1*3+j] + A[3*i+2] * f.mat[2*3+j];
        o[i] = A[3*i+0] * f.off[0] + A[3*i+1] * f.off[1] + A[3*i+2] * f.off[2] + b[i];
    }
    for(int i=0; i<9; i++)
        f.mat[i] = M[i];
    for(int i=0; i<3; i++)
        f.off[i] = o[i];
    f.kphi *= k;
    f.sc   *= s;
}


// four kinds of modifiers, which can be applied to density or potential classes
// (unfortunately, some code duplication is inevitable here)
template<class BaseDensityOrPotential> class Shifted;
template<class BaseDensityOrPotential> class Tilted;
template<class BaseDensityOrPotential> class Rotating;
template<class BaseDensityOrPotential> class Scaled;


/** Modifier of any density profile adding an arbitrary, possibly time-dependent offset */
template<> class Shifted<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and splines representing time-dependent offsets
    Shifted(const PtrDensity& _dens,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        dens(_dens), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual coord::SymmetryType symmetry() const
    { return isUnknown(dens->symmetry()) ? coord::ST_UNKNOWN : coord::ST_NONE; }  // in general...
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Shifted"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return dens->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding an arbitrary, possibly time-dependent offset */
template<> class Shifted<BasePotential>: public BasePotentialCar, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and splines representing time-dependent offsets
    Shifted(const PtrPotential& _pot,
        const math::CubicSpline& _centerx,
        const math::CubicSpline& _centery,
        const math::CubicSpline& _centerz)
    :
        pot(_pot), centerx(_centerx), centery(_centery), centerz(_centerz) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual coord::SymmetryType symmetry() const
    { return isUnknown(pot->symmetry()) ? coord::ST_UNKNOWN : coord::ST_NONE; }  // in general...
    virtual std::string name() const { return Shifted<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

    /** Export this modifier's own stage of the GPU transform chain, evaluated at
        `time`: the position map x -> A*x + b applied before handing the query to
        the wrapped object, the output factor k, and the uniform scale s.
        See GpuPotXform above for how the stages compose. All four modifiers
        expose this same signature; the descriptor builder in
        potential_descriptor.h walks the chain and calls it on each wrapper.

        A shift is a pure translation: A = I, b = -center(time), k = s = 1. */
    void gpuXformStage(double time, /*out*/ double A[9], double b[3], double& k, double& s) const
    {
        A[0] = 1; A[1] = 0; A[2] = 0;
        A[3] = 0; A[4] = 1; A[5] = 0;
        A[6] = 0; A[7] = 0; A[8] = 1;
        b[0] = -centerx(time);
        b[1] = -centery(time);
        b[2] = -centerz(time);
        k = 1;
        s = 1;
    }

    /** True if gpuXformStage() returns the same stage for every `time`, so the
        stage can be folded once into a by-value descriptor. A caller that
        cannot re-evaluate the splines per step (the orbit kernel, which sees a
        different t at every RK stage) must refuse a chain where this is false
        rather than silently freeze it at one time. */
    bool gpuXformConstant() const
    { return centerx.isConstant() && centery.isConstant() && centerz.isConstant(); }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent offsets of the potential center from origin
    const math::CubicSpline centerx, centery, centerz;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        pot->eval(coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)),
            potential, deriv, deriv2, time);
    }

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return pot->density(
            coord::PosCar(pos.x-centerx(time), pos.y-centery(time), pos.z-centerz(time)), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return densityCar(toPosCar(pos), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile tilting the coordinate axes by a triplet of Euler angles */
template<> class Tilted<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and a triplet of Euler angles
    Tilted(const PtrDensity& _dens, double alpha, double beta, double gamma) :
        dens(_dens), orientation(alpha, beta, gamma) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual double enclosedMass(const double radius) const { return dens->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Tilted"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// transformation between input/output (external) and intrinsic coordinate systems
    const coord::Orientation orientation;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    { return dens->density(orientation.toRotated(pos), time); }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return dens->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return dens->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile tilting the coordinate axes by a triplet of Euler angles */
template<> class Tilted<BasePotential>: public BasePotentialCar, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and a triplet of Euler angles
    Tilted(const PtrPotential& _pot, double alpha, double beta, double gamma) :
        pot(_pot), orientation(alpha, beta, gamma) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual double enclosedMass(const double radius) const { return pot->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return Tilted<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }  // should check if index==0?

    /** GPU transform stage (see Shifted<BasePotential>::gpuXformStage): a pure
        rotation by the stored Euler angles. `orientation.mat` is row-major and is
        exactly what Orientation::toRotated applies, so it IS the stage matrix A. */
    void gpuXformStage(double /*time*/, /*out*/ double A[9], double b[3], double& k, double& s) const
    {
        for(int i=0; i<9; i++)
            A[i] = orientation.mat[i];
        b[0] = b[1] = b[2] = 0;
        k = 1;
        s = 1;
    }

    /// Euler angles are fixed at construction, so this stage never varies with time
    bool gpuXformConstant() const { return true; }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// transformation between input/output (external) and intrinsic coordinate systems
    const coord::Orientation orientation;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
    {
        pot->eval(orientation.toRotated(pos), potential, deriv, deriv2, time);
        if(deriv)  *deriv  = orientation.fromRotated(*deriv);
        if(deriv2) *deriv2 = orientation.fromRotated(*deriv2);
    }

    virtual double densityCar(const coord::PosCar &pos, double time) const
    { return pot->density(orientation.toRotated(pos), time); }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return pot->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return pot->density(orientation.toRotated(toPosCar(pos)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile adding a rotation about the Z axis
    by an angle that varies smoothly with time (or is constant) */
template<> class Rotating<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given potential and a time-dependent rotation angle given by a spline
    Rotating(const PtrDensity& _dens, const math::CubicSpline& _angle) :
        dens(_dens), angle(_angle) {}

    virtual double totalMass() const { return dens->totalMass(); }
    virtual double enclosedMass(const double radius) const { return dens->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Rotating"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent rotation angle
    const math::CubicSpline angle;

    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return dens->density(coord::PosCyl(pos.R, pos.z, pos.phi - angle(time)), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return dens->density(coord::PosSph(pos.r, pos.theta, pos.phi - angle(time)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding a rotation about the Z axis
    by an angle that varies smoothly with time (or is constant) */
template<> class Rotating<BasePotential>: public BasePotential, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and a time-dependent rotation angle given by a spline
    Rotating(const PtrPotential& _pot, const math::CubicSpline& _angle) :
        pot(_pot), angle(_angle) {}

    virtual double totalMass() const { return pot->totalMass(); }
    virtual double enclosedMass(const double radius) const { return pot->enclosedMass(radius); }
    virtual coord::SymmetryType symmetry() const;
    virtual std::string name() const { return Rotating<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

    /** GPU transform stage (see Shifted<BasePotential>::gpuXformStage): a rotation
        about the z axis by angle(time). The matrix is the one applied inline by
        evalCar below -- x_inner = (x*ca + y*sa, y*ca - x*sa, z) -- i.e. the
        rotation by -angle, not +angle; the sign lives here and nowhere else. */
    void gpuXformStage(double time, /*out*/ double A[9], double b[3], double& k, double& s) const
    {
        double sa, ca;
        math::sincos(angle(time), sa, ca);
        A[0] =  ca; A[1] = sa; A[2] = 0;
        A[3] = -sa; A[4] = ca; A[5] = 0;
        A[6] =   0; A[7] =  0; A[8] = 1;
        b[0] = b[1] = b[2] = 0;
        k = 1;
        s = 1;
    }

    /// a constant rotation angle is equivalent to a Tilted modifier and is time-independent
    bool gpuXformConstant() const { return angle.isConstant(); }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent rotation angle
    const math::CubicSpline angle;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;

    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    { return pot->density(coord::PosCyl(pos.R, pos.z, pos.phi - angle(time)), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    { return pot->density(coord::PosSph(pos.r, pos.theta, pos.phi - angle(time)), time); }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};


/** Modifier of any density profile adding a time-dependent modulation in amplitude and length scale */
template<> class Scaled<BaseDensity>: public BaseDensity, public BaseComposite<BaseDensity> {
public:
    /// initialize from the given density and two splines representing time-dependent amplitude and scale
    Scaled(const PtrDensity& _dens, const math::CubicSpline& _ampl, const math::CubicSpline& _scale) :
        dens(_dens), ampl(_ampl), scale(_scale) {}

    virtual double totalMass() const
    { return dens->totalMass() * ampl(0); }  // no way to specify time here, use t=0
    virtual double enclosedMass(const double radius) const
    { return dens->enclosedMass(radius / scale(0) ) * ampl(0); }  // same here - evaluated at t=0
    virtual coord::SymmetryType symmetry() const { return dens->symmetry(); }
    virtual std::string name() const { return myName() + " " + dens->name(); }
    static std::string myName() { return "Scaled"; }
    virtual unsigned int size() const { return 1; }
    virtual PtrDensity component(unsigned int) const { return dens; }  // should check if index==0?

private:
    /// the instance of the actual density
    const PtrDensity dens;

    /// time-dependent amplitude (mass normalization)
    const math::CubicSpline ampl;

    /// time-dependent length scale factor
    const math::CubicSpline scale;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosCar(pos.x * s, pos.y * s, pos.z * s), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosCyl(pos.R * s, pos.z * s, pos.phi), time);
    }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * dens->density(coord::PosSph(pos.r * s, pos.theta, pos.phi), time);
    }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

/** Modifier of any potential profile adding a time-dependent modulation in amplitude and length scale */
template<> class Scaled<BasePotential>: public BasePotential, public BaseComposite<BasePotential> {
public:
    /// initialize from the given potential and two splines representing time-dependent amplitude and scale
    Scaled(const PtrPotential& _pot, const math::CubicSpline& _ampl, const math::CubicSpline& _scale) :
        pot(_pot), ampl(_ampl), scale(_scale) {}

    virtual double totalMass() const
    { return pot->totalMass() * ampl(0); }  // no way to specify time here, use t=0
    virtual double enclosedMass(const double radius) const
    { return pot->enclosedMass(radius / scale(0) ) * ampl(0); }  // same here - evaluated at t=0
    virtual coord::SymmetryType symmetry() const { return pot->symmetry(); }
    virtual std::string name() const { return Scaled<BaseDensity>::myName() + " " + pot->name(); }
    virtual unsigned int size() const { return 1; }
    virtual PtrPotential component(unsigned int) const { return pot; }

    /** GPU transform stage (see Shifted<BasePotential>::gpuXformStage): an isotropic
        contraction of the query position by s = 1/scale(time), with the potential
        rescaled by ampl(time)*s. This is the only stage with k != 1 and s != 1, so
        it is the only one that makes the acceleration pick up a second factor of s
        (via M^T) and the density a third (via sc^2) -- which is exactly the
        as1/as2/as3 ladder that evalCar/densityCar below apply by hand.

        The a == 0 shortcut in evalCar (skip the wrapped potential entirely) is
        reproduced by the kphi == 0 branch in gpu_term_phi_acc, not here: kphi
        simply comes out zero. */
    void gpuXformStage(double time, /*out*/ double A[9], double b[3], double& k, double& s) const
    {
        s = 1 / scale(time);
        A[0] = s; A[1] = 0; A[2] = 0;
        A[3] = 0; A[4] = s; A[5] = 0;
        A[6] = 0; A[7] = 0; A[8] = s;
        b[0] = b[1] = b[2] = 0;
        k = ampl(time) * s;
    }

    /// both the amplitude and the length scale must be time-independent
    bool gpuXformConstant() const { return ampl.isConstant() && scale.isConstant(); }

private:
    /// the instance of the actual potential
    const PtrPotential pot;

    /// time-dependent amplitude
    const math::CubicSpline ampl;

    /// time-dependent length scale factor
    const math::CubicSpline scale;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const;

    virtual double densityCar(const coord::PosCar &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosCar(pos.x * s, pos.y * s, pos.z * s), time);
    }

    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosCyl(pos.R * s, pos.z * s, pos.phi), time);
    }

    virtual double densitySph(const coord::PosSph &pos, double time) const
    {
        double s = 1 / scale(time);
        return s*s*s * ampl(time) * pot->density(coord::PosSph(pos.r * s, pos.theta, pos.phi), time);
    }

    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const;
};

}  // namespace potential

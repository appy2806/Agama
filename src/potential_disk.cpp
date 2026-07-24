/*
This is a new implementation of GalPot written by Eugene Vasiliev, 2015-2017.

The original GalPot code:
Copyright Walter Dehnen, 1996-2004
e-mail:   walter.dehnen@astro.le.ac.uk
address:  Department of Physics and Astronomy, University of Leicester
          University Road, Leicester LE1 7RH, United Kingdom

Version 0.0    15. July      1997
Version 0.1    24. March     1998
Version 0.2    22. September 1998
Version 0.3    07. June      2001
Version 0.4    22. April     2002
Version 0.5    05. December  2002
Version 0.6    05. February  2003
Version 0.7    23. September 2004
Version 0.8    24. June      2005
*/

#include "potential_disk.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential{

// Tier 1 GPU migration note: the 5 concrete radial/vertical functor classes
// (DiskDensityRadialExp, DiskDensityRadialRichExp, DiskDensityVerticalExp,
// DiskDensityVerticalIsothermal, DiskDensityVerticalThin) that used to live in
// this file's anonymous namespace have moved to potential_disk.h (no longer
// anonymous), alongside the AGAMA_DEVICE_INLINE leaf functions their
// evalDeriv methods now thin-wrap. This lets DiskAnsatz::gpuDesc()
// dynamic_cast the stored radialFnc/verticalFnc against these types to build
// the tagged descriptor consumed by the GPU batch path -- an anonymous-
// namespace type has no linkage visible outside this translation unit, so it
// could not be dynamic_cast-recognized from potential_disk.h. The formulas
// themselves are unchanged (only relocated); DiskAnsatz::evalCyl and
// DiskAnsatz::densityCyl below still call the functors' evalDeriv virtuals
// exactly as before.

namespace{  // internal

/** integrand for computing the total mass:  2pi R Sigma(R); x=R/scaleRadius */
class DiskDensityRadialRichExpIntegrand: public math::IFunctionNoDeriv {
public:
    DiskDensityRadialRichExpIntegrand(const DiskParam& _params): params(_params) {};
private:
    const DiskParam params;
    virtual double value(double x) const {
        if(x==0 || x==1) return 0;
        double Rrel = x/(1-x);
        return x / pow_3(1-x) *
            exp(-params.innerCutoffRadius/params.scaleRadius/Rrel - math::pow(Rrel, 1/params.sersicIndex)
                +params.modulationAmplitude*cos(Rrel));
    }
};

}  // internal ns

/** helper routine to create an instance of radial density function */
math::PtrFunction createRadialDiskFnc(const DiskParam& params) {
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Disk scale radius cannot be <=0");
    if(params.innerCutoffRadius<0)
        throw std::invalid_argument("Disk inner cutoff radius cannot be <0");
    if(params.sersicIndex<=0)
        throw std::invalid_argument("Disk Sersic index must be positive");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0 && params.sersicIndex==1)
        return math::PtrFunction(new DiskDensityRadialExp(params));
    else
        return math::PtrFunction(new DiskDensityRadialRichExp(params));
}

/** helper routine to create an instance of vertical density function */
math::PtrFunction createVerticalDiskFnc(const DiskParam& params) {
    if(params.scaleHeight>0)
        return math::PtrFunction(new DiskDensityVerticalExp(params.scaleHeight));
    if(params.scaleHeight<0)
        return math::PtrFunction(new DiskDensityVerticalIsothermal(-params.scaleHeight));
    else
        return math::PtrFunction(new DiskDensityVerticalThin());
}

double DiskParam::mass() const
{
    if(modulationAmplitude==0) {  // have an analytic expression
        if(innerCutoffRadius==0)
            return M_PI * pow_2(scaleRadius) * surfaceDensity * math::gamma(2*sersicIndex+1);
        else if(sersicIndex==1) {
            double p = sqrt(innerCutoffRadius / scaleRadius);
            return 4*M_PI * pow_2(scaleRadius) * surfaceDensity *
                p * (p * math::besselK(0, 2*p) + math::besselK(1, 2*p));
        }
    }
    return 2*M_PI * pow_2(scaleRadius) * surfaceDensity *
        math::integrate(DiskDensityRadialRichExpIntegrand(*this), 0, 1, 1e-6);
}

double DiskDensity::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    double h;
    verticalFnc->evalDeriv(pos.z, NULL, NULL, &h);
    return radialFnc->value(pos.R) * h;
}

double DiskAnsatz::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    double h, H, Hp, f, fp, fpp, r=sqrt(pow_2(pos.R) + pow_2(pos.z));
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    return f*h + (pos.z!=0 ? 2*fp*(H+pos.z*Hp)/r : 0) + fpp*H;
}

void DiskAnsatz::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const
{
    double r = sqrt(pow_2(pos.R) + pow_2(pos.z));
    double h=0, H=0, Hp=0, f=0, fp=0, fpp=0;
    bool deriv1 = deriv!=NULL || deriv2!=NULL;  // compute derivatives of f and H only if necessary
    verticalFnc->evalDeriv(pos.z, &H, deriv1? &Hp : NULL, deriv2? &h   : NULL);
    radialFnc  ->evalDeriv(r,     &f, deriv1? &fp : NULL, deriv2? &fpp : NULL);
    f  *= 4*M_PI;
    fp *= 4*M_PI;
    fpp*= 4*M_PI;
    double rinv = r>0 ? 1./r : 1.;  // if r==0, avoid indeterminacy in 0/0
    double Rr   = pos.R * rinv;
    double zr   = pos.z * rinv;
    if(potential) {
        *potential = f==0 ? 0 : f * H;
    }
    if(deriv) {
        deriv->dR = H * Rr * fp;
        deriv->dz = H * zr * fp + Hp * f;
        deriv->dphi=0;
    }
    if(deriv2) {
        deriv2->dR2 = H * (fpp * pow_2(Rr) + fp * rinv * pow_2(zr));
        deriv2->dz2 = H * (fpp * pow_2(zr) + fp * rinv * pow_2(Rr)) + fp * Hp * zr * 2 + f * h;
        deriv2->dRdz= H * Rr * zr * (fpp - fp * rinv) + fp * Hp * Rr;
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

} // namespace

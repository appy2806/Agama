/** \name   test_density_grid.cpp
    \author Eugene Vasiliev
    \date   2017-2026

    Test the spatial density discretization scheme for Schwarzschild modelling
*/
#include "galaxymodel_densitygrid.h"
#include "potential_ferrers.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <iostream>

// mass, radius and axis ratios of triaxial Ferrers model
const double
    mass     = 2.345,
    radius   = 0.8765,
    axisYtoX = 0.8,
    axisZtoX = 0.6,
    rmult    = radius * cbrt(axisYtoX * axisZtoX);
// number of shells
const int NR = 10;
// radii of Ferrers model enclosing mass equal to 0.1, 0.2, ..., 1
const double radii[NR] = {
    rmult * 0.2939973372408013,
    rmult * 0.3795586772117003,
    rmult * 0.4447632539265056,
    rmult * 0.5013744085912341,
    rmult * 0.5540282084616279,
    rmult * 0.6055625378937006,
    rmult * 0.6585714934216137,
    rmult * 0.7167206042341939,
    rmult * 0.7885680919310256,
    rmult * 1.0 };

bool check(const galaxymodel::TargetDensityClassic<1>& grid, double x, double y, double z,
    int c1, int c2, int c3, int c4)
{
    const int N=grid.numValues(), P=(N-1) / NR;
    double p[3] = {x, y*axisYtoX, z*axisZtoX}, r=sqrt(x*x+y*y+z*z);
    std::vector<double> v(N);
    grid.eval(p, &v[0]);
    double sum = 0.;
    int s=0;
    while(s<NR-1 && radii[s]<r)  ++s;
    int i1 = c1 + s*P+1, i2 = c2 + s*P+1, i3 = c3 + s*P+1, i4 = c4 + s*P+1,
        i5 = std::max(0, i1-P), i6 = std::max(0, i2-P), i7 = std::max(0, i3-P), i8 = std::max(0, i4-P);
    for(int i=0; i<N; i++) {
        if((i==i1||i==i2||i==i3||i==i4||i==i5||i==i6||i==i7||i==i8) ^ (v[i]!=0))
            return false;
        sum += v[i];
    }
    return fabs(sum-1.)<1e-14;
}

bool test(const galaxymodel::BaseTargetDensity& grid)
{
    const int N=grid.numValues();
    std::vector<double> v(N);
    bool ok=true;
    for(int i=0; i<N; i++) {
        std::string str = grid.coefName(i);  // has the form "x=... y=... z=..."
        // parse the coordinates, and also slightly reduce the numbers
        // to avoid falling out of the grid for the outermost shell
        double p[3] = {
            utils::toDouble(str.substr(str.find("x=")+2)) * 0.999999,
            utils::toDouble(str.substr(str.find("y=")+2)) * 0.999999,
            utils::toDouble(str.substr(str.find("z=")+2)) * 0.999999 };
        grid.eval(p, &v[0]);
        // check that only the indicated basis function is (close to) unity at this point,
        // and all others are nearly zero (up to float-to-string conversion roundoff)
        for(int j=0; j<N; j++)
            if(fabs(v[j]-(i==j)) > 1e-4) {
                ok=false;
                std::cout << str << " : " << i << ", " << j << " => " << v[j] << '\n';
            }
    }
    return ok;
}

int main()
{
    potential::Ferrers dens(mass, radius, axisYtoX, axisZtoX);
    // triaxial schemes with the same axis ratio as the density model
    std::vector<double> rad(radii, radii + NR);
    galaxymodel::TargetDensityClassic<0> cl0t(4, rad, axisYtoX, axisZtoX);
    galaxymodel::TargetDensityClassic<1> cl1t(4, rad, axisYtoX, axisZtoX);
    galaxymodel::TargetDensitySphHarm    sh1t(0, 0, rad, axisYtoX, axisZtoX);
    galaxymodel::TargetDensitySphHarm    sh1T(4, 4, rad, axisYtoX, axisZtoX);
    // spherical schemes with a somewhat larger radius equal to the major axis of the density model
    math::blas_dmul(radius/rmult, rad);
    galaxymodel::TargetDensityClassic<0> cl0s(4, rad);
    galaxymodel::TargetDensityClassic<1> cl1s(4, rad);
    galaxymodel::TargetDensitySphHarm    sh1s(0, 0, rad);
    galaxymodel::TargetDensitySphHarm    sh1S(4, 4, rad);
    galaxymodel::TargetDensityCylindrical<0> cy0s(0, rad, rad);
    galaxymodel::TargetDensityCylindrical<0> cy0S(4, rad, rad);
    galaxymodel::TargetDensityCylindrical<1> cy1s(0, rad, rad);
    bool ok = true;
    ok &= check(cl1t, 0.51, 0.01, 0.50, 15, 16, 44, 49);
    ok &= check(cl1t, 0.01, 0.51, 0.50, 23, 24, 28, 29);
    ok &= check(cl1t, 0.01, 0.50, 0.51, 55, 56, 24, 29);
    ok &= check(cl1t, 0.50, 0.51, 0.01, 35, 36,  4,  9);
    ok &= check(cl1t, 0.50, 0.50, 0.51, 58, 59, 39, 60);
    ok &= check(cl1t, 0.51, 0.50, 0.50, 18, 19, 59, 60);
    ok &= check(cl1t, 0.50, 0.51, 0.50, 38, 39, 19, 60);
    ok &= check(cl1t, 0.10, 0.10, 0.11, 58, 59, 39, 60);
    ok &= test (cl1t);
    ok &= test (cl0t);

    // when the grid in Classic and SphHarm schemes is aligned with the shape of the ellipsoidal model,
    // integration in radius becomes exact (by virtue of the input density being polynomial function),
    // and there is no variation of density in angles when expressed as a function of ellipsoidal radius,
    // so the result should be very precise
    std::vector<double>
    masses = cl0t.computeDensityProjection(dens);
    double sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 1e-11;
    masses = cl1t.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 1e-11;
    masses = sh1t.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 1e-15;  // expect errors at the level of machine precision
    masses = sh1T.computeDensityProjection(dens);
    // for a non-spherical SphHarm, the mass is obtained by summing up only the values of the l=0 term
    sum = std::accumulate(masses.begin(), masses.begin() + NR + 1, 0.);
    ok &= fabs(sum / mass - 1) < 1e-15;
    // and remaining (l>0) terms should all be zero
    sum = std::accumulate(masses.begin() + NR + 1, masses.end(), 0.);
    ok &= sum == 0;

    // when the grid is not of the same shape as the density model, the errors are larger,
    // because the integrand is only piecewise-polynomial and not aligned with the grid boundaries
    masses = cl0s.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 1e-6;
    masses = cl1s.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 1e-6;

    masses = sh1s.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 2e-6;
    masses = sh1S.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.begin() + NR + 1, 0.);
    ok &= fabs(sum / mass - 1) < 2e-6;
    // remaining higher-order (l>0) terms should be non-negligible
    ok &= fabs(std::accumulate(masses.begin() + NR + 1, masses.end(), 0.)) > 1e-4;

    masses = cy0s.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 2e-7;
    masses = cy1s.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum / mass - 1) < 2e-7;
    // same considerations for non-axisymmetric Cylindrical
    masses = cy0S.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.begin() + pow_2(NR), 0.);
    ok &= fabs(sum / mass - 1) < 2e-7;
    ok &= fabs(std::accumulate(masses.begin() + pow_2(NR), masses.end(), 0.)) > 1e-4;

    std::vector<galaxymodel::StorageNumT> output(sh1t.numCoefs());
    {
        orbit::OrbitIntegrator<coord::Car> orbint(dens);
        orbint.init(coord::PosVelCar(0.7, 0, 0, 0, 0.2, 0.3));
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(
            new galaxymodel::RuntimeFncTarget(orbint, sh1t, &output.front())));
        orbint.run(3);
    }
    sum = std::accumulate(output.begin(), output.end(), 0.);
    // error should be at the level of floating-point precision of StorageNumT
    ok &= fabs(sum - 1) < (sizeof(galaxymodel::StorageNumT) == sizeof(double) ? 1e-15 : 1e-7);

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}

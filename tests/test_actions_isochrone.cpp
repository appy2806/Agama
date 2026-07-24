/** \file    test_actions_isochrone.cpp
    \author  Eugene Vasiliev
    \date    February 2016

    This test checks the correctness of (exact) action-angle determination for Isochrone potential
    (and additionally for an arbitrary spherical potential)
*/
#include "potential_analytic.h"
#include "actions_isochrone.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include "debug_utils.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>

// whether to do performance test
//#define PERFTEST

bool test_isochrone(const coord::PosVelCyl& initial_conditions, const char* title)
{
    const bool output = utils::verbosityLevel >= utils::VL_VERBOSE; // whether to write a text file
    const double epsr = 2e-4;  // accuracy of comparison for radial action found with different methods
    const double epsa = 1e-7;  // accuracy of action conservation along the orbit for each method
    const double epst = 1e-9;  // accuracy of reverse transformation (pv=>aa=>pv) for isochrone
    const double epss = 1e-7;  // accuracy of reverse transformation for spherical a/a mapping
    const double epsi = 1e-6;  // accuracy of reverse transformation for interpolated spherical mapping
    const double epsf = 1e-7;  // accuracy of frequency determination
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    const double total_time=50;// integration time
    const double timestep=1./8;// sampling rate of trajectory
    std::cout << "\033[1;37m"<<title<<"\033[0m\n";
    potential::Isochrone pot(M, b);
    orbit::OrbitIntParams params;
    params.accuracy = 1e-15;
    std::vector< std::pair<coord::PosVelCar, double> > traj = orbit::integrateTraj(
        toPosVelCar(initial_conditions), total_time, timestep, pot, /*Omega*/0, params);
    actions::ActionFinderSpherical actGrid(pot);  // interpolation-based action finder/mapper
    actions::ActionStat statI, statS, statF, statG;
    actions::Actions acI, acF, acS, acG;
    actions::Angles  anI, anF, anS, anG;
    actions::Frequencies frI, frF, frS, frG, frIinv, frSinv;
    math::Averager statfrIr, statfrIz, statH, statE;
    actions::Angles aoldF(0,0,0), aoldI(0,0,0), aoldS(0,0,0);
    bool anglesMonotonic= true;   // angle determination is reasonable
    bool reversible_iso = true;   // forward-reverse transform for isochrone gives the original point
    bool reversible_sph = true;   // same for spherical a/a finder/mapper
    bool reversible_grid= true;   // same for grid-interpolated spherical a/a finder/mapper
    std::ofstream strm;
    if(output) {
        strm.open(("test_actions_isochrone_"+std::string(title)+".dat").c_str());
        strm << std::setprecision(15);
    }
    double ifd = 1e-5;
    for(size_t i=0; i<traj.size(); i++) {
        coord::PosVelCyl point = toPosVelCyl(traj[i].first);
        statE.add(totalEnergy(pot, point));
        point.phi = math::wrapAngle(point.phi);
        actions::evalIsochrone(M, b, point, &acI, &anI, &frI);
        actions::evalSpherical(pot,  point, &acS, &anS, &frS);
        actions::evalAxisymFudge(pot,point, &acF, &anF, &frF, ifd);
        actGrid.eval(point, &acG, &anG, &frG);
        statH.add(actions::computeHamiltonianSpherical(pot, acI));  // find H(J)
        statI.add(acI);
        statF.add(acF);
        statS.add(acS);
        statG.add(acG);
        statfrIr.add(frI.Omegar);
        statfrIz.add(frI.Omegaz);
        actions::Angles anewF, anewI, anewS;
        anewF.thetar   = math::unwrapAngle(anF.thetar,   aoldF.thetar);
        anewF.thetaz   = math::unwrapAngle(anF.thetaz,   aoldF.thetaz);
        anewF.thetaphi = math::unwrapAngle(anF.thetaphi, aoldF.thetaphi);
        anewI.thetar   = math::unwrapAngle(anI.thetar,   aoldI.thetar);
        anewI.thetaz   = math::unwrapAngle(anI.thetaz,   aoldI.thetaz);
        anewI.thetaphi = math::unwrapAngle(anI.thetaphi, aoldI.thetaphi);
        anewS.thetar   = math::unwrapAngle(anS.thetar,   aoldS.thetar);
        anewS.thetaz   = math::unwrapAngle(anS.thetaz,   aoldS.thetaz);
        anewS.thetaphi = math::unwrapAngle(anS.thetaphi, aoldS.thetaphi);
        anglesMonotonic &= i==0 || (
            anewI.thetar >= aoldI.thetar && anewS.thetar >= aoldS.thetar &&
           (anewF.thetar >= aoldF.thetar || acF.Jr<1e-10) &&
            anewI.thetaz >= aoldI.thetaz && anewS.thetaz >= aoldS.thetaz &&
            anewF.thetaz >= aoldF.thetaz &&
            math::sign(acI.Jphi) * anewI.thetaphi >= math::sign(acI.Jphi) * aoldI.thetaphi &&
            math::sign(acS.Jphi) * anewS.thetaphi >= math::sign(acS.Jphi) * aoldS.thetaphi &&
            math::sign(acF.Jphi) * anewF.thetaphi >= math::sign(acF.Jphi) * aoldF.thetaphi);
        aoldI = anewI;
        aoldS = anewS;
        aoldF = anewF;
        // inverse transformation for spherical potential
        coord::PosVelCyl pinv = actions::mapSpherical(pot, actions::ActionAngles(acS, anS), &frSinv);
        reversible_sph &= equalPosVel(pinv, point, epss) && 
            math::fcmp(frS.Omegar, frSinv.Omegar, epss) == 0 &&
            math::fcmp(frS.Omegaz, frSinv.Omegaz, epss) == 0 &&
            math::fcmp(frS.Omegaphi, frSinv.Omegaphi, epss) == 0;

        // inverse transformation for interpolated spherical action finder
        pinv = actGrid.map(actions::ActionAngles(acG, anG), &frSinv);
        reversible_grid &= equalPosVel(pinv, point, epsi) &&
            math::fcmp(frG.Omegar, frSinv.Omegar, epsi) == 0 &&
            math::fcmp(frG.Omegaz, frSinv.Omegaz, epsi) == 0 &&
            math::fcmp(frG.Omegaphi, frSinv.Omegaphi, epsi) == 0;

        // inverse transformation for Isochrone
        coord::PosVelCyl pp = actions::mapIsochrone(M, b, actions::ActionAngles(acI, anI), &frIinv);
        reversible_iso &= equalPosVel(pp, point, epst) &&
            math::fcmp(frI.Omegar,   frIinv.Omegar,   epst) == 0 &&
            math::fcmp(frI.Omegaz,   frIinv.Omegaz,   epst) == 0 &&
            math::fcmp(frI.Omegaphi, frIinv.Omegaphi, epst) == 0;
        if(output) {
            strm << i*timestep<<"   "<<point.R<<" "<<point.z<<" "<<point.phi<<"  "<<
                toPosVelCyl(pp).R<<" "<<toPosVelCyl(pp).z<<" "<<pp.phi<<"   "<<
                anI.thetar<<" "<<anI.thetaz<<" "<<anI.thetaphi<<"  "<<
                anS.thetar<<" "<<anS.thetaz<<" "<<anS.thetaphi<<"  "<<
                anG.thetar<<" "<<anG.thetaz<<" "<<anG.thetaphi<<"  "<<
                anF.thetar<<" "<<anF.thetaz<<" "<<anF.thetaphi<<"  "<<
            "\n";
        }
    }
    statI.finish();
    statS.finish();
    statG.finish();
    statF.finish();

    bool dispI_ok = statI.rms.Jr<epsa && statI.rms.Jz<epsa && statI.rms.Jphi<epsa;
    bool dispS_ok = statS.rms.Jr<epsa && statS.rms.Jz<epsa && statS.rms.Jphi<epsa;
    bool dispG_ok = statG.rms.Jr<epsa && statG.rms.Jz<epsa && statG.rms.Jphi<epsa;
    bool dispF_ok = statF.rms.Jr<epsa && statF.rms.Jz<epsa && statF.rms.Jphi<epsa;
    bool compareIF =
             fabs(statI.avg.Jr-statF.avg.Jr)<epsr
          && fabs(statI.avg.Jz-statF.avg.Jz)<epsr
          && fabs(statI.avg.Jphi-statF.avg.Jphi)<epsa;
    bool freq_ok = statfrIr.disp() < epsf*epsf && statfrIz.disp() < epsf*epsf;
    bool HofJ_ok = statH.disp() < pow_2(epsf*statH.mean());

    std::cout << "Isochrone"
    ":  Jr="  <<utils::pp(statI.avg.Jr,  14)<<" +- "<<utils::pp(statI.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statI.avg.Jz,  14)<<" +- "<<utils::pp(statI.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statI.avg.Jphi, 6)<<" +- "<<utils::pp(statI.rms.Jphi, 7)<<
    (dispI_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_iso?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<std::endl;

#ifdef PERFTEST
    size_t ncycles=1000, npoints=traj.size();
    std::vector<coord::PosVelCyl> trajc(npoints);
    for(size_t i=0; i<npoints; i++)
        trajc[i] = toPosVelCyl(traj[i].first);
    utils::Timer timer;
    double t_iso_begin = 0;
    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalIsochrone(M, b, trajc[i], &acI);
    double t_iso_act = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalIsochrone(M, b, trajc[i], &acI, &anI);
    double t_iso_ang = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++) {
            actions::ActionAngles aa(statI.avg, actions::Angles(i*j*0.12345,i*j*0.23456,i*j*0.34567));
            actions::mapIsochrone(M, b, aa);
        }
    double t_iso_map = timer.deltaSeconds();

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/(t_iso_act-t_iso_begin), 5)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/(t_iso_ang-t_iso_act), 5)<<
    ",  map="<<utils::pp(npoints*ncycles/(t_iso_map-t_iso_ang), 5)<<std::endl;
#endif

    std::cout << "Spherical"
    ":  Jr="  <<utils::pp(statS.avg.Jr,  14)<<" +- "<<utils::pp(statS.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statS.avg.Jz,  14)<<" +- "<<utils::pp(statS.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statS.avg.Jphi, 6)<<" +- "<<utils::pp(statS.rms.Jphi, 7)<<
    //",  rmserrInverse="<<utils::pp(sqrt(errSinv/traj.size()),7) <<
    (dispS_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_sph?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<std::endl;

#ifdef PERFTEST
    ncycles /= 10;  // other methods are much slower than isochrone
    double t_sph_begin = timer.deltaSeconds();
    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalSpherical(pot, trajc[i], &acS);
    double t_sph_act = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalSpherical(pot, trajc[i], &acS, &anS);
    double t_sph_ang = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++) {
            actions::ActionAngles aa(statS.avg, actions::Angles(i*j*0.12345,i*j*0.23456,i*j*0.34567));
            actions::mapSpherical(pot, aa);
        }
    double t_sph_map = timer.deltaSeconds();

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/(t_sph_act-t_sph_begin), 5)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/(t_sph_ang-t_sph_act), 5)<<
    ",  map="<<utils::pp(npoints*ncycles/(t_sph_map-t_sph_ang), 5)<<std::endl;
#endif

    std::cout << "Interpol."
    ":  Jr="  <<utils::pp(statG.avg.Jr,  14)<<" +- "<<utils::pp(statG.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statG.avg.Jz,  14)<<" +- "<<utils::pp(statG.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statG.avg.Jphi, 6)<<" +- "<<utils::pp(statG.rms.Jphi, 7)<<
    //",  rmserrInverse="<<utils::pp(sqrt(errGinv/traj.size()),7) <<
    (dispG_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_grid?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<std::endl;

#ifdef PERFTEST
    double t_grid_begin = timer.deltaSeconds();
    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actGrid.eval(trajc[i], &acG);
    double t_grid_act = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actGrid.eval(trajc[i], &acG, &anG);
    double t_grid_ang = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++) {
            actions::ActionAngles aa(statG.avg, actions::Angles(i*j*0.12345,i*j*0.23456,i*j*0.34567));
            actGrid.map(aa);
        }
    double t_grid_map = timer.deltaSeconds();

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/(t_grid_act-t_grid_begin), 5)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/(t_grid_ang-t_grid_act), 5)<<
    ",  map="<<utils::pp(npoints*ncycles/(t_grid_map-t_grid_ang), 5)<<std::endl;
#endif

    std::cout << "Axi.Fudge"
    ":  Jr="  <<utils::pp(statF.avg.Jr,  14)<<" +- "<<utils::pp(statF.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statF.avg.Jz,  14)<<" +- "<<utils::pp(statF.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statF.avg.Jphi, 6)<<" +- "<<utils::pp(statF.rms.Jphi, 7)<<
    (dispF_ok?"":" \033[1;31m**\033[0m")<<std::endl;

#ifdef PERFTEST
    double t_fudge_begin = timer.deltaSeconds();
    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalAxisymFudge(pot, trajc[i], &acF, NULL, NULL, ifd);
    double t_fudge_act = timer.deltaSeconds();

    for(size_t j=0; j<ncycles; j++)
        for(size_t i=0; i<npoints; i++)
            actions::evalAxisymFudge(pot, trajc[i], &acF, &anF, NULL, ifd);
    double t_fudge_ang = timer.deltaSeconds();

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/(t_fudge_act-t_fudge_begin), 5)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/(t_fudge_ang-t_fudge_act), 5)<<std::endl;
#endif

    std::cout <<
    "Hamiltonian H(J)="<<utils::pp(statH.mean(), 14)<<" +- "<<utils::pp(sqrt(statH.disp()), 7)<<
    ",  H(x,v)="<<utils::pp(statE.mean(), 14)<<" +- "<<utils::pp(sqrt(statE.disp()), 7)<<
    (HofJ_ok?"":" \033[1;31m**\033[0m") <<
    (compareIF?"":" \033[1;31mNOT EQUAL\033[0m ")<<
    (freq_ok?"":" \033[1;31mFREQS NOT CONST\033[0m ")<<
    (anglesMonotonic?"":" \033[1;31mANGLES NON-MONOTONIC\033[0m ")<<std::endl;
    return dispI_ok && dispS_ok && dispG_ok && dispF_ok
        && reversible_iso && reversible_sph && reversible_grid
        && HofJ_ok && compareIF && freq_ok && anglesMonotonic;
}

void test_sph_iso()
{
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    potential::PtrPotential pot(new potential::Isochrone(M, b));
    actions::ActionFinderSpherical af(*pot);
    std::ofstream strm ("test_actions_isochrone_spherical.dat");
    strm << std::setprecision(15);
    for(double lr=-13; lr<=24; lr+=.25) {
        double r = pow(2., lr);
        double vc= v_circ(*pot, r);
        double Lc= vc * r;
        double E = 0.5*pow_2(vc) + pot->value(coord::PosCyl(r,0,0));
        for(double ll=0; ll<1; ll+=1./128) {
            double L = Lc * pow_2(sin(M_PI/2*ll));
            double Omegar, Omegaz, Jr= af.Jr(E, L, &Omegar, &Omegaz);
            actions::Actions as, ai;
            actions::Frequencies fi, fs;
            coord::PosVelCyl point(r, 0, 0, sqrt(vc*vc-pow_2(L/r)), 0, L/r);
            actions::evalSpherical(*pot, point, &as, NULL, &fs);
            actions::evalIsochrone(M, b, point, &ai, NULL, &fi);
            strm << E << ' ' << L/Lc << ' ' <<
                ai.Jr/(Lc-L) << ' ' << as.Jr/(Lc-L) << ' ' << Jr/(Lc-L) << ' ' <<
                fi.Omegar << ' ' << fs.Omegar << ' ' << Omegar << ' ' <<
                fi.Omegaz << ' ' << fs.Omegaz << ' ' << Omegaz << ' ' <<'\n'; 
        }
        strm <<'\n';
    }
}

int main()
{
    //test_sph_iso();
    bool ok=true;
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.3, 1.1, 0.1, 0.4,  0.1), "ordinary case");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 2.2, 1.0, 0.0,  0.5), "Jz=0");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 3.3, 0.0, 0.21, 0.9), "Jr small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 4.4, 0.6, 1.0, 1e-4), "Jphi small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.5, 5.5, 0.5, 0.7, -0.5), "Jphi negative");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0,M_PI, 0.0, 0.0, -0.5), "Jz=0, Jphi negative");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}

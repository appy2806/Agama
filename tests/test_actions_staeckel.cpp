/** \file    test_actions_staeckel.cpp
    \author  Eugene Vasiliev
    \date    2015-2016

    This test verifies the correctness of action/angle determination by the Staeckel fudge.

    We create an instance of Staeckel potential (in this example, oblate perfect ellipsoid),
    perform numerical orbit integration in this potential, and for each point
    on the trajectory, compute the values of actions and angles.
    Two related methods are employed - the first is valid only for a Staeckel potential,
    and the second (fudge) is an approximation which essentially turns out to be exact
    for this potential.
    We test that the values of actions are nearly constant (to the limit of orbit
    integration accuracy), and that angles increase linearly with time.
    We use several initial conditions corresponding to various generic and extreme cases
    (e.g., an in-plane orbit, or an orbit with very small radial action, or an orbit
    close to the separatrix between box and tube orbits in x-z plane).
*/
#include "potential_perfect_ellipsoid.h"
#include "potential_factory.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include "math_core.h"
#include "math_random.h"
#include "debug_utils.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double eps=5e-8;               // accuracy of comparison
const double epsi=5e-7;              // accuracy of comparison for interpolator
const double epsint=3e-3;            // accuracy of action interpolator
const double axis_a=1.6, axis_c=1.0; // axes of perfect ellipsoid
const bool output=utils::verbosityLevel >= utils::VL_VERBOSE;  // whether to create text files with orbits
const char* err=" \033[1;31m**\033[0m\n";
const char* exc=" \033[1;33mCAUGHT EXCEPTION\033[0m\n";

bool test(const potential::OblatePerfectEllipsoid& potential,
    const actions::ActionFinderAxisymFudge& actfinder,
    const coord::PosVelCar& initial_conditions,
    const char* title)
{
    const double total_time=100.;
    const double timestep=1./8;
    std::cout << "\033[1;37m   ===== "<<title<<" =====\033[0m\n";
    std::vector<std::pair<coord::PosVelCar, double> > traj = orbit::integrateTraj(
        initial_conditions, total_time, timestep, potential);
    actions::ActionStat stats, statf, stati;
    actions::Actions act;
    actions::Angles  ang;
    bool exs=false, exf=false, exi=false;
    std::ofstream strm;
    if(output) {
        std::ostringstream s;
        double x[6];
        initial_conditions.unpack_to(x);
        s<<"ActionsStaeckel_"<<
            x[0]<<'_'<<x[1]<<'_'<<x[2]<<'_'<<x[3]<<'_'<<x[4]<<'_'<<x[5];
        strm.open(s.str().c_str());
    }
    // two estimates of focal distance: from the trajectory
    // (sensible unless z==0 everywhere, in which case its value doesn't matter anyway)
    // and from the interpolator
    std::vector<coord::PosCar> trajpoints(traj.size());
    for(size_t i=0; i<traj.size(); i++)
        trajpoints[i] = traj[i].first;
    double ifd_p = actions::estimateFocalDistancePoints(potential, trajpoints);
    double ifd_i = actfinder.focalDistance(toPosVelCyl(initial_conditions));
    for(size_t i=0; i<traj.size(); i++) {
        const coord::PosVelCyl pc = coord::toPosVelCyl(traj[i].first);
        try {
            actions::evalAxisymStaeckel(potential, pc, &act);
            stats.add(act);
        }
        catch(std::exception &e) {
            if(!exs) std::cout << "Exception in Staeckel at i="<<i<<": "<<e.what()<<"\n";
            exs=true;
        }
        try {
            actions::evalAxisymFudge(potential, pc, &act, &ang, NULL, ifd_p);
            statf.add(act);
            if(output) {
                const coord::PosVelProlSph pp = coord::toPosVel<coord::Cyl,coord::ProlSph>
                    (pc, potential.coordsys());
                strm << i*timestep<<"   "<<
                    pc.phi<<" "<<pc.vphi<<" "<<pp.lambda<<" "<<pp.nu<<" "<<
                    pp.lambdadot<<" "<<pp.nudot<<"  "<<
                    ang.thetar<<" "<<ang.thetaz<<" "<<ang.thetaphi<<"  "<<
                    utils::pp(act.Jr,12)<<" "<<utils::pp(act.Jz,12)<<"  ";
            }
        }
        catch(std::exception &e) {
            if(!exf) std::cout << "Exception in Fudge at i="<<i<<": "<<e.what()<<"\n";
            exf=true;
        }
        try {
            actions::Actions a = actfinder.actions(pc);
            stati.add(a);
            if(output)
                strm << utils::pp(a.Jr,12)<<" "<<utils::pp(a.Jz,12)<<"\n";
        }
        catch(std::exception &e) {
            if(!exi) std::cout << "Exception in Interpolator at i="<<i<<": "<<e.what()<<"\n";
            exi=true;
        }
    }
    stats.finish();
    statf.finish();
    stati.finish();
    bool oks  = stats.rms.Jr<eps && stats.rms.Jz<eps && stats.rms.Jphi<eps && !exs;
    bool okf  = statf.rms.Jr<eps && statf.rms.Jz<eps && statf.rms.Jphi<eps && !exf
        && fabs(stats.avg.Jr-statf.avg.Jr)<eps
        && fabs(stats.avg.Jz-statf.avg.Jz)<eps
        && fabs(stats.avg.Jphi-statf.avg.Jphi)<eps
        &&     (stats.avg.Jz==0 || fabs(ifd_p - ifd_i)<1e-5);
    bool oki  = stati.rms.Jr<epsi&& stati.rms.Jz<epsi&& stati.rms.Jphi<eps && !exi
        && fabs(stats.avg.Jr-stati.avg.Jr)<epsint
        && fabs(stats.avg.Jz-stati.avg.Jz)<epsint;
    std::cout << "Exact"
    ":  Jr="  << utils::pp(stats.avg.Jr,  9) <<" +- "<< utils::pp(stats.rms.Jr,  7) <<
    ",  Jz="  << utils::pp(stats.avg.Jz,  9) <<" +- "<< utils::pp(stats.rms.Jz,  7) <<
    ",  Jphi="<< utils::pp(stats.avg.Jphi,9) <<" +- "<< utils::pp(stats.rms.Jphi,7) <<
    (exs ? exc : !oks ? err : "\n");
    std::cout << "Fudge"
    ":  Jr="  << utils::pp(statf.avg.Jr,  9) <<" +- "<< utils::pp(statf.rms.Jr,  7) <<
    ",  Jz="  << utils::pp(statf.avg.Jz,  9) <<" +- "<< utils::pp(statf.rms.Jz,  7) <<
    ",  Jphi="<< utils::pp(statf.avg.Jphi,9) <<" +- "<< utils::pp(statf.rms.Jphi,7) <<
    (exf ? exc : !okf ? err : "\n");
    std::cout << "Inter"
    ":  Jr="  << utils::pp(stati.avg.Jr,  9) <<" +- "<< utils::pp(stati.rms.Jr,  7) <<
    ",  Jz="  << utils::pp(stati.avg.Jz,  9) <<" +- "<< utils::pp(stati.rms.Jz,  7) <<
    ",  Jphi="<< utils::pp(stati.avg.Jphi,9) <<" +- "<< utils::pp(stati.rms.Jphi,7) <<
    (exi ? exc : !oki ? err : "\n");
    return oks && okf && oki;
}

void testPerformance()
{
    potential::PtrPotential pot = potential::createPotential(utils::KeyValueMap("type=disk scaleradius=1 scaleheight=0.1"));
    actions::ActionFinderAxisymFudge af(pot, false);
    const size_t N = 100000;
    std::vector<coord::PosVelCyl> points(N);
    for(size_t i=0; i<N; i++) {
        double R, z, vR, vphi;
        math::getNormalRandomNumbers(R, z);
        math::getNormalRandomNumbers(vR, vphi);
        R*=R;
        double vesc = sqrt(-2*pot->value(coord::PosCyl(R, z, 0)));
        points[i] = coord::PosVelCyl(R, z, 0, vesc*vR*0.2, 0, vesc*vphi*0.2);
    }
    std::cout << "Testing the performance of " << af.name() << std::endl;
    utils::Timer timer;
    actions::Actions ac;
    actions::Angles an;
    actions::Frequencies fr;
    for(size_t i=0; i<N; i++) {
        af.eval(points[i], &ac);
    }
    double t1 = timer.deltaSeconds();
    std::cout << "Evaluate actions: " << utils::toString(N/t1, 3) << " points/s" << std::flush;
    for(size_t i=0; i<N; i++) {
        af.eval(points[i], &ac, NULL, &fr);
    }
    double t2 = timer.deltaSeconds();
    std::cout << ", +frequencies: " << utils::toString(N/(t2-t1), 3) << " points/s" << std::flush;
    for(size_t i=0; i<N; i++) {
        af.eval(points[i], &ac, &an, &fr);
    }
    double t3 = timer.deltaSeconds();
    std::cout << ", +angles: " << utils::toString(N/(t3-t2), 3) << " points/s" << std::endl;
}

int main() {
    potential::PtrOblatePerfectEllipsoid pot(new potential::OblatePerfectEllipsoid(1.0, axis_a, axis_c));
    const actions::ActionFinderAxisymFudge af(pot, true);
    bool allok=true;
    allok &= test(*pot, af, coord::PosVelCar(1, 0.3, 0.1, 0.1, 0.4, 0.1   ), "ordinary case");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0  , 0.2, 0.3486), "thin orbit (Jr~0)");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4732), "thin orbit in x-z plane (Jr~0, Jphi=0)");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4097), "tube orbit in x-z plane near separatrix");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4096), "box orbit in x-z plane near separatrix");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. ,1e-8, 0.4097), "orbit with Jphi<<J_z");
    allok &= test(*pot, af, coord::PosVelCar(1, 0.3, 0. , 0.1, 0.4, 1e-4  ), "almost in-plane orbit (Jz~0)");
    allok &= test(*pot, af, coord::PosVelCar(1, 0.3, 0. , 0.1, 0.4, 0.    ), "exactly in-plane orbit (Jz=0)");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. ,.296, 0.    ), "almost circular in-plane orbit (Jz=0,Jr~0)");
    allok &= test(*pot, af, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.    ), "exactly radial in-plane orbit (Jz=0,Jphi=0)");
    testPerformance();
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
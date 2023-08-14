#ifndef EIGHTDOF_H
#define EIGHTDOF_H
#include <stdint.h>
#include "../utils.h"
/*
Header file for the 8 DOF model implemented in cpp
*/

namespace EightDOF{

    // TMeasy parameter structure
    struct TMeasyParam{

        // constructor that takes default values of HMMWV
        TMeasyParam()
            : _jw(6.69), _rr(0.015), _mu(0.8), _r0(0.4699), _pn(8562.8266),
            _pnmax(29969.893), _cx(185004.42), _cy(164448.37),  _kt(411121.0), 
            _dx(3700.), _dy(3488.), _rdyncoPn(0.375), _rdyncoP2n(0.75), _fzRdynco(0), _rdyncoCrit(0),
            _dfx0Pn(151447.29), _dfx0P2n(236412.79), _fxmPn(7575.3614), _fxmP2n(12808.276),
            _fxsPn(4657.9208), _fxsP2n(8625.3352), _sxmPn(0.12), _sxmP2n(0.15),
            _sxsPn(0.9), _sxsP2n(0.95), _dfy0Pn(50931.693), _dfy0P2n(94293.847),
            _fymPn(6615.0404), _fymP2n(12509.947), _fysPn(6091.5092), _fysP2n(11443.875),
            _symPn(0.38786), _symP2n(0.38786), _sysPn(0.82534), _sysP2n(0.91309), _step(1e-2) {}

        // constructor that takes given values - ugly looking code - can this be beutified?
        TMeasyParam(double jw, double rr, double mu, double r0, double pn,
            double pnmax, double cx, double cy, double dx, double dy, double kt, 
            double rdyncoPn, double rdyncoP2n, double fzRdynco, double rdyncoCrit, double dfx0Pn, double dfx0P2n, 
            double fxmPn, double fxmP2n, double fxsPn, double fxsP2n, double sxmPn, 
            double sxmP2n, double sxsPn, double sxsP2n, double dfy0Pn, double dfy0P2n, 
            double fymPn, double fymP2n, double fysPn, double fysP2n, double symPn, 
            double symP2n, double sysPn, double sysP2n, double step)
            : _jw(jw), _rr(rr), _mu(mu), _r0(r0), _pn(pn),
            _pnmax(pnmax), _cx(cx), _cy(cy), _kt(kt), _dx(dx),
            _dy(dy), _rdyncoPn(rdyncoPn), _rdyncoP2n(rdyncoP2n), _fzRdynco(fzRdynco),
            _rdyncoCrit(rdyncoCrit), _dfx0Pn(dfx0Pn), _dfx0P2n(dfx0P2n), _fxmPn(fxmPn), _fxmP2n(fxmP2n),
            _fxsPn(fxsPn), _fxsP2n(fxsP2n), _sxmPn(sxmPn), _sxmP2n(sxmP2n),
            _sxsPn(sxsPn), _sxsP2n(sxsP2n), _dfy0Pn(dfy0Pn), _dfy0P2n(dfy0P2n),
            _fymPn(fymPn), _fymP2n(fymP2n), _fysPn(fysPn), _fysP2n(fysP2n),
            _symPn(symPn), _symP2n(symP2n), _sysPn(sysPn), _sysP2n(sysP2n), _step(step) {}


        // basic tire parameters
        double _jw; // wheel inertia
        double _rr; // rolling resistance of tire
        double _mu; // friction constant
        double _r0; // unloaded tire radius

        // TM easy specific tire params
        double _pn, _pnmax; // nominal and max vertical force
        double _cx, _cy, _kt; // longitudinal, lateral and vertical stiffness
        double _dx, _dy; // longitudinal and lateral damping coeffs. No vertical damping


        // TM easy force characteristic params 
        // 2 values - one at nominal load and one at max load

        // dynamic radius weighting coefficient and a critical value for the vertical force
        double _rdyncoPn, _rdyncoP2n, _fzRdynco, _rdyncoCrit;

        // Longitudinal
        double _dfx0Pn, _dfx0P2n; // intial longitudinal slopes dFx/dsx [N]
        double _fxmPn,_fxmP2n; // maximum longituidnal force [N]
        double _fxsPn, _fxsP2n; // Longitudinal load at sliding [N]
        double _sxmPn, _sxmP2n; // slip sx at maximum longitudinal load Fx
        double _sxsPn, _sxsP2n; // slip sx where sliding begins

        // Lateral
        double _dfy0Pn, _dfy0P2n; // intial lateral slopes dFx/dsx [N]
        double _fymPn,_fymP2n; // maximum lateral force [N]
        double _fysPn, _fysP2n; // Lateral load at sliding [N]
        double _symPn, _symP2n; // slip sx at maximum lateral load Fx
        double _sysPn, _sysP2n; // slip sx where sliding begins

        double _step; // integration time step




    };

    // Tm easy state structure - actual states + things that we need to keep track of
    struct TMeasyState{
        // default contructor to 0's
        TMeasyState ()
            : _xe(0.), _ye(0.), _xedot(0.), _yedot(0.), _omega(0.),
            _xt(0.), _rStat(0.), _fx(0.), _fy(0.), _fz(0.), _vsx(0.), _vsy(0.), _My(0.), _engTor(0.) {}


        // special constructor in case we want to start the simualtion at
        // some other time step
        TMeasyState(double xe, double ye, double xedot, double yedot,
            double omega, double xt, double rStat, double fx, double fy,
            double fz, double vsx, double vsy, double init_ratio)
            : _xe(xe), _ye(ye), _xedot(xedot), _yedot(yedot),
            _omega(omega), _xt(xt), _rStat(rStat), _fx(fx), _fy(fy),
            _fz(fz), _vsx(vsx), _vsy(vsy) {}


        // the actual state that are intgrated
        double _xe, _ye; // long and lat tire deflection
        double _xedot, _yedot; // long and lat tire deflection velocity
        double _omega; // angular velocity of wheel


        // other "states" that we need to keep track of
        double _xt; // vertical tire compression
        double _rStat; // loaded tire radius
        double _fx, _fy, _fz; // long, lateral and vertical force in tire frame

        // velocities in tire frame
        double _vsx, _vsy;

        double _My; // Rolling resistance moment (negetive)

        // torque from engine that we keep track of
        double _engTor;



        


    };
    // vehicle Parameters structure
    struct VehicleParam{
        
        // default constructor with pre tuned values from HMMVW calibration
        VehicleParam() 
            : _a(1.6889), _b(1.6889), _h(0.713), _m(2097.85), _jz(4519.), _jx(1289.),
            _jxz(3.265), _cf(1.82), _cr(1.82), _muf(127.866),
            _mur(129.98), _hrcf(0.379), _hrcr(0.327), _krof(31000),
            _kror(31000), _brof(3300), _bror(3300),_nonLinearSteer(false), _maxSteer(0.6525249), _crankInertia(1.1), _upshift_RPS(10000),
            _downshift_RPS(0), _tcbool(false), _maxBrakeTorque(4000.), 
            _c1(0.), _c0(0.), _step(1e-2), _throttleMod(0)  {}
        
        
        // constructor
        VehicleParam(double a, double b, double h, double m, double Jz
            , double Jx, double Jxz, double cf, double cr, double muf,
            double mur, double hrcf, double hrcr, double krof, double kror,
            double brof, double bror,bool steer_bool, double maxSteer, double crank_inertia, double up_RPS, double down_RPS, bool tc_bool,
            double maxTorque, double brakeTorque, double maxSpeed, double c1, double c0, double step, bool throttle_mod)
            : _a(a), _b(b), _h(h), _m(m), _jz(Jz), _jx(Jx), _jxz(Jxz),
            _cf(cf), _cr(cr), _muf(muf), _mur(mur), _hrcf(hrcf), _hrcr(hrcr),
            _krof(krof), _kror(kror), _brof(bror), _bror(bror),_nonLinearSteer(steer_bool), _maxSteer(maxSteer), _crankInertia(crank_inertia),
            _upshift_RPS(up_RPS), _downshift_RPS(down_RPS), _tcbool(tc_bool), _maxBrakeTorque(brakeTorque),
            _c1(c1), _c0(c0), _step(step), _throttleMod(throttle_mod) {}


        double _a, _b; // distance c.g. - front axle & distance c.g. - rear axle (m)
        double _h; // height of c.g
        double _m; // total vehicle mass (kg)
        double _jz; // yaw moment inertia (kg.m^2)
        double _jx; // roll inertia 
        double _jxz; // XZ inertia
        double _cf, _cr; // front and rear track width
        double _muf,_mur; // front and rear unsprung mass
        double _hrcf,_hrcr; //front and rear roll centre height below C.g
        double _krof,_kror,_brof,_bror; // front and rear roll stiffness and damping

        // Bool that checks if the steering is non linea
        // 1 -> Steering is non linear, requires a steering map defined - example in json
        // 0 -> Steering is linear. Need only a max steer defined. The normalized sterring then just multiplies against
        // this value to give a wheel angle
        bool _nonLinearSteer; 
        // Non linear steering map in case the steering mechanism is not linear
        std::vector<MapEntry> _steerMap;
        // max steer angle parameters of the vehicle
        double _maxSteer;

        // crank shaft inertia
        double _crankInertia;
        // some gear parameters
        double _upshift_RPS;
        double _downshift_RPS;
        std::vector<double> _gearRatios; // gear ratio

        // boolean for torque converter presense
        bool _tcbool;


        
        // double _maxTorque; // Max torque
        double _maxBrakeTorque; // max brake torque
        // double _maxSpeed; // Max speed
        double _c1, _c0; // motor resistance - mainly needed for rc car 
        
        double _step; // vehicle integration time step

        // Bool that defines how the throttle modulates the maps
        // 1 -> Modulates like in a motor -> Modifies the entire torque and RPM map
        // 0 -> Modulates like in an engine -> multiplies only against the torque - > Default
        bool _throttleMod; 
        // We will always define the powertrain with a map 
        std::vector<MapEntry> _powertrainMap;
        std::vector<MapEntry> _lossesMap;

        // torque converter maps
        std::vector<MapEntry> _CFmap; // capacity factor map
        std::vector<MapEntry> _TRmap; // Torque ratio map

        
    };

    // vehicle states structure
    struct VehicleState{
        
        // default constructor just assigns zero to all members
        VehicleState() 
            : _x(0.), _y(0.), _u(0.), _v(0.), _psi(0.), _wz(0.), 
            _phi(0.), _wx(0.), _udot(0.), _vdot(0.), _wxdot(0.), _wzdot(0.),
            _fzlf(0.), _fzrf(0.), _fzlr(0.), _fzrr(0.), _tor(0.), _crankOmega(0.), _current_gr(0), _tc_reverse_flow(false) {}


        // special constructor in case need to start simulation
        // from some other state
        double _x, _y; // x and y position
        double _u, _v; // x and y velocity
        double _psi, _wz; // yaw angle and yaw rate
        double _phi, _wx; // roll angle and roll rate
        

        // acceleration 'states'
        double _udot, _vdot; 
        double _wxdot, _wzdot;


        // vertical forces on each tire
        double _fzlf, _fzrf, _fzlr, _fzrr;

        // crank torque (used to transmit torque to tires) and crank angular velocity state
        double _tor;
        double _crankOmega;
        double _debugtor;
        int _current_gr;
        double _tc_inp_tor;
        double _tc_out_tor;
        double _tc_out_omg;
        bool _tc_reverse_flow;
        double _sr;


    };

////////////////////////////////////////////////////////////////////////// Vehicle Functions /////////////////////////////////////////////////
    // sets the vertical forces based on the vehicle weight
    void vehInit(VehicleState& v_state, const VehicleParam& v_params);

    double driveTorque(VehicleParam& v_params, const double throttle, const double omega);

    inline double brakeTorque(const VehicleParam& v_params, const double brake){
        return v_params._maxBrakeTorque * brake;
    }
    void differentialSplit(double torque,
                        double max_bias,
                        double speed_left,
                        double speed_right,
                        double& torque_left,
                        double& torque_right);


    void evalPowertrain(VehicleState& v_states, TMeasyState& tirelf_st,
                        TMeasyState& tirerf_st, TMeasyState& tirelr_st, 
                        TMeasyState& tirerr_st, VehicleParam& v_params, const TMeasyParam& t_params,
                        const std::vector <double>& controls);

    // function to advance the time step of the vehicle
    void vehAdv(VehicleState& v_states, const VehicleParam& v_params,
                const std::vector <double>& fx, const std::vector <double>& fy, const double huf, const double hur);



    // setting vehicle parameters using a JSON file
    void setVehParamsJSON(VehicleParam& v_params, const char *fileName);



    void vehToTireTransform(TMeasyState& tirelf_st,TMeasyState& tirerf_st,
                                TMeasyState& tirelr_st, TMeasyState& tirerr_st, 
                                const VehicleState& v_states, const VehicleParam& v_params, const std::vector <double>& controls);

    void tireToVehTransform(TMeasyState& tirelf_st,TMeasyState& tirerf_st,
                                TMeasyState& tirelr_st, TMeasyState& tirerr_st,
                                const VehicleState& v_states, const VehicleParam& v_params, const std::vector <double>& controls);


/////////////////////////////////////////////////////////////////////// Tire Functions ///////////////////////////////////////////////////////////


    // sets the vertical tire deflection based on the vehicle weight 
    // template based on which tire
    void tireInit(TMeasyParam& t_params);


    // function to calculate the force from the force charactristics
    // used by tireSync
    void tmxy_combined(double& f, double& fos, double s, double df0, double sm, double fm, double ss, double fs);

    // Advances the time step
    // updates the tire forces which is then used by the vehicle model 
    void tireAdv(TMeasyState& t_states, const TMeasyParam& t_params, const VehicleState& v_states, 
                    const VehicleParam& v_params, const std::vector <double>& controls);


    // setting tire parameters using a JSON file
    void setTireParamsJSON(TMeasyParam& t_params, const char * fileName);
}

#endif
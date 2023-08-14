#include<iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include "../third_party/rapidjson/document.h"
#include "../third_party/rapidjson/filereadstream.h"
#include "Eightdof.h"
#include "../utils.h"

using namespace EightDOF;

///////////////////////////////////////////////////////////////// Vehicle Functions ///////////////////////////////////////////////////////////////////////
/*
Code for the Eight dof model implemented in cpp
*/

// sets the vertical forces based on the vehicle weight
void EightDOF::vehInit(VehicleState& v_state, const VehicleParam& v_param){
    double weight_split = ((v_param._m*G*v_param._b) /
                        (2*(v_param._a+v_param._b))+v_param._muf*G);
    v_state._fzlf = v_state._fzrf = weight_split;
    
    weight_split = ((v_param._m*G*v_param._b) /
                        (2*(v_param._a+v_param._b))+v_param._mur*G);

    v_state._fzlr = v_state._fzrr = weight_split;

}

// returns drive toruqe at a given omega 
double EightDOF::driveTorque(VehicleParam& v_params, const double throttle, const double motor_speed){



    // need to do this because lower_bound does not take a constant vector
    // std::vector<MapEntry> powertrain_map = v_params._powertrainMap;
    // std::vector<MapEntry> losses_map = v_params._lossesMap;
    
    double motor_torque = 0.;
    // If we have throttle modulation like in a motor
    if(v_params._throttleMod){
        unsigned int len = v_params._powertrainMap.size();
        for(unsigned int i = 0; i<len; i++){
            v_params._powertrainMap[i]._x = v_params._powertrainMap[i]._x * throttle;
            v_params._powertrainMap[i]._y = v_params._powertrainMap[i]._y * throttle;
        }
        // interpolate in the torque map to get the torque at this paticular speed
        motor_torque = getMapY(v_params._powertrainMap, motor_speed);
        double motor_losses = getMapY(v_params._lossesMap, motor_speed);
        motor_torque = motor_torque + motor_losses;
    }
    else{ // Else we don't multiply the map but just the output torque
        motor_torque = getMapY(v_params._powertrainMap, motor_speed);
        double motor_losses = getMapY(v_params._lossesMap, motor_speed);
        motor_torque = motor_torque * throttle + motor_losses;

    }
    return motor_torque;
}
/*
Function that calculates the torque split to each tire based on the differential max bias
Exactly the same as Chrono implementation
*/

void EightDOF::differentialSplit(double torque,
                       double max_bias,
                       double speed_left,
                       double speed_right,
                       double& torque_left,
                       double& torque_right) {
    double diff = std::abs(speed_left - speed_right);

    // The bias grows from 1 at diff=0.25 to max_bias at diff=0.5
    double bias = 1;
    if (diff > 0.5)
        bias = max_bias;
    else if (diff > 0.25)
        bias = 4 * (max_bias - 1) * diff + (2 - max_bias);

    // Split torque to the slow and fast wheels.
    double alpha = bias / (1 + bias);
    double slow = alpha * torque;
    double fast = torque - slow;

    if (std::abs(speed_left) < std::abs(speed_right)) {
        torque_left = slow;
        torque_right = fast;
    } else {
        torque_left = fast;
        torque_right = slow;
    }
}


/*
Function to evalaute the drive line and engine torques and advance the wheel angular velocites
*/

void EightDOF::evalPowertrain(VehicleState& v_states, TMeasyState& tirelf_st,
                    TMeasyState& tirerf_st, TMeasyState& tirelr_st, 
                    TMeasyState& tirerr_st, VehicleParam& v_params, const TMeasyParam& t_params,
                    const std::vector <double>& controls){

                        // get controls
                        double throttle = controls[2];
                        double brake = controls[3];


                        // some variables needed outside
                        double torque_t = 0;
                        double max_bias = 2;
                        // If we have a torque converter
                        if(v_params._tcbool){
                            // set reverse flow to false at each timestep
                            v_states._tc_reverse_flow = false;
                            // Split the angular velocities all the way uptill the gear box. All from previous time step
                            double omega_t = 0.25 * (tirelf_st._omega + tirerf_st._omega + tirelr_st._omega + tirerr_st._omega);
                            
                            // get the angular velocity at the torque converter wheel side 
                            // Note, the gear includes the differential gear as well
                            double omega_out = omega_t / (v_params._gearRatios[v_states._current_gr]);

                            // Get the omega input to the torque from the engine from the previous time step
                            double omega_in = v_states._crankOmega;


                            //extract maps
                            std::vector<MapEntry> CF_map = v_params._CFmap;
                            std::vector<MapEntry> TR_map = v_params._TRmap;

                            double sr, cf, tr;
                            if((omega_out < 1e-9) || (omega_in < 1e-9)){ // if we are at the start things can get unstable
                                sr = 0;
                                // Get capacity factor from capacity lookup table
                                cf = getMapY(CF_map,sr);

                                // Get torque ratio from Torque ratio lookup table 
                                tr = getMapY(TR_map,sr);
                            }
                            else{
                                // speed ratio for torque converter
                                sr =  omega_out / omega_in;

                                // Check reverse flow
                                if(sr > 1.){
                                    sr = 1. - (sr - 1.);
                                    v_states._tc_reverse_flow = true;
                                }

                                if(sr < 0){
                                    sr = 0;
                                }

                                // get capacity factor from lookup table
                                cf = getMapY(CF_map,sr);

                                // Get torque ratio from Torque ratio lookup table 
                                tr = getMapY(TR_map,sr);                              
                            }
                            // torque applied to the crank shaft
                            double torque_in = -std::pow((omega_in / cf),2);

                            // if its reverse flow, this should act as a brake
                            if(v_states._tc_reverse_flow){
                                torque_in = -torque_in;
                            }

                            // torque applied to the shaft from torque converter on the wheel side
                            double torque_out;
                            if(v_states._tc_reverse_flow){
                                torque_out = -torque_in; 
                            }
                            else{
                                torque_out = -tr * torque_in ;
                            }

                            // Now torque after the transimission
                            torque_t = torque_out / v_params._gearRatios[v_states._current_gr];
                            if(std::abs((v_states._u - 0) < 1e-9) && (torque_t < 0)){
                                torque_t = 0;
                            } 

                            /////// DEBUG
                            v_states._tc_inp_tor = -torque_in;
                            v_states._tc_out_tor = torque_out;
                            v_states._tc_out_omg = omega_out;
                            v_states._sr = sr;

                            //////// Integrate Crank shaft

                            v_states._debugtor = driveTorque(v_params, throttle, v_states._crankOmega); //// DEBUG
                            double dOmega_crank = (1./v_params._crankInertia) * (driveTorque(v_params, throttle, v_states._crankOmega) + torque_in);

                            v_states._crankOmega = v_states._crankOmega + v_params._step * dOmega_crank;


                            ////// Gear shift for the next time step -> Here we have to check the RPM of the shaft from the T.C
                            if(omega_out > v_params._upshift_RPS){
                                
                                // check if we have enough gears to upshift
                                if(v_states._current_gr < v_params._gearRatios.size() - 1){
                                    v_states._current_gr++;
                                }
                            }
                            // downshift
                            else if(omega_out < v_params._downshift_RPS){
                                // check if we can down shift
                                if(v_states._current_gr > 0){
                                    v_states._current_gr--;
                                }
                            }                           


                        }
                        else{ // if there is no torque converter, things are simple

                            // In this case, there is no state for the engine omega
                            v_states._crankOmega = 0.25 * (tirelf_st._omega + tirerf_st._omega + tirelr_st._omega + tirerr_st._omega)
                                                    / v_params._gearRatios[v_states._current_gr];

                            // The torque after tranny will then just become as there is no torque converter
                            torque_t = driveTorque(v_params, throttle, v_states._crankOmega) / v_params._gearRatios[v_states._current_gr];

                            if(std::abs((v_states._u - 0) < 1e-9) && (torque_t < 0)){
                                torque_t = 0;
                            } 

                            ///// Upshift gear for next time step -> Here the crank shaft is directly connected to the gear box
                            if(v_states._crankOmega > v_params._upshift_RPS){
                                
                                // check if we have enough gears to upshift
                                if(v_states._current_gr < v_params._gearRatios.size() - 1){
                                    v_states._current_gr++;
                                }
                            }
                            // downshift
                            else if(v_states._crankOmega < v_params._downshift_RPS){
                                // check if we can down shift
                                if(v_states._current_gr > 1){
                                    v_states._current_gr--;
                                }
                            }               


                        }

                        //////// Amount of torque transmitted to the wheels

                        // torque split between the  front and rear (always half)
                        double torque_front = torque_t * 0.5;
                        double torque_rear = torque_t * 0.5;


                        // first the front wheels
                        differentialSplit(torque_front, max_bias, tirelf_st._omega, tirerf_st._omega, tirelf_st._engTor, tirerf_st._engTor);
                        // then rear wheels
                        differentialSplit(torque_rear, max_bias, tirelr_st._omega, tirerr_st._omega, tirelr_st._engTor, tirerr_st._engTor);


                        // now use this force for our omegas
                        // double tor = driveTorque(v_params, t_states._current_gr, throttle, t_states._omega)/4. + My - t_states._fx * t_states._rStat;
                        // dOmega = (1/t_params._jw) * (tor - sgn(t_states._omega) * brakeTorque(v_params, brake) );

                        // Get dOmega for each tire
                        double dOmega_lf = (1/t_params._jw) * (tirelf_st._engTor + tirelf_st._My - sgn(tirelf_st._omega) 
                                            * brakeTorque(v_params, brake) - tirelf_st._fx * tirelf_st._rStat);

                        // double dOmega_lf = (1/t_params._jw) * (torque_t*0.25 + tirelf_st._My - sgn(tirelf_st._omega) 
                        //                     * brakeTorque(v_params, brake) - tirelf_st._fx * tirelf_st._rStat);

                        double dOmega_rf = (1/t_params._jw) * (tirerf_st._engTor + tirerf_st._My - sgn(tirerf_st._omega) 
                                            * brakeTorque(v_params, brake) - tirerf_st._fx * tirerf_st._rStat);


                        // double dOmega_rf = (1/t_params._jw) * (torque_t*0.25 + tirerf_st._My - sgn(tirerf_st._omega) 
                        //                     * brakeTorque(v_params, brake) - tirerf_st._fx * tirerf_st._rStat);
                        double dOmega_lr = (1/t_params._jw) * (tirelr_st._engTor + tirelr_st._My - sgn(tirelr_st._omega) 
                                            * brakeTorque(v_params, brake) - tirelr_st._fx * tirelr_st._rStat);

                        // double dOmega_lr = (1/t_params._jw) * (torque_t*0.25 + tirelr_st._My - sgn(tirelr_st._omega) 
                        //                     * brakeTorque(v_params, brake) - tirelr_st._fx * tirelr_st._rStat);
                        double dOmega_rr = (1/t_params._jw) * (tirerr_st._engTor + tirerr_st._My - sgn(tirerr_st._omega) 
                                            * brakeTorque(v_params, brake) - tirerr_st._fx * tirerr_st._rStat);
                        // double dOmega_rr = (1/t_params._jw) * (torque_t*0.25 + tirerr_st._My - sgn(tirerr_st._omega) 
                        //                     * brakeTorque(v_params, brake) - tirerr_st._fx * tirerr_st._rStat);

                        // integrate omega using the latest dOmega
                        tirelf_st._omega = tirelf_st._omega + t_params._step * dOmega_lf;
                        tirerf_st._omega = tirerf_st._omega + t_params._step * dOmega_rf;
                        tirelr_st._omega = tirelr_st._omega + t_params._step * dOmega_lr;
                        tirerr_st._omega = tirerr_st._omega + t_params._step * dOmega_rr;

}

/*
function to advance the time step of the 8DOF vehicle
along with the vehicle state that will be updated, we pass the
vehicle paramters , a vector containing the longitudinal forces,
and a vector containing the lateral forces by reference, the front
and rear unspring mass positions (loaded tire radius) and the controls 
--- There is no point passing all the tire states and paramters 
since none of the are used here - however, there is thus an additional
copy in main for getting these states into a vector. So thus, do not know
if this way is actually faster, but I think it is much cleaner than passing 
4 tire states and paramaters
*/

void EightDOF::vehAdv(VehicleState& v_states, const VehicleParam& v_params,
            const std::vector <double>& fx, const std::vector <double>& fy, const double huf, const double hur){


    // get the total mass of the vehicle and the vertical distance from the sprung
    // mass C.M. to the vehicle 
    double mt = v_params._m + 2 * (v_params._muf + v_params._mur);
    double hrc = (v_params._hrcf * v_params._b + v_params._hrcr * v_params._a) / (v_params._a + v_params._b);


    // a bunch of varaibles to simplify the formula
    double E1 = -mt * v_states._wz * v_states._u + (fy[0] + fy[1] + fy[2] + fy[3]);
    
    
    double E2 = (fy[0] + fy[1])*v_params._a - (fy[2] + fy[3])*v_params._b + (fx[1] - fx[0])*v_params._cf/2 +
                (fx[3] - fx[2])*v_params._cr/2 + (-v_params._muf*v_params._a + 
                v_params._mur*v_params._b)*v_states._wz*v_states._u;
    
    
    double E3 = v_params._m * G * hrc * v_states._phi - (v_params._krof + v_params._kror)*v_states._phi - 
                (v_params._brof + v_params._bror)*v_states._wx + hrc*v_params._m*v_states._wz*v_states._u;

    double A1 = v_params._mur*v_params._b - v_params._muf*v_params._a;

    double A2 = v_params._jx + v_params._m * std::pow(hrc,2);

    double A3 = hrc * v_params._m;


    // Integration using half implicit - level 2 variables found first in next time step

    // update the acceleration states - level 2 variables

    v_states._udot = v_states._wz*v_states._v + (1/mt)*((fx[0] + fx[1] + fx[2] + fx[3]) +  
                        (-v_params._mur*v_params._b + v_params._muf*v_params._a)*std::pow(v_states._wz,2) -
                        2.*hrc*v_params._m*v_states._wz*v_states._wx);

    // common denominator
    double denom =(A2*std::pow(A1,2) - 2.*A1*A3*v_params._jxz + v_params._jz*std::pow(A3,2) +
                    mt*std::pow(v_params._jxz,2) - A2*v_params._jz*mt);
    

    v_states._vdot = (E1*std::pow(v_params._jxz,2) - A1*A2*E2 + A1*E3*v_params._jxz + 
                        A3*E2*v_params._jxz - A2*E1*v_params._jz - A3*E3*v_params._jz) / denom;

    v_states._wxdot = (std::pow(A1,2)*E3 - A1*A3*E2 + A1*E1*v_params._jxz - A3*E1*v_params._jz +
                        E2*v_params._jxz*mt - E3*v_params._jz*mt) / denom;

    v_states._wzdot = (std::pow(A3,2)*E2 - A1*A2*E1 - A1*A3*E3 + A3*E1*v_params._jxz -
                        A2*E2*mt + E3*v_params._jxz*mt) / denom;      

    
    // update the level 1 varaibles using the next time step level 2 variable
    v_states._u = v_states._u + v_params._step * v_states._udot;
    v_states._v = v_states._v + v_params._step * v_states._vdot;
    v_states._wx = v_states._wx + v_params._step * v_states._wxdot;
    v_states._wz = v_states._wz + v_params._step * v_states._wzdot;


    // update the level 0 varaibles using the next time step level 1 varibales
    // over here still using the old psi and phi.. should we update psi and phi 
    // first and then use those????

    v_states._x = v_states._x + v_params._step * 
                    (v_states._u * std::cos(v_states._psi) - v_states._v * std::sin(v_states._psi));

    v_states._y = v_states._y + v_params._step * 
                    (v_states._u * std::sin(v_states._psi) + v_states._v * std::cos(v_states._psi));
    
    v_states._psi = v_states._psi + v_params._step * v_states._wz;
    v_states._phi = v_states._phi + v_params._step * v_states._wx; 


    // update the vertical forces
    // sketchy load transfer technique

    double Z1 = (v_params._m*G*v_params._b) / (2.*(v_params._a + v_params._b)) +
                (v_params._muf*G)/2.;
    
    double Z2 = ((v_params._muf*huf)/v_params._cf 
                    + v_params._m*v_params._b*(v_params._h - v_params._hrcf) /
                    (v_params._cf*(v_params._a + v_params._b)))*(v_states._vdot 
                    + v_states._wz*v_states._u);

    double Z3 = (v_params._krof * v_states._phi + v_params._brof * v_states._wx) / v_params._cf;
    
    double Z4 = ((v_params._m*v_params._h + v_params._muf*huf + v_params._mur*hur) *
                (v_states._udot - v_states._wz*v_states._v)) / (2.*(v_params._a + v_params._b));

    // evaluate the vertical forces for front
    v_states._fzlf = (Z1 - Z2 - Z3 - Z4) > 0. ? (Z1 - Z2 - Z3 - Z4) : 0.;
    v_states._fzrf = (Z1 + Z2 + Z3 - Z4) > 0. ? (Z1 + Z2 + Z3 - Z4) : 0.;

    Z1 = (v_params._m*G*v_params._a) / (2.*(v_params._a + v_params._b)) +
                (v_params._mur*G)/2.;

    Z2 =  ((v_params._mur*hur)/v_params._cr 
                    + v_params._m*v_params._a*(v_params._h - v_params._hrcr) /
                    (v_params._cr*(v_params._a + v_params._b)))*(v_states._vdot 
                    + v_states._wz*v_states._u);
    
    Z3 = (v_params._kror * v_states._phi + v_params._bror * v_states._wx) / v_params._cr;

    // evaluate vertical forces for the rear
    v_states._fzlr = (Z1 - Z2 - Z3 + Z4) > 0. ? (Z1 - Z2 - Z3 + Z4) : 0.;
    v_states._fzrr = (Z1 + Z2 + Z3 + Z4) > 0. ? (Z1 + Z2 + Z3 + Z4) : 0.; 

 
    

}

void EightDOF::vehToTireTransform(TMeasyState& tirelf_st,TMeasyState& tirerf_st,
                            TMeasyState& tirelr_st, TMeasyState& tirerr_st, 
                            const VehicleState& v_states, const VehicleParam& v_params, const std::vector <double>& controls){
                                
                             // get the controls and time out
                            double t = controls[0];
                            // Get the steering considering the mapping might be non linear
                            double delta = 0;
                            if(v_params._nonLinearSteer){
                                // Extract steer map
                                std::vector<MapEntry> steer_map = v_params._steerMap;
                                delta = getMapY(steer_map,controls[1]);
                            }
                            else{
                                delta = controls[1] * v_params._maxSteer;

                            }
                            double throttle = controls[2];
                            double brake = controls[3];

                            // left front
                            tirelf_st._fz = v_states._fzlf; 
                            tirelf_st._vsy = v_states._v + v_states._wz * v_params._a;
                            tirelf_st._vsx = (v_states._u - (v_states._wz * v_params._cf)/2.) * std::cos(delta) +
                                     tirelf_st._vsy * std::sin(delta);

                            // right front
                            tirerf_st._fz = v_states._fzrf;
                            tirerf_st._vsy = v_states._v + v_states._wz * v_params._a;
                            tirerf_st._vsx = (v_states._u + (v_states._wz * v_params._cf)/2.) * std::cos(delta) +
                                    tirerf_st._vsy * std::sin(delta);

                            // left rear - No steer
                            tirelr_st._fz = v_states._fzlr;
                            tirelr_st._vsy = v_states._v - v_states._wz * v_params._b;
                            tirelr_st._vsx = v_states._u - (v_states._wz * v_params._cr)/2.;

                            // rigth rear - No steer
                            tirerr_st._fz = v_states._fzrr;
                            tirerr_st._vsy = v_states._v - v_states._wz * v_params._b;
                            tirerr_st._vsx = v_states._u + (v_states._wz * v_params._cr)/2.;

                            }


void EightDOF::tireToVehTransform(TMeasyState& tirelf_st,TMeasyState& tirerf_st,
                            TMeasyState& tirelr_st, TMeasyState& tirerr_st,
                            const VehicleState& v_states, const VehicleParam& v_params, const std::vector <double>& controls){
                            
                            // get the controls and time out
                            double t = controls[0];
                            // Get the steering considering the mapping might be non linear
                            double delta = 0;
                            if(v_params._nonLinearSteer){
                                // Extract steer map
                                std::vector<MapEntry> steer_map = v_params._steerMap;
                                delta = getMapY(steer_map,controls[1]);
                            }
                            else{
                                delta = controls[1] * v_params._maxSteer;

                            }
                            double throttle = controls[2];
                            double brake = controls[3];                          
                            
                            double _fx,_fy;

                            // left front
                            _fx = tirelf_st._fx * std::cos(delta) - tirelf_st._fy * std::sin(delta);
                            _fy = tirelf_st._fx * std::sin(delta) + tirelf_st._fy * std::cos(delta);
                            tirelf_st._fx = _fx;
                            tirelf_st._fy = _fy;

                            // right front
                            _fx = tirerf_st._fx * std::cos(delta) - tirerf_st._fy * std::sin(delta);
                            _fy = tirerf_st._fx * std::sin(delta) + tirerf_st._fy * std::cos(delta);
                            tirerf_st._fx = _fx;
                            tirerf_st._fy = _fy;

                            // rear tires - No steer so no need to transform
                            }



// setting Vehicle parameters using a JSON file
void EightDOF::setVehParamsJSON(VehicleParam& v_params, const char *fileName){

    // Open the file
    FILE* fp = fopen(fileName,"r");

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    // parse the stream into DOM tree
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);


    if (d.HasParseError()) {
        std::cout << "Error with rapidjson:" << std::endl << d.GetParseError() << std::endl;
    }

    // the file should have all these parameters defined
    v_params._a = d["a"].GetDouble();
    v_params._b = d["b"].GetDouble();
    v_params._m = d["m"].GetDouble();
    v_params._h = d["h"].GetDouble();
    v_params._jz = d["jz"].GetDouble();
    v_params._jx = d["jx"].GetDouble();
    v_params._jxz = d["jxz"].GetDouble();
    v_params._cf = d["cf"].GetDouble();
    v_params._cr = d["cr"].GetDouble();
    v_params._muf = d["muf"].GetDouble();
    v_params._mur = d["mur"].GetDouble();
    v_params._hrcf = d["hrcf"].GetDouble();
    v_params._hrcr = d["hrcr"].GetDouble();
    v_params._krof = d["krof"].GetDouble();
    v_params._kror = d["kror"].GetDouble();
    v_params._brof = d["brof"].GetDouble();
    v_params._bror = d["bror"].GetDouble();

    // Non linear steering which maps the normalized steering input to wheel angle
    v_params._nonLinearSteer = d["nonLinearSteer"].GetBool();
    if(v_params._nonLinearSteer){
        unsigned int steerMapSize = d["steerMap"].Size();
        for(unsigned int i = 0; i < steerMapSize; i++){
            MapEntry m;
            m._x = d["steerMap"][i][0u].GetDouble();
            m._y = d["steerMap"][i][1u].GetDouble();
            v_params._steerMap.push_back(m);
        }
    }
    // If there is no non linear steer then the normalized steering input is just multiplied by the max steering wheel angle
    else{
        v_params._maxSteer = d["maxSteer"].GetDouble();
    }

    v_params._maxSteer = d["maxSteer"].GetDouble();
    
    // read the gear ratios
    unsigned int gears = d["gearRatios"].Size();
    for(unsigned int i = 0; i < gears; i++){
        v_params._gearRatios.push_back(d["gearRatios"][i].GetDouble());
    }

    v_params._tcbool = d["tcBool"].GetBool();

    // v_params._gearRatio = d["gearRatio"].GetDouble();
    // v_params._maxTorque = d["maxTorque"].GetDouble();
    v_params._upshift_RPS = d["upshiftRPM"].GetDouble() * rpm2rad;
    v_params._downshift_RPS = d["downshiftRPM"].GetDouble() * rpm2rad;
    v_params._maxBrakeTorque = d["maxBrakeTorque"].GetDouble();
    // v_params._maxSpeed = d["maxSpeed"].GetDouble();
    v_params._c1 = d["c1"].GetDouble();
    v_params._c0 = d["c0"].GetDouble();
    v_params._step = d["step"].GetDouble();

    v_params._throttleMod = d["throttleMod"].GetBool();
    // Read the powertrain map
    unsigned int map_size = d["torqueMap"].Size();
    for(unsigned int i = 0; i < map_size; i++){
        MapEntry m;
        m._x = d["torqueMap"][i][0u].GetDouble()*rpm2rad;
        m._y = d["torqueMap"][i][1u].GetDouble();
        v_params._powertrainMap.push_back(m);
    }


    // Read the losses map
    unsigned int map_size2 = d["lossesMap"].Size();
    for(unsigned int i = 0; i < map_size2; i++){
        MapEntry m;
        m._x = d["lossesMap"][i][0u].GetDouble()*rpm2rad;
        m._y = d["lossesMap"][i][1u].GetDouble();
        v_params._lossesMap.push_back(m);
    }


    // if we have a torque converter, we need this data
    if(v_params._tcbool){
        v_params._crankInertia = d["crankInertia"].GetDouble();
        unsigned int map_size_cf = d["capacityFactorMap"].Size();
        for(unsigned int i = 0; i < map_size_cf; i++){
            MapEntry m;
            m._x = d["capacityFactorMap"][i][0u].GetDouble();
            m._y = d["capacityFactorMap"][i][1u].GetDouble();
            v_params._CFmap.push_back(m);
        }

        unsigned int map_size_tr = d["torqueRatioMap"].Size();
        for(unsigned int i = 0; i < map_size_tr; i++){
            MapEntry m;
            m._x = d["torqueRatioMap"][i][0u].GetDouble();
            m._y = d["torqueRatioMap"][i][1u].GetDouble();
            v_params._TRmap.push_back(m);
        }
    }
}


///////////////////////////////////////////////////////////////////////////// Tire Functions /////////////////////////////////////////////////////////

/*
Code for the TM easy tire model implemented with the 8DOF model
*/
void EightDOF::tireInit(TMeasyParam& t_params){
    
    // calculates some critical values that are needed
    t_params._fzRdynco = (t_params._pn * (t_params._rdyncoP2n - 2.0 * t_params._rdyncoPn + 1.)) /
                            (2. * (t_params._rdyncoP2n - t_params._rdyncoPn));

    t_params._rdyncoCrit = InterpL(t_params._fzRdynco, t_params._rdyncoPn, t_params._rdyncoP2n,t_params._pn);

}



void EightDOF::tmxy_combined(double& f, double& fos, double s, double df0, double sm, double fm, double ss, double fs){

    double df0loc = 0.0;
    if (sm > 0.0) {
        df0loc = std::max(2.0 * fm / sm, df0);
    }

    if (s > 0.0 && df0loc > 0.0) {  // normal operating conditions
        if (s > ss) {               // full sliding
            f = fs;
            fos = f / s;
        } else {
            if (s < sm) {  // adhesion
                double p = df0loc * sm / fm - 2.0;
                double sn = s / sm;
                double dn = 1.0 + (sn + p) * sn;
                f = df0loc * sm * sn / dn;
                fos = df0loc / dn;
            } else {
                double a = std::pow(fm / sm, 2.0) / (df0loc * sm);  // parameter from 2. deriv. of f @ s=sm
                double sstar = sm + (fm - fs) / (a * (ss - sm));    // connecting point
                if (sstar <= ss) {                                  // 2 parabolas
                    if (s <= sstar) {
                        // 1. parabola sm < s < sstar
                        f = fm - a * (s - sm) * (s - sm);
                    } else {
                        // 2. parabola sstar < s < ss
                        double b = a * (sstar - sm) / (ss - sstar);
                        f = fs + b * (ss - s) * (ss - s);
                    }
                } else {
                    // cubic fallback function
                    double sn = (s - sm) / (ss - sm);
                    f = fm - (fm - fs) * sn * sn * (3.0 - 2.0 * sn);
                }
                fos = f / s;
            }
        }
    } else {
        f = 0.0;
        fos = 0.0;
    }


}


// Advance the tire to the next time step
// update the tire forces which will be used by the vehicle
void EightDOF::tireAdv(TMeasyState& t_states, const TMeasyParam& t_params, const VehicleState& v_states, const VehicleParam& v_params, 
                const std::vector <double>& controls){
    
    // get the controls and time out
    double t = controls[0];

    double delta = 0;
    if(v_params._nonLinearSteer){
        // Extract steer map
        std::vector<MapEntry> steer_map = v_params._steerMap;
        delta = getMapY(steer_map,controls[1]);

    }
    else{
        delta = controls[1] * v_params._maxSteer;

    }
    double throttle = controls[2];
    double brake = controls[3];

    // Get the whichTire based variables out of the way
    double fz = t_states._fz; // vertical force 
    double vsy = t_states._vsy; // y slip velocity
    double vsx = t_states._vsx; // x slip velocity

    // get our tire deflections so that we can get the loaded radius
    t_states._xt = fz / t_params._kt;
    t_states._rStat = t_params._r0 - t_states._xt;


    double r_eff;
    double rdynco;
    if(fz <= t_params._fzRdynco){
        rdynco = InterpL(fz, t_params._rdyncoPn, t_params._rdyncoP2n,t_params._pn);
        r_eff = rdynco * t_params._r0 + (1. - rdynco) * t_states._rStat; 
    }
    else {
        rdynco = t_params._rdyncoCrit;
        r_eff = rdynco * t_params._r0 + (1. - rdynco) * t_states._rStat;  
    }
    // with this r_eff, we can finalize the x slip velocity
    vsx = vsx - (t_states._omega * r_eff);

    // get the transport velocity - 0.01 here is to prevent singularity
    double vta = r_eff * std::abs(t_states._omega) + 0.01;

    // evaluate the slips
    double sx = -vsx / vta;
    double alpha;
    // only front wheel steering
    alpha = std::atan2(vsy,vta) - delta;
    double sy = -std::tan(alpha);

    // limit fz
    if(fz > t_params._pnmax){
        fz = t_params._pnmax;
    }

    // calculate all curve parameters through interpolation
    double dfx0 = InterpQ(fz, t_params._dfx0Pn, t_params._dfx0P2n, t_params._pn);
    double dfy0 = InterpQ(fz, t_params._dfy0Pn, t_params._dfy0P2n, t_params._pn);

    double fxm = InterpQ(fz, t_params._fxmPn, t_params._fxmP2n, t_params._pn);
    double fym = InterpQ(fz, t_params._fymPn, t_params._fymP2n, t_params._pn);
    
    double fxs = InterpQ(fz, t_params._fxsPn, t_params._fxsP2n, t_params._pn);
    double fys = InterpQ(fz, t_params._fysPn, t_params._fysP2n, t_params._pn);

    double sxm = InterpL(fz, t_params._sxmPn, t_params._sxmP2n, t_params._pn);
    double sym = InterpL(fz, t_params._symPn, t_params._symP2n, t_params._pn);

    double sxs = InterpL(fz, t_params._sxsPn, t_params._sxsP2n, t_params._pn);
    double sys = InterpL(fz, t_params._sysPn, t_params._sysP2n, t_params._pn);

    // slip normalizing factors
    double hsxn = sxm / (sxm + sym) + (fxm / dfx0) / (fxm / dfx0 + fym / dfy0);
    double hsyn = sym / (sxm + sym) + (fym / dfy0) / (fxm / dfx0 + fym / dfy0);


    // normalized slip
    double sxn = sx / hsxn;
    double syn = sy / hsyn;

    // combined slip
    double sc = std::hypot(sxn, syn);

    // cos and sine alphs
    double calpha;
    double salpha;
    if(sc > 0){
        calpha = sxn/sc;
        salpha = syn/sc;
    }
    else{
        calpha = std::sqrt(2.) / 2.;
        salpha = std::sqrt(2.) / 2.;
    }

    // resultant curve parameters in both directions
    double df0 = std::hypot(dfx0 * calpha * hsxn, dfy0 * salpha * hsyn);
    double fm  = std::hypot(fxm * calpha, fym * salpha);
    double sm = std::hypot(sxm * calpha / hsxn, sym * salpha / hsyn);
    double fs = std::hypot(fxs * calpha, fys * salpha);
    double ss = std::hypot(sxs * calpha / hsxn, sys * salpha / hsyn);

    // calculate force and force /slip from the curve characteritics
    double f,fos;
    tmxy_combined(f, fos, sc, df0, sm, fm, ss, fs);

    // static or "structural" force
    double Fx,Fy;
    if(sc > 0.){
        Fx = f * sx / sc;
        Fy = f * sy / sc;
    }
    else{
        Fx = 0.;
        Fy = 0.;
    }

    // rolling resistance with smoothing
    double vx_min = 0.;
    double vx_max = 0.;


    t_states._My = -sineStep(vta,vx_min,0.,vx_max,1.) * t_params._rr * fz * t_states._rStat * sgn(t_states._omega);

    
    double h;
    double dOmega;

    // some normalised slip velocities
    double vtxs = vta * hsxn;
    double vtys = vta * hsyn;


    // some varables needed in the loop
    double fxdyn, fydyn;
    double fxstr, fystr;
    double v_step = v_params._step;
    double tire_step = t_params._step;
    // now we integrate to the next vehicle time step
    double tEnd = t + v_step;
    while(t < tEnd){

        // ensure that we integrate exactly to step
        h = std::min(tire_step, tEnd - t);

        // always integrate using half implicit
        // just a placeholder to simplify the forumlae
        double dFx = -vtxs * t_params._cx / (vtxs * t_params._dx + fos);
        
        t_states._xedot = 1. / (1. - h * dFx) * 
                    (-vtxs * t_params._cx * t_states._xe - fos * vsx) /
                    (vtxs * t_params._dx + fos);

        t_states._xe = t_states._xe + h * t_states._xedot;

        double dFy = -vtys * t_params._cy / (vtys * t_params._dy + fos);
        t_states._yedot = (1. / (1. - h * dFy)) *
                    (-vtys * t_params._cy * t_states._ye - fos * (-sy * vta)) /
                    (vtys * t_params._dy + fos);

        t_states._ye = t_states._ye + h * t_states._yedot;

        // update the force since we need to update the force to get the omegas
        // some wierd stuff happens between the dynamic and structural force
        fxdyn = t_params._dx * (-vtxs * t_params._cx * t_states._xe - fos * vsx) /
                (vtxs * t_params._dx + fos) + t_params._cx * t_states._xe;
        
        fydyn = t_params._dy * ((-vtys * t_params._cy * t_states._ye - fos * (-sy * vta)) /
                (vtys * t_params._dy + fos)) + (t_params._cy * t_states._ye);

        
        fxstr = clamp(t_states._xe * t_params._cx + t_states._xedot * t_params._dx, -t_params._fxmP2n, t_params._fxmP2n);
        fystr = clamp(t_states._ye * t_params._cy + t_states._yedot * t_params._dy, -t_params._fymP2n, t_params._fymP2n);

        double weightx = sineStep(std::abs(vsx), 1., 1., 1.5, 0.);
        double weighty = sineStep(std::abs(-sy*vta), 1., 1., 1.5, 0.);

        // now finally get the resultant force
        t_states._fx = weightx * fxstr + (1.-weightx) * fxdyn;
        t_states._fy = weighty * fystr + (1.-weighty) * fydyn;

        t += h;
    }
}

// setting Tire parameters using a JSON file
void EightDOF::setTireParamsJSON(TMeasyParam& t_params, const char *fileName){
    // Open the file
    FILE* fp = fopen(fileName,"r");

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    // parse the stream into DOM tree
    rapidjson::Document d;
    d.ParseStream(is);
    fclose(fp);


    if (d.HasParseError()) {
        std::cout << "Error with rapidjson:" << std::endl << d.GetParseError() << std::endl;
    }

    // pray to what ever you believe in and hope that the json file has all these
    t_params._jw = d["jw"].GetDouble();
    t_params._rr = d["rr"].GetDouble();
    t_params._r0 = d["r0"].GetDouble();
    t_params._pn = d["pn"].GetDouble();
    t_params._pnmax = d["pnmax"].GetDouble();
    t_params._cx = d["cx"].GetDouble();
    t_params._cy = d["cy"].GetDouble();
    t_params._kt = d["kt"].GetDouble();
    t_params._dx = d["dx"].GetDouble();
    t_params._dy = d["dy"].GetDouble();
    t_params._rdyncoPn = d["rdyncoPn"].GetDouble();
    t_params._rdyncoP2n = d["rdyncoP2n"].GetDouble();
    t_params._fzRdynco = d["fzRdynco"].GetDouble();
    t_params._rdyncoCrit = d["rdyncoCrit"].GetDouble();

    t_params._dfx0Pn = d["dfx0Pn"].GetDouble();
    t_params._dfx0P2n = d["dfx0P2n"].GetDouble();
    t_params._fxmPn = d["fxmPn"].GetDouble();
    t_params._fxmP2n = d["fxmP2n"].GetDouble();
    t_params._fxsPn = d["fxsPn"].GetDouble();
    t_params._fxsP2n = d["fxsP2n"].GetDouble();
    t_params._sxmPn = d["sxmPn"].GetDouble();
    t_params._sxmP2n = d["sxmP2n"].GetDouble();
    t_params._sxsPn = d["sxsPn"].GetDouble();
    t_params._sxsP2n = d["sxsP2n"].GetDouble();

    t_params._dfy0Pn = d["dfy0Pn"].GetDouble();
    t_params._dfy0P2n = d["dfy0P2n"].GetDouble();
    t_params._fymPn = d["fymPn"].GetDouble();
    t_params._fymP2n = d["fymP2n"].GetDouble();
    t_params._fysPn = d["fysPn"].GetDouble();
    t_params._fysP2n = d["fysP2n"].GetDouble();
    t_params._symPn = d["symPn"].GetDouble();
    t_params._symP2n = d["symP2n"].GetDouble();
    t_params._sysPn = d["sysPn"].GetDouble();
    t_params._sysP2n = d["sysP2n"].GetDouble();

    t_params._step = d["step"].GetDouble();

}
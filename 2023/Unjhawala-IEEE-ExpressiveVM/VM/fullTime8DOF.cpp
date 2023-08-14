#include <iostream>
#include <stdint.h>
#include <chrono>
#include "../utils.h"
#include "Eightdof.h"


using std::chrono::high_resolution_clock;
using std::chrono::duration;
using namespace EightDOF;

/*
Test file for the eight DOF model with TMEasy implemented in c++
*/


int main(int argc, char *argv[]){

    // declaring start and stop clocks
    high_resolution_clock::time_point start; 
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // Input files
    std::string fileName = "./inputs/acc3.txt";

    
    // Vehicle parameters JSON file
    char *vehParamsJSON = (char *)"./jsons/HMMWV.json";

    // Tire parameters JSON file
    char *tireParamsJSON = (char *)"./jsons/TMeasy.json";



    std::vector<Entry> driverData;
    driverInput(driverData, fileName);

    // initialize controls to 0
    std::vector <double> controls(4,0);

    // lets initialize a vehicle struct and see where we land
    VehicleState veh1_st;
    VehicleParam veh1_param;

    // Set vehicle parameters from JSON file
    setVehParamsJSON(veh1_param,vehParamsJSON);
    vehInit(veh1_st,veh1_param);

    // lets define our tires, we have 4 different 
    // tires so 4 states
    TMeasyState tirelf_st;
    TMeasyState tirerf_st;
    TMeasyState tirelr_st;
    TMeasyState tirerr_st;

    // but all of them have the same parameters
    // so only one parameter structure
    TMeasyParam tire_param;

    // set the tire parameters from a JSON file
    setTireParamsJSON(tire_param,tireParamsJSON);

    // // now we initialize each of our parameters
    tireInit(tire_param);


    double endTime = 10.0;

    veh1_param._step = 0.001;
    tire_param._step = 0.001;
    double step = veh1_param._step;

    // now lets run our simulation
    double t = 0;
    int timeStepNo = 0; // time step counter



    // Get the starting timestamp
    start = high_resolution_clock::now();

    while(t < (endTime - step/10)){
        // get the controls for this time step
        getControls(controls, driverData, t);
        
        // transform velocities and other needed quantities from
        // vehicle frame to tire frame
        vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls);

        // advance our 4 tires
        tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls);
        tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls);

        // modify controls for our rear tires as they dont take steering
        std::vector <double> mod_controls = {controls[0],0,controls[2],controls[3]};
        tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls);
        tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls);


        // Evalaute the powertrain and advance the tire angular velocities and the angular
        // velocity of the crank shaft (if we have Torque converter on)

        evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls);

        // transform tire forces to vehicle frame
        tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls);

        // copy the useful stuff that needs to be passed onto the vehicle
        std::vector<double> fx = {tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx};
        std::vector<double> fy = {tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy};
        double huf = tirelf_st._rStat;
        double hur = tirerr_st._rStat;
        

        vehAdv(veh1_st,veh1_param, fx,fy,huf,hur);

        
        t += step;
        timeStepNo += 1;
    }

    // Get the ending timestamp
    end = high_resolution_clock::now();

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout<<duration_sec.count()<<"\n";
    

    return 0;
}
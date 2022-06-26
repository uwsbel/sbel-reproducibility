// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Main driver function for a vehicle specified through JSON files.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================
#include "chrono/core/ChFrameMoving.h"
#include "chrono/core/ChRealtimeStep.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChIrrGuiDriver.h"

#include "chrono_vehicle/driver/ChDataDriver.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledTrailer.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "chrono/utils/ChFilters.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono/core/ChVector.h"
#include <cmath>
#include "chrono/core/ChTimer.h"

using namespace chrono;
using namespace chrono::vehicle;


class Vehicle_Model {
  public:
    virtual std::string ModelName() const = 0;
    virtual std::string VehicleJSON() const = 0;
    virtual std::string TireJSON() const = 0;
    virtual std::string PowertrainJSON() const = 0;
    virtual double CameraDistance() const = 0;
};



class Calib_Model : public Vehicle_Model {
  public:
    virtual std::string ModelName() const override { return "Calibration"; }
    virtual std::string VehicleJSON() const override {
        // return "calib_mod/vehicle/Vehicle.json";
        return "calib_mod/vehicle/Vehicle_4WD.json";
    }
    virtual std::string TireJSON() const override {
        ////return "calib_mod/tire/RigidTire.json";
        ////return "calib_mod/tire/FialaTire.json";
        ////return "calib_mod/tire/TMeasyTire.json";
        ////return "calib_mod/tire/Pac89Tire.json";
        return "calib_mod/tire/FialaTire.json";
    }
    virtual std::string PowertrainJSON() const override {
        // return "calib_mod/powertrain/ShaftsPowertrain.json";
        ////return "calib_mod/powertrain/SimpleCVTPowertrain.json";
        return "calib_mod/powertrain/SimplePowertrain.json";
    }
    virtual double CameraDistance() const override { return 6.0; }
};




// Current vehicle model selection
auto vehicle_model = Calib_Model();

// JSON files for terrain
std::string rigidterrain_file("terrain/RigidPlane.json");


// Initial vehicle position and orientation
ChVector<> initLoc(0, 0, 0.5);
// Changing the initial yaw to 0
double initYaw = 0 * CH_C_DEG_TO_RAD;

// Simulation step size 
double step_size = 2e-4;

// Output directory
const std::string out_dir = GetChronoOutputPath() + "WHEELED_JSON";

// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // Whether the outputs should be saved into a csv file
    bool data_output = true;
    // Create the vehicle system
    WheeledVehicle vehicle(vehicle::GetDataFile(vehicle_model.VehicleJSON()), ChContactMethod::SMC);
    vehicle.Initialize(ChCoordsys<>(initLoc, Q_from_AngZ(initYaw)));
    vehicle.GetChassis()->SetFixed(false);
    vehicle.SetChassisVisualizationType(VisualizationType::NONE);
    vehicle.SetChassisRearVisualizationType(VisualizationType::NONE);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(vehicle_model.PowertrainJSON()));
    vehicle.InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle.GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(vehicle_model.TireJSON()));
            vehicle.InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    // Containing system
    auto system = vehicle.GetSystem();

    // Create the terrain
    RigidTerrain terrain(system, vehicle::GetDataFile(rigidterrain_file));
    terrain.Initialize();

    // Create Irrilicht visualization
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    // vis->SetWindowTitle("Vehicle demo - JSON specification");
    // vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), vehicle_model.CameraDistance(), 0.5);
    vis->Initialize();
    // vis->AddTypicalLights();
    // vis->AddSkyBox();
    // vis->AddLogo();
    vehicle.SetVisualSystem(vis);


    // Create data driven driver
    ChDataDriver driver(vehicle, vehicle::GetDataFile("calib_mod/driver/test_set.txt"));
    driver.Initialize();

    // Initialize output directories
    std::string veh_dir = out_dir + "/" + vehicle_model.ModelName();

    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(veh_dir))) {
        std::cout << "Error creating directory " << veh_dir << std::endl;
        return 1;
    }

    // Generate JSON information with available output channels
    std::string out_json = vehicle.ExportComponentList();
    // std::cout << out_json << std::endl;
    // vehicle.ExportComponentList(veh_dir + "/component_list.json");

    // vehicle.LogSubsystemTypes();


    std::cout<<"The Vehicle Mass : "<<vehicle.GetMass()<<std::endl;
    std::cout<<"Front Unsprung Mass :"<<(vehicle.GetSuspension(0)->GetMass()/2 + vehicle.GetWheel(0,RIGHT)->GetMass()*2 + 
    vehicle.GetTire(0,RIGHT)->GetMass()*2 + vehicle.GetBrake(0,RIGHT)->GetMass()*2)<<std::endl;
    // std::cout<<"Front unsprung mass : "<<(vehicle.GetSuspension(0)->GetMass())<<std::endl;
    std::cout<<"Rear Unsprung Mass :"<<(vehicle.GetSuspension(1)->GetMass()/2 + vehicle.GetWheel(1,RIGHT)->GetMass()*2 + 
    vehicle.GetTire(1,RIGHT)->GetMass()*2 + vehicle.GetBrake(1,RIGHT)->GetMass()*2)<<std::endl;
    // std::cout<<"Rear unsprung mass : "<<(vehicle.GetSuspension(1)->GetMass())<<std::endl;
    std::cout<<"Unsprung Mass : "<<(vehicle.GetChassis()->GetMass() + vehicle.GetSteering(0)->GetMass()) <<std::endl;
    std::cout<<"The Vehicle Inertia matrix is "<<vehicle.GetInertia()<<std::endl;
    std::cout<<"The Vehicle Maximum Steeting angle is "<<vehicle.GetMaxSteeringAngle()<<std::endl;
    std::cout<<"The Sprung Mass CG height is "<<vehicle.GetChassis()->GetTransform().TransformLocalToParent(vehicle.GetChassis()->GetCOMFrame().GetPos())<<std::endl;
    std::cout<<"The Vehicle CG height is "<<vehicle.GetCOMFrame().GetPos()<<std::endl;
    std::cout<<"The Vehicle Front Track Width is "<<vehicle.GetWheeltrack(0)<<std::endl;
    std::cout<<"The Vehicle Rear Track Width is "<<vehicle.GetWheeltrack(1)<<std::endl;
    std::cout<<"Suspension COM "<<vehicle.GetSuspension(0)->GetTransform().GetPos()<<std::endl;
    std::cout<<"Spindle Global Position "<<vehicle.GetSuspension(0)->GetSpindle(LEFT)->GetPos()<<std::endl;

    // Roll centre computed using a python script
    auto roll_c = ChVector< double >(0.0,0.0,-0.14751);
    std::cout<<"Global position of front roll centre "<<vehicle.GetChassis()->GetTransform().TransformLocalToParent(
        vehicle.GetSuspension(0)->GetCOMFrame().TransformLocalToParent(roll_c))
    <<std::endl;
     

    


    // Initilaizing the CSV writer to write the output file
    utils::CSV_writer csv(",");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);

    csv << "time";
    csv << "Steering_input";
    csv << "x";
    csv << "y";
    csv << "vx";
    csv << "vy";
    csv << "ax";
    csv << "ay";
    csv << "yaw";
    csv << "roll";
    csv << "yaw_rate";
    csv << "roll_rate";
    csv << "slip_angle";
    csv << "long_slip";
    csv << "toe_in_r";
    csv << "toe_in_avg";
    csv << "toe_in_l";
    csv << "wlf";
    csv << "wlr";
    csv << "wrf";
    csv << "wrr";
    csv <<"tiredef_rf";
    csv <<"tiredef_rr";
    csv <<"tiredef_lf";
    csv <<"tiredef_lr";
    csv <<"sp_tor";
    csv << std::endl;


    double time = 0.;
    int time_step = 0.;

    while (vis->Run()) {
        // Render scene
        // vis->BeginScene();
        // vis->DrawAll();
        // vis->EndScene();

        // Get driver inputs
        ChDriver::Inputs driver_inputs = driver.GetInputs();
        time = vehicle.GetSystem()->GetChTime();

        if(time_step % 50 == 0){


            // Get the veclocities with respect to the local frame of reference
            auto chassis_vel_abs = vehicle.GetPointVelocity(vehicle.GetChassis()->GetCOMFrame().GetPos());
            auto chassis_vel_veh = vehicle.GetTransform().TransformDirectionParentToLocal(chassis_vel_abs);

            // Get the vehicle accelerations
            auto chassis_acc_abs = vehicle.GetPointAcceleration(vehicle.GetChassis()->GetCOMFrame().GetPos());
            auto chassis_acc_veh = vehicle.GetTransform().TransformDirectionParentToLocal(chassis_acc_abs);

            // Orientation angles of the vehicle
            // auto rot = vehicle.GetTransform().GetRot();
            auto rot_v = vehicle.GetRot();
            auto euler123 = rot_v.Q_to_Euler123();

            // Orientation rates of the vehicle
            // auto rot_dot = ChFrameMoving(vehicle.GetTransform()).GetWvel_par();
            auto move_frame = ChFrameMoving<double>(vehicle.GetChassis()->GetCOMFrame());
            // auto rot_dot = move_frame.GetWvel_loc();
            auto rot_dot = move_frame.GetPos_dt();


            // Below caluclations to caluclate the toe-in angle

///////////////////////////////////////////////////////////////////////RIGHT WHEEL - TOE-IN///////////////////////////////////////////////
            // Get the Right wheel state
            auto state = vehicle.GetWheel(0,VehicleSide {RIGHT})->GetState();
            // Wheel normal (expressed in global frame)
            ChVector<> wheel_normal = state.rot.GetYaxis();

            // Terrain normal at wheel location (expressed in global frame)
            ChVector<> Z_dir = terrain.GetNormal(state.pos);

            // Longitudinal (heading) and lateral directions, in the terrain plane
            ChVector<> X_dir = Vcross(wheel_normal, Z_dir);
            X_dir.Normalize();
            ChVector<> Y_dir = Vcross(Z_dir, X_dir);

            // Tire reference coordinate system
            ChMatrix33<> rot;
            rot.Set_A_axis(X_dir, Y_dir, Z_dir);
            ChCoordsys<> tire_csys(state.pos, rot.Get_A_quaternion());

            // Express wheel normal in tire frame
            ChVector<> n = tire_csys.TransformDirectionParentToLocal(wheel_normal);

            // Wheel normal in the Vehicle frame
            ChVector<> n_v = vehicle.GetTransform().TransformDirectionParentToLocal(wheel_normal);

            // Toe-in
            auto toe_in_r = std::atan2(n_v.x(),n_v.y());


////////////////////////////////////////////////////////////////////////////////LEFT WHEEL - TOE-IN/////////////////////////////////////////////
            // Same process for the left wheel
            auto state_l = vehicle.GetWheel(0,VehicleSide {LEFT})->GetState();
            // Wheel normal (expressed in global frame)
            ChVector<> wheel_normal_l = state_l.rot.GetYaxis();
            // Terrain normal at wheel location (expressed in global frame)
            ChVector<> Z_dir_l = terrain.GetNormal(state_l.pos);

            // Longitudinal (heading) and lateral directions, in the terrain plane
            ChVector<> X_dir_l = Vcross(wheel_normal_l, Z_dir_l);
            X_dir_l.Normalize();
            ChVector<> Y_dir_l = Vcross(Z_dir_l, X_dir_l);

            // Tire reference coordinate system
            ChMatrix33<> rot_l;
            rot.Set_A_axis(X_dir_l, Y_dir_l, Z_dir_l);
            ChCoordsys<> tire_csys_l(state_l.pos, rot_l.Get_A_quaternion());

            // Express wheel normal in tire frame
            ChVector<> n_l = tire_csys_l.TransformDirectionParentToLocal(wheel_normal_l);

            // Wheel normal in the Vehicle frame
            ChVector<> n_v_l = vehicle.GetTransform().TransformDirectionParentToLocal(wheel_normal_l);


            auto toe_in_l = std::atan2(n_v_l.x(),n_v_l.y());


////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Slip angle just to check
            double slip_angle = vehicle.GetTire(0, VehicleSide {RIGHT})->GetSlipAngle();
            double long_slip = vehicle.GetTire(1, VehicleSide {RIGHT})->GetLongitudinalSlip();
            auto omega = vehicle.GetChassisBody()->GetWvel_loc();


            // Write CSV file
            csv << time;
            csv << driver.GetSteering();
            csv << vehicle.GetPos().x();
            csv << vehicle.GetPos().y();
            csv << chassis_vel_veh[0];
            csv << chassis_vel_veh[1];
            csv << chassis_acc_veh[0];
            csv << chassis_acc_veh[1];
            csv << euler123[2];
            csv << euler123[0];
            csv << omega[2];
            csv << omega[0];
            csv << slip_angle;
            csv << long_slip;
            csv << toe_in_r;
            csv << (toe_in_l+toe_in_r)/2;
            csv << toe_in_l;
            csv << vehicle.GetSpindleAngVel(0,LEFT)[1];
            csv << vehicle.GetSpindleAngVel(1,LEFT)[1];
            csv << vehicle.GetSpindleAngVel(0,RIGHT)[1];
            csv << vehicle.GetSpindleAngVel(1,RIGHT)[1];
            csv << vehicle.GetTire(0,RIGHT)->GetDeflection();
            csv << vehicle.GetTire(1,RIGHT)->GetDeflection();
            csv << vehicle.GetTire(0,LEFT)->GetDeflection();
            csv << vehicle.GetTire(1,LEFT)->GetDeflection();
            csv << vehicle.GetDriveline()->GetSpindleTorque(0,VehicleSide {LEFT});
            csv << std::endl;
        }
            

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        vehicle.Synchronize(time, driver_inputs, terrain);
        terrain.Synchronize(time);


        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        vehicle.Advance(step_size);
        terrain.Advance(step_size);


        time_step+=1;

        // End simulation
        if (time >= 12.5)
            break;  
    }


    if (data_output) {
        csv.write_to_file(veh_dir + "/test___.csv");
    }

    return 0;
}

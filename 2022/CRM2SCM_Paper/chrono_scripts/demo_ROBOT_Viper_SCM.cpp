// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2021 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Jason Zhou
// =============================================================================
//
// Demo to show Viper Rover operated on SCM Terrain
//
// =============================================================================

#include "chrono_models/robot/viper/Viper.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/utils/ChUtilsInputOutput.h"

// #include "chrono_irrlicht/ChVisualSystemIrrlicht.h"

#include "chrono/physics/ChLinkMotorRotationSpeed.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
// using namespace chrono::irrlicht;
using namespace chrono::geometry;
using namespace chrono::viper;

// using namespace irr;

bool output = true;
// const std::string out_dir = GetChronoOutputPath() + "SCM_Viper";
std::string out_dir = GetChronoOutputPath() + "SCM_Viper/final_real_wheel_new_Janosi_Ks_15s/slip0";

// SCM grid spacing
double mesh_resolution = 0.01;

// Enable/disable bulldozing effects
bool enable_bulldozing = false;

// Enable/disable moving patch feature
bool enable_moving_patch = false;

// If true, use provided callback to change soil properties based on location
bool var_params = false;

// Define Viper rover wheel type
ViperWheelType wheel_type = ViperWheelType::RealWheel;

// wheel specifics
double wheel_rad = 0.25;
double wheel_slip = 0.0;
double wheel_vel = 0.25;
double wheel_ang = wheel_vel / ( wheel_rad * (1.0 - wheel_slip) );

// spring specifics
// double K_s = 1000.0;
// double Ini_s = -3.0; // initial position that kicked in

// Rover initial location
ChVector<> init_loc(-3.5, 0, 0.27);

// Simulation time and stepsize
double dT = 5e-4;
double total_time = 15.0;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();

// Custom callback for setting location-dependent soil properties.
// Note that the location is given in the SCM reference frame.
class MySoilParams : public vehicle::SCMDeformableTerrain::SoilParametersCallback {
  public:
    virtual void Set(const ChVector<>& loc,
                     double& Bekker_Kphi,
                     double& Bekker_Kc,
                     double& Bekker_n,
                     double& Mohr_cohesion,
                     double& Mohr_friction,
                     double& Janosi_shear,
                     double& elastic_K,
                     double& damping_R) override {
        Bekker_Kphi = 0.82e6;
        Bekker_Kc = 0.14e4;
        Bekker_n = 1.0;
        Mohr_cohesion = 0.017e4;
        Mohr_friction = 35.0;
        Janosi_shear = 1.78e-2;
        elastic_K = 2e8;
        damping_R = 3e4;
    }
};

// Use custom material for the Viper Wheel
bool use_custom_mat = true;

// Return customized wheel material parameters
std::shared_ptr<ChMaterialSurface> CustomWheelMaterial(ChContactMethod contact_method) {
    float mu = 0.7f;   // coefficient of friction
    float cr = 0.1f;   // coefficient of restitution
    float Y = 2e7f;    // Young's modulus
    float nu = 0.3f;   // Poisson ratio
    float kn = 2e5f;   // normal stiffness
    float gn = 40.0f;  // normal viscous damping
    float kt = 2e5f;   // tangential stiffness
    float gt = 20.0f;  // tangential viscous damping

    switch (contact_method) {
        case ChContactMethod::NSC: {
            auto matNSC = chrono_types::make_shared<ChMaterialSurfaceNSC>();
            matNSC->SetFriction(mu);
            matNSC->SetRestitution(cr);
            return matNSC;
        }
        case ChContactMethod::SMC: {
            auto matSMC = chrono_types::make_shared<ChMaterialSurfaceSMC>();
            matSMC->SetFriction(mu);
            matSMC->SetRestitution(cr);
            matSMC->SetYoungModulus(Y);
            matSMC->SetPoissonRatio(nu);
            matSMC->SetKn(kn);
            matSMC->SetGn(gn);
            matSMC->SetKt(kt);
            matSMC->SetGt(gt);
            return matSMC;
        }
        default:
            return std::shared_ptr<ChMaterialSurface>();
    }
}

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // Global parameter for moving patch size:
    double wheel_range = 0.5;
    ////double body_range = 1.2;

    // Create a Chrono::Engine physical system
    ChSystemSMC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    if (argc == 2) {
        wheel_slip = 0.01 * std::stod(argv[1]);
        wheel_ang = wheel_vel / ( wheel_rad * (1.0 - wheel_slip) );
        out_dir = out_dir + std::string(argv[1]) + "/";
    }

    // Initialize output
    if (output) {
        if (!filesystem::create_directory(filesystem::path(out_dir))) {
            std::cout << "Error creating directory " << out_dir << std::endl;
            return 1;
        }
    }
    utils::CSV_writer csv(" ");

    // Create the rover
    auto driver = chrono_types::make_shared<ViperSpeedDriver>(0.1, wheel_ang);

    Viper viper(&sys, wheel_type);

    viper.SetDriver(driver);
    if (use_custom_mat)
        viper.SetWheelContactMaterial(CustomWheelMaterial(ChContactMethod::NSC));

    viper.Initialize(ChFrame<>(init_loc, QUNIT));

    // Get wheels and bodies to set up SCM patches
    auto Wheel_1 = viper.GetWheel(ViperWheelID::V_LF)->GetBody();
    auto Wheel_2 = viper.GetWheel(ViperWheelID::V_RF)->GetBody();
    auto Wheel_3 = viper.GetWheel(ViperWheelID::V_LB)->GetBody();
    auto Wheel_4 = viper.GetWheel(ViperWheelID::V_RB)->GetBody();
    auto body = viper.GetChassis()->GetBody();

    //
    // THE DEFORMABLE TERRAIN
    //

    // Create the 'deformable terrain' object
    vehicle::SCMDeformableTerrain terrain(&sys);

    // Displace/rotate the terrain reference plane.
    // Note that SCMDeformableTerrain uses a default ISO reference frame (Z up). Since the mechanism is modeled here in
    // a Y-up global frame, we rotate the terrain plane by -90 degrees about the X axis.
    // Note: Irrlicht uses a Y-up frame
    terrain.SetPlane(ChCoordsys<>(ChVector<>(0, 0, 0)));

    // Use a regular grid:
    double length = 14;
    double width = 4;
    terrain.Initialize(length, width, mesh_resolution);

    // Set the soil terramechanical parameters
    if (var_params) {
        // Here we use the soil callback defined at the beginning of the code
        auto my_params = chrono_types::make_shared<MySoilParams>();
        terrain.RegisterSoilParametersCallback(my_params);
    } else {
        // If var_params is set to be false, these parameters will be used
        terrain.SetSoilParameters(2.22e6, // Bekker Kphi
                                 -1.1e5,  // Bekker Kc
                                  1.2,    // Bekker n exponent
                                  2497,   // Mohr cohesive limit (Pa)
                                  24,     // Mohr friction limit (degrees)
                                  0.00305,  // Janosi shear coefficient (m)
                                  4e7,    // Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
                                  1e6     // Damping (Pa s/m), proportional to negative vertical speed (optional)
        );
    }

    // Set up bulldozing factors
    if (enable_bulldozing) {
        terrain.EnableBulldozing(true);  // inflate soil at the border of the rut
        terrain.SetBulldozingParameters(
            55,  // angle of friction for erosion of displaced material at the border of the rut
            1,   // displaced material vs downward pressed material.
            5,   // number of erosion refinements per timestep
            6);  // number of concentric vertex selections subject to erosion
    }

    // We need to add a moving patch under every wheel
    // Or we can define a large moving patch at the pos of the rover body
    if (enable_moving_patch) {
        terrain.AddMovingPatch(Wheel_1, ChVector<>(0, 0, 0), ChVector<>(0.5, 2 * wheel_range, 2 * wheel_range));
        terrain.AddMovingPatch(Wheel_2, ChVector<>(0, 0, 0), ChVector<>(0.5, 2 * wheel_range, 2 * wheel_range));
        terrain.AddMovingPatch(Wheel_3, ChVector<>(0, 0, 0), ChVector<>(0.5, 2 * wheel_range, 2 * wheel_range));
        terrain.AddMovingPatch(Wheel_4, ChVector<>(0, 0, 0), ChVector<>(0.5, 2 * wheel_range, 2 * wheel_range));
    }

    // Set some visualization parameters: either with a texture, or with falsecolor plot, etc.
    terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_PRESSURE, 0, 20000);

    terrain.SetMeshWireframe(true);

    // Create the Irrlicht visualization sys
    // auto vis = chrono_types::make_shared<ChVisualSystemIrrlicht>();
    // vis->AttachSystem(&sys);
    // vis->SetCameraVertical(CameraVerticalDir::Z);
    // vis->SetWindowSize(800, 600);
    // vis->SetWindowTitle("Viper Rover on SCM");
    // vis->Initialize();
    // vis->AddLogo();
    // vis->AddSkyBox();
    // vis->AddCamera(ChVector<>(2.0, 0.0, 1.4), ChVector<>(0, 0, wheel_range));
    // vis->AddTypicalLights();
    // vis->AddLightWithShadow(ChVector<>(-5.0, -0.5, 8.0), ChVector<>(-1, 0, 0), 100, 1, 35, 85, 512,
    //                         ChColor(0.8f, 0.8f, 0.8f));
    // vis->EnableShadows();


    {
        auto box = chrono_types::make_shared<ChBodyEasyBox>(10, 10, 0.02, 1000, false, false);
        box->SetPos(ChVector<>(0, 0, 0));
        box->SetBodyFixed(true);
        sys.Add(box);

        // Create the chassis of the testing rig
        auto chassis = chrono_types::make_shared<ChBody>();
        chassis->SetMass(100.0);
        chassis->SetPos(init_loc);
        chassis->SetCollide(false);
        chassis->SetBodyFixed(false);

        // Add geometry of the chassis.
        chassis->GetCollisionModel()->ClearModel();
        chrono::utils::AddBoxGeometry(chassis.get(), CustomWheelMaterial(ChContactMethod::NSC), ChVector<>(0.1, 0.1, 0.1), ChVector<>(0, 0, 0));
        chassis->GetCollisionModel()->BuildModel();
        sys.AddBody(chassis);

        // Create the axle
        auto axle = chrono_types::make_shared<ChBody>();
        axle->SetMass(10.0);
        axle->SetPos(init_loc + + ChVector<>(0, 0, 1));
        axle->SetCollide(false);
        axle->SetBodyFixed(false);

        // Add geometry of the axle.
        axle->GetCollisionModel()->ClearModel();
        chrono::utils::AddSphereGeometry(axle.get(), CustomWheelMaterial(ChContactMethod::NSC), 0.5, ChVector<>(0, 0, 0));
        axle->GetCollisionModel()->BuildModel();
        sys.AddBody(axle);

        // Connect the chassis to the containing bin (ground) through a translational joint and create a linear actuator.
        auto prismatic1 = chrono_types::make_shared<ChLinkLockPrismatic>();
        prismatic1->Initialize(box, chassis, ChCoordsys<>(chassis->GetPos(), Q_from_AngY(CH_C_PI_2)));
        prismatic1->SetName("prismatic_chassis_ground");
        sys.AddLink(prismatic1);

        auto actuator_fun = chrono_types::make_shared<ChFunction_Ramp>(0.0, wheel_vel);
        actuator->Initialize(box, chassis, false, ChCoordsys<>(chassis->GetPos(), QUNIT),
                            ChCoordsys<>(chassis->GetPos() + ChVector<>(1, 0, 0), QUNIT));
        actuator->SetName("actuator");
        actuator->SetDistanceOffset(1);
        actuator->SetActuatorFunction(actuator_fun);
        sys.AddLink(actuator);

        // Connect the axle to the chassis through a vertical translational joint.
        auto prismatic2 = chrono_types::make_shared<ChLinkLockPrismatic>();
        prismatic2->Initialize(chassis, axle, ChCoordsys<>(chassis->GetPos(), QUNIT));
        prismatic2->SetName("prismatic_axle_chassis");
        sys.AddLink(prismatic2);

        // Connect the rover body to the axle through a engine joint.
        auto lock_link = chrono_types::make_shared<ChLinkLockLock>();
        auto rover_body = viper.GetChassis()->GetBody();
        lock_link->SetName("rover_axle_lock");
        lock_link->Initialize(axle, rover_body, ChCoordsys<>(chassis->GetPos(), QUNIT));
        sys.AddLink(lock_link);
    }


    // Write position and velocity to file
    std::ofstream ofile;
    std::ofstream myDBP;
    if (output) {
        ofile.open(out_dir + "/body_position.txt", std::ios::trunc);
        myDBP.open(out_dir + "/DBP.txt", std::ios::trunc);
    }

    double time = 0.0;
    while (time < total_time) {
        // vis->BeginScene();
        // vis->GetActiveCamera()->setTarget(core::vector3dfCH(body->GetPos()));
        // vis->Render();
        // tools::drawColorbar(vis.get(), 0, 20000, "Pressure yield [Pa]", 1180);
        // vis->EndScene();

        if (output) {
            // write drive torques of all four wheels into file
            csv << sys.GetChTime() << viper.GetWheelTracTorque(ViperWheelID::V_LF)
                << viper.GetWheelTracTorque(ViperWheelID::V_RF) << viper.GetWheelTracTorque(ViperWheelID::V_LB)
                << viper.GetWheelTracTorque(ViperWheelID::V_RB) << std::endl;
        }

        // Add a spring between rover and ground to measure dbp
        // double wheel_vel = 0.25 * (Wheel_1->GetPos_dt().x() + Wheel_2->GetPos_dt().x() + 
        //                            Wheel_3->GetPos_dt().x() + Wheel_4->GetPos_dt().x());
        // double wheel_ang = 0.25 * (Wheel_1->GetWvel_loc().z() + Wheel_2->GetWvel_loc().z() + 
        //                             Wheel_3->GetWvel_loc().z() + Wheel_4->GetWvel_loc().z());
        // double wheel_slip = 1.0 - wheel_vel / (wheel_ang * wheel_rad);
        double wheel_torque = viper.GetWheelTracTorque(ViperWheelID::V_LF)
                            + viper.GetWheelTracTorque(ViperWheelID::V_RF)
                            + viper.GetWheelTracTorque(ViperWheelID::V_LB)
                            + viper.GetWheelTracTorque(ViperWheelID::V_RB);
        double drawbar_pull = actuator->Get_react_force().x();

        // double spring_ext;
        // double spring_force;
        // if (body->GetPos().x() > Ini_s) {
        //     spring_ext = body->GetPos().x() - Ini_s;
        // } else {
        //     spring_ext = 0.0;
        // }
        // spring_force = K_s * spring_ext;
        // ChVector<> SpringF = ChVector<>(-spring_force, 0.0, 0.0);
        // body->Empty_forces_accumulators();
        // body->Accumulate_force(SpringF, body->GetPos(), false);

        if (output) {
            ofile << time << "  " << body->GetPos() << "  " << body->GetPos_dt() << std::endl;
            myDBP << time << "\t" 
                  << body->GetPos().x() << "\t" 
                  << wheel_vel << "\t" 
                  << wheel_ang << "\t" 
                  << wheel_slip << "\t" 
                  << drawbar_pull << "\t"
                  << wheel_torque << "\n" ;
        }

        sys.DoStepDynamics(dT);
        viper.Update();
        ////terrain.PrintStepStatistics(std::cout);
        time += dT;
    }

    if (output) {
        csv.write_to_file(out_dir + "/output.dat");
    }

    if (output){
        ofile.close();
        myDBP.close();
    }

    return 0;
}

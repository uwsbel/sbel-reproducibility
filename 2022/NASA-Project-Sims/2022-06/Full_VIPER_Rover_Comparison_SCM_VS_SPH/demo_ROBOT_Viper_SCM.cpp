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

#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/assets/ChTexture.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/physics/ChParticlesClones.h"
#include "chrono/physics/ChLinkMotorRotationSpeed.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/physics/ChLinkDistance.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "chrono_irrlicht/ChVisualSystemIrrlicht.h"
#include "chrono/physics/ChLinkTSDA.h"

using namespace chrono;
using namespace chrono::irrlicht;
using namespace chrono::geometry;
using namespace chrono::viper;

using namespace irr;

bool output = true;
const std::string out_dir = GetChronoOutputPath() + "SCM_DEF_SOIL";

// SCM grid spacing
double mesh_resolution = 0.04;

// Enable/disable bulldozing effects
bool enable_bulldozing = false;

// Enable/disable moving patch feature
bool enable_moving_patch = true;

// If true, use provided callback to change soil properties based on location
bool var_params = false;

// Define Viper rover wheel type
ViperWheelType wheel_type = ViperWheelType::SimpleWheel;

double wheel_rad = 0.47;
double wheel_ang = 0.2 * CH_C_PI;
double ini_vel = 1 * wheel_rad * wheel_ang;

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
    float mu = 0.9f;   // coefficient of friction
    float cr = 0.4f;   // coefficient of restitution
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
    ChSystemNSC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Initialize output
    if (output) {
        if (!filesystem::create_directory(filesystem::path(out_dir))) {
            std::cout << "Error creating directory " << out_dir << std::endl;
            return 1;
        }
    }
    utils::CSV_writer csv(" ");
    utils::CSV_writer dbp(" ");

    csv << "time,rot_x,rot_y,rot_z" << std::endl;
    dbp << "time,vel,ang,slip,extension,dbp,torque" << std::endl;

    // Create the rover
    auto driver = chrono_types::make_shared<ViperSpeedDriver>(0.1, wheel_ang);
    // auto driver = chrono_types::make_shared<ViperDCMotorControl>();

    Viper viper(&sys, wheel_type);

    viper.SetDriver(driver);
    if (use_custom_mat)
        viper.SetWheelContactMaterial(CustomWheelMaterial(ChContactMethod::NSC));

    viper.Initialize(ChFrame<>(ChVector<>(-1.5, 0, wheel_rad + 0.01), QUNIT));

    // Get wheels and bodies to set up SCM patches
    auto Wheel_1 = viper.GetWheel(ViperWheelID::V_LF)->GetBody();
    auto Wheel_2 = viper.GetWheel(ViperWheelID::V_RF)->GetBody();
    auto Wheel_3 = viper.GetWheel(ViperWheelID::V_LB)->GetBody();
    auto Wheel_4 = viper.GetWheel(ViperWheelID::V_RB)->GetBody();
    auto Body_1 = viper.GetChassis()->GetBody();
    Body_1->SetCollide(false);
    // Body_1->SetPos_dt(ChVector<>(ini_vel, 0.0, 0.0));

    auto ground = chrono_types::make_shared<ChBodyEasyBox>(50, 50, 0.02, 1000, false, false, CustomWheelMaterial(ChContactMethod::NSC));
    ground->SetPos(ChVector<>(0, 0, 0));
    ground->SetBodyFixed(true);
    // ground->GetVisualShape(0)->SetTexture(GetChronoDataFile("textures/concrete.jpg"), 60, 45);
    sys.Add(ground);

    // Spring constraint
    std::shared_ptr<ChLinkTSDA> spring;
    ChVector<> pos1 = ChVector<>(-100, 0, wheel_rad + 0.01);
    ChVector<> pos2 = ChVector<>(-1.5, 0, wheel_rad + 0.01);
    spring = chrono_types::make_shared<ChLinkTSDA>();
    // spring->Initialize(ground, Body_1, false, pos1, pos2);
    // spring->SetSpringCoefficient(500.0);
    // spring->SetDampingCoefficient(50.0);
    // spring->SetRestLength(102);
    // sys.AddLink(spring);

    double f1 = spring->GetForce();
    std::cout << f1 << std::endl;
    // std::string dummy;
    // std::getline(std::cin, dummy);

    // rocks
    std::vector<std::shared_ptr<ChBodyAuxRef>> rocks;

    // create default SMC materials for the obstacles
    std::shared_ptr<ChMaterialSurface> rockSufaceMaterial = ChMaterialSurface::DefaultMaterial(sys.GetContactMethod());

    // create rocks
    auto rock_Body = chrono_types::make_shared<ChBodyAuxRef>();
    for (int i = 0; i < 1; i++) {
        // Create a rock
        auto rock_mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string rock_obj_path = GetChronoDataFile("robot/curiosity/rocks/rock3.obj");
        double scale_ratio = 0.2;
        rock_mmesh->LoadWavefrontMesh(rock_obj_path, false, true);
        rock_mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));  // scale to a different size
        rock_mmesh->RepairDuplicateVertexes(1e-9);                              // if meshes are not watertight

        // set the abs orientation, position and velocity
        // auto rock_Body = chrono_types::make_shared<ChBodyAuxRef>();
        // ChQuaternion<> rock_rot = ChQuaternion<>(1, 0, 0, 0);
        ChQuaternion<> rock_rot = Q_from_AngX(0.5 * CH_C_PI);
        ChVector<> rock_pos;
        rock_pos = ChVector<>(-0.3, 0.6, 0.2);

        // compute mass inertia from mesh
        double mmass;
        ChVector<> mcog;
        ChMatrix33<> minertia;
        double mdensity = 2000;  // paramsH->bodyDensity;
        rock_mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
        ChMatrix33<> principal_inertia_rot;
        ChVector<> principal_I;
        ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

        rock_Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        rock_Body->SetMass(mmass * mdensity);  // mmass * mdensity
        rock_Body->SetInertiaXX(mdensity * principal_I);

        rock_Body->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(rock_pos), ChQuaternion<>(rock_rot)));
        sys.Add(rock_Body);

        rock_Body->SetBodyFixed(true);
        rock_Body->GetCollisionModel()->ClearModel();
        rock_Body->GetCollisionModel()->AddTriangleMesh(rockSufaceMaterial, rock_mmesh, false, false, VNULL,
                                                        ChMatrix33<>(1), 0.005);
        rock_Body->GetCollisionModel()->BuildModel();
        rock_Body->SetCollide(false);

        auto rock_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        rock_mesh->SetMesh(rock_mmesh);
        rock_mesh->SetBackfaceCull(true);
        // rock_Body->AddVisualShape(rock_mesh);

        rocks.push_back(rock_Body);
    }

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
    double length = 50;
    double width = 4;
    terrain.Initialize(length, width, mesh_resolution);

    // Set the soil terramechanical parameters
    if (var_params) {
        // Here we use the soil callback defined at the beginning of the code
        auto my_params = chrono_types::make_shared<MySoilParams>();
        terrain.RegisterSoilParametersCallback(my_params);
    } else {
        // If var_params is set to be false, these parameters will be used
        terrain.SetSoilParameters(3.2 * 0.2e6,  // Bekker Kphi
                                  0,      // Bekker Kc
                                  0.72 * 1.1,    // Bekker n exponent
                                  0,      // Mohr cohesive limit (Pa)
                                  0.64 * 30,     // Mohr friction limit (degrees)
                                  3.1 * 0.01,   // Janosi shear coefficient (m)
                                  4e7,    // Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
                                  3e4     // Damping (Pa s/m), proportional to negative vertical speed (optional)
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
        for (int i = 0; i < rocks.size(); i++) {
            terrain.AddMovingPatch(rocks[i], ChVector<>(0, 0, 0), ChVector<>(0.5, 2 * wheel_range, 2 * wheel_range));
        }
    }

    // Set some visualization parameters: either with a texture, or with falsecolor plot, etc.
    terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_PRESSURE, 0, 20000);

    terrain.SetMeshWireframe(true);

    // Create the Irrlicht visualization (open the Irrlicht device,
    // bind a simple user interface, etc. etc.)
    // Create the Irrlicht visualization sys
    auto vis = chrono_types::make_shared<ChVisualSystemIrrlicht>();
    sys.SetVisualSystem(vis);
    vis->SetCameraVertical(CameraVerticalDir::Z);
    vis->SetWindowSize(2560, 1600);
    vis->SetWindowTitle("Viper Rover on SCM");
    vis->Initialize();
    vis->AddLogo();
    vis->AddSkyBox();
    vis->AddCamera(ChVector<>(2.0, 0.0, 1.4), ChVector<>(0, 0, wheel_range));
    vis->AddTypicalLights();
    vis->AddLightWithShadow(ChVector<>(-6.0, -1.5, 10.0), ChVector<>(-1, 0, 0), 100, 0.5, 25, 85, 512,
                            ChColor(0.8f, 0.8f, 1.0f));
    vis->EnableShadows();

    std::cout << "Rover mass is: " << viper.GetRoverMass() << std::endl;
    std::cout << "Rover pos is: " << viper.GetChassisPos().x() << viper.GetChassisPos().y() <<viper.GetChassisPos().z() << std::endl;

    bool add_spring = true;
    while (vis->Run() && sys.GetChTime() < 50) {
        vis->BeginScene();
        vis->GetActiveCamera()->setTarget(core::vector3dfCH(Body_1->GetPos()));
        vis->DrawAll();
        tools::drawColorbar(vis.get(), 0, 20000, "Pressure yield [Pa]", 1180);
        vis->EndScene();

        if (output) {
            // write drive torques of all four wheels into file
            csv << sys.GetChTime() << viper.GetWheelTracTorque(ViperWheelID::V_LF)
                << viper.GetWheelTracTorque(ViperWheelID::V_RF) << viper.GetWheelTracTorque(ViperWheelID::V_LB)
                << viper.GetWheelTracTorque(ViperWheelID::V_RB) << std::endl;
        }

        double wheel_vel = Wheel_1->GetPos_dt().x();
        double wheel_Wvel = Wheel_1->GetWvel_loc().z();
        double rover_slip = 1.0 - wheel_vel / (wheel_Wvel * wheel_rad);
        double spring_extension = spring->GetLength() - spring->GetRestLength();
        double spring_force = spring->GetForce();
        double wheel_torque = viper.GetWheelTracTorque(ViperWheelID::V_LF)
                            + viper.GetWheelTracTorque(ViperWheelID::V_RF)
                            + viper.GetWheelTracTorque(ViperWheelID::V_LB)
                            + viper.GetWheelTracTorque(ViperWheelID::V_RB);
        if (sys.GetChTime() > 0.0){
            std::cout << sys.GetChTime() << ", "
                      << wheel_vel << ", "
                      << wheel_Wvel << ", "
                      << rover_slip << ", "
                      << spring_extension << ", "
                      << spring_force << ", "
                      << wheel_torque << std::endl;
        }

        double buff_time = 0.0;
        if (sys.GetChTime() > buff_time && add_spring){
            pos2 = viper.GetChassisPos();
            spring->Initialize(ground, Body_1, false, pos1, pos2);
            spring->SetSpringCoefficient(1000.0);
            spring->SetDampingCoefficient(100.0);
            sys.AddLink(spring);
            add_spring = false;
        }
        if (sys.GetChTime() > buff_time){
            dbp << sys.GetChTime() - buff_time
                << wheel_vel
                << wheel_Wvel
                << rover_slip
                << spring_extension
                << spring_force 
                << wheel_torque << std::endl;
        }

        
        sys.DoStepDynamics(5e-4);
        viper.Update();
        ////terrain.PrintStepStatistics(std::cout);
    }

    if (output) {
        // csv.write_to_file(out_dir + "/output.csv");
        dbp.write_to_file(out_dir + "/dbp_scm_HWWMA_02.csv");
    }

    return 0;
}

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
// Authors: Bo-Hsun Chen, Harry Zhang
// =============================================================================
//
// Demo to show Viper Rover operated on digitized terrains from POLAR dataset based on SCM model
//
// =============================================================================
#include <iostream>
// #include <ctime>
#include <time.h> 
#include <cstring>
#include <sstream>

#include "chrono_models/robot/viper/Viper.h"

// #include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/assets/ChVisualMaterial.h"
#include "chrono/assets/ChVisualShape.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/physics/ChInertiaUtils.h"

#include "chrono_irrlicht/ChVisualSystemIrrlicht.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/SCMTerrain.h"

#include "chrono_sensor/sensors/ChCameraSensor.h"
#include "chrono_sensor/ChSensorManager.h"
#include "chrono_sensor/filters/ChFilterAccess.h"
#include "chrono_sensor/filters/ChFilterGrayscale.h"
#include "chrono_sensor/filters/ChFilterSave.h"
#include "chrono_sensor/filters/ChFilterVisualize.h"
#include "chrono_sensor/filters/ChFilterCameraNoise.h"
#include "chrono_sensor/filters/ChFilterCameraExposure.h"
#include "chrono_sensor/filters/ChFilterImageOps.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "chrono_vehicle/terrain/RigidTerrain.h"

using namespace chrono;
using namespace chrono::irrlicht;
// using namespace chrono::geometry;
using namespace chrono::viper;
using namespace chrono::sensor;
using namespace irr;

//// camera model parameter setting ////
CameraLensModelType lens_model = CameraLensModelType::PINHOLE; // Camera lens model, Either PINHOLE or SPHERICAL
float update_rate = 30; // [Hz], Update rate
unsigned int image_width = 1920; // [pixel], viewer width
unsigned int image_height = 1080; // [pixel], viewer height
unsigned int front_camera_width = 1936; // [pixel], front camera width
unsigned int front_camera_height = 1216; // [pixel], front camera height
float viewer_fov = (float)CH_PI / 2.; // viewer's horizontal field of view (hFOV)
float front_camera_fov = (float)((44.0 + 1.0/60) * CH_PI / 180.0); // [rad], front camera's horizontal field of view (hFOV)
float lag = 0.0f; // [sec], lag between sensing and when data becomes accessible
int alias_factor = 1;

//// dynamic solver parameter setting ////
float end_time = 40.f; // [sec], simulation end time
double mesh_resolution = 0.02; // [m], SCM grid spacing
int nthreads = 40; // number of SCM and collision threads

//// Viper dynamic model setting ////
ViperWheelType wheel_type = ViperWheelType::RealWheel; // define Viper rover wheel type (RealWheel, SimpleWheel, or CylWheel)
double wheel_range = 0.25; // [m], global parameter for moving patch size
// double body_range = 1.2; // [m]
double speed_driver_ramped_time = 0.1; // [sec], ramped time
// double dsr_driver_angular_velocity = 0.74; // [rad/sec], desired angular velocity, good for SCM model
double dsr_driver_angular_velocity = 0.; // [rad/sec], desired angular velocity

//// terrain setting ////
float terrain_scale_ratio = 1.0;

//// path setting ////
const std::string out_dir = GetChronoOutputPath() + "SCM_VIPER_POLAR/"; // output folder for saved images

//// switch setting ////
bool save = false; // Save camera and viewer images
bool vis = true; // visualize camera
bool exposure_correction_switch = true; // Exposure correction filter
bool enable_bulldozing = true; // enable/disable bulldozing effects
bool enable_moving_patch = true; // enable/disable moving patch feature
bool var_params = true; // if true, use provided callback to change soil properties based on location
bool use_custom_mat = false; // use custom material for the Viper wheel
bool debug_terrain = false; // whether just check the terrain and stop running VIPER

// Custom callback for setting location-dependent soil properties.
// Note that the location is given in the SCM reference frame.
class MySoilParams : public vehicle::SCMTerrain::SoilParametersCallback {
  public:
    virtual void Set(const ChVector3d& loc,
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
        elastic_K = 2e8; // original 2e8
        damping_R = 3e4;
    }
};

// Return customized wheel material parameters
std::shared_ptr<ChContactMaterial> CustomWheelMaterial(ChContactMethod contact_method) {
    float mu = 0.8f;   // coefficient of friction, original 0.4f
    float cr = 0.1f;   // coefficient of restitution
    float Y = 2e7f;    // Young's modulus
    float nu = 0.3f;   // Poisson ratio
    float kn = 2e5f;   // normal stiffness
    float gn = 40.0f;  // normal viscous damping
    float kt = 2e5f;   // tangential stiffness
    float gt = 20.0f;  // tangential viscous damping

    switch (contact_method) {
        case ChContactMethod::NSC: {
            auto matNSC = chrono_types::make_shared<ChContactMaterialNSC>();
            matNSC->SetFriction(mu);
            matNSC->SetRestitution(cr);
            return matNSC;
        }
        case ChContactMethod::SMC: {
            auto matSMC = chrono_types::make_shared<ChContactMaterialSMC>();
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
            return std::shared_ptr<ChContactMaterial>();
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "./demo_ROBOT_Viper_SCM_POLAR [Sun Position: 1, 2, 3, 4] [enable Hapke: 0, 1] [exposure time (sec)]\n";
        exit(1);
    }
    std::cout << "Copyright (c) 2024 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
    
    bool enable_hapke = (std::atoi(argv[2]) > 0) ? true : false; // enable Hapke or not?---0(disable hapke)--1(enable hapke)
    float expsr_time = std::atof(argv[3]); // [sec], exposure time
	std::ostringstream expsr_time_str;
	expsr_time_str << std::setfill('0') << std::setw(4) << static_cast<int>(expsr_time * 1000 + 0.5);

    std::string brdf_type = "";
    if (enable_hapke) {
        brdf_type = "hapke";
    }
    else {
        brdf_type = "default";
    }
    
    //// define different postion of the sun ////
    float sun_height = 100 * tanf(8 * CH_PI/180.0); // [m]

    auto pos1 = ChVector3d( 100.0, 0.0, sun_height); // [m], azi = 0 deg
    auto pos2 = ChVector3d( 100.0 * cosf(315 * CH_PI/180.0), 100.0 * sinf(315 * CH_PI/180.0), sun_height); // [m], azi = 315 deg
    auto pos3 = ChVector3d( 100.0 * cosf(225 * CH_PI/180.0), 100.0 * sinf(225 * CH_PI/180.0), sun_height); // [m], azi = 225 deg
    auto pos4 = ChVector3d(-100.0, 0.0, sun_height); // [m], azi = 180 deg
    auto sun_pose = ChVector3d{0.0, 0.0, 0.0};
    switch (std::atoi(argv[1])) {
        case 1:
            sun_pose = pos1;
            break;
        case 2:
            sun_pose = pos2;
            break;
        case 3:
            sun_pose = pos3;
            break;
        case 4:
            sun_pose = pos4;
            break;
        default:
            std::cout << "unknown Sun position ID ...";
            exit(1);
    }
    std::cout << "Sun posi: " << sun_pose << std::endl;
	std::cout << "BRDF: " << brdf_type << std::endl;
	std::cout << "exposure time [sec]: " << expsr_time << std::endl;

    //// initialize visual material ////
    auto polar_mat = chrono_types::make_shared<ChVisualMaterial>();
    polar_mat->SetAmbientColor({0.0, 0.0, 0.0}); //0.65f, 0.65f, 0.65f
    polar_mat->SetDiffuseColor({0.7, 0.7, 0.7});
    polar_mat->SetSpecularColor({1.0, 1.0, 1.0});
    polar_mat->SetUseSpecularWorkflow(true);
    polar_mat->SetRoughness(0.8f);
    polar_mat->SetAnisotropy(1.f);
	if (enable_hapke) {
		polar_mat->SetBSDF((unsigned int)BSDFType::HAPKE);
		polar_mat->SetHapkeParameters(0.32357f, 0.23955f, 0.30452f, 1.80238f, 0.07145f, 0.3f, 23.4f * (CH_PI / 180));
	}
    polar_mat->SetClassID(30000);
    polar_mat->SetInstanceID(20000);

    // ------------------- //
    // Configuration table //
    // ------------------- //
    
    // ---- data path ---- //
    const std::string out_folder = "SENSOR_OUTPUT/LunarProject/"; // output folder for saved images
    // const std::string terrain_mesh_path = GetChronoDataFile("robot/curiosity/rocks/terrain_and_rocks.obj");
    // const std::string terrain_mesh_path = GetChronoDataFile("robot/curiosity/rocks/only_terrain(hi_res).obj");

    // ---- create maps for positions of Sun and cameras ---- //
    std::unordered_map<char, std::unordered_map<char, ChVector3f>> camera_posi_map{ // [m]
        {'A', {{'L', {3.541f, -0.303f, 1.346f}}, {'R', {3.532f, -0.002f, 1.344f}}}},
        {'B', {{'L', {5.500f, -0.303f, 1.345f}}, {'R', {5.500f, -0.002f, 1.344f}}}},
        {'C', {{'L', {0.316f, -3.540f, 1.346f}}, {'R', {0.611f, -3.479f, 1.344f}}}}
    };

    // Create a Chrono::Engine physical system
    ChSystemNSC sys;
    sys.SetGravitationalAcceleration(ChVector3d(0, 0, -9.81)); // [m/sec^2]
    sys.SetCollisionSystemType(ChCollisionSystem::Type::BULLET);
    sys.SetNumThreads(nthreads, nthreads, 1);

    // ------------------ //
    // Create Viper rover //
    // ------------------ //
    
    auto driver = chrono_types::make_shared<ViperDCMotorControl>();
    //auto driver = chrono_types::make_shared<ViperSpeedDriver>(0.1,0.37f);
    // auto driver = chrono_types::make_shared<ViperSpeedDriver>(speed_driver_ramped_time, dsr_driver_angular_velocity);
    Viper viper(&sys, wheel_type);
    viper.SetDriver(driver);
    // viper.SetChassisCollide(true);
    viper.SetWheelContactMaterial(CustomWheelMaterial(ChContactMethod::NSC));
    
    // viper.Initialize(ChFrame<>(ChVector3d(6.0, 0., 0.2), QuatFromAngleAxis(CH_PI, {0., 0, 1.}))); // [m, quaternion], good init. posi. for SCM model scaled 1.5
    viper.Initialize(ChFrame<>(ChVector3d(5.0, 0., 0.2), QuatFromAngleAxis(CH_PI, {0., 0, 1.}))); // [m, quaternion], good init. posi. for SCM model scaled 1.0
    // viper.Initialize(ChFrame<>(ChVector3d(5, 5, 0.2), QUNIT)); // Used for visualizing how big the rock is

    // Get wheels and bodies to set up SCM patches
    auto wheel_LF = viper.GetWheel(ViperWheelID::V_LF)->GetBody();
    auto wheel_RF = viper.GetWheel(ViperWheelID::V_RF)->GetBody();
    auto wheel_LB = viper.GetWheel(ViperWheelID::V_LB)->GetBody();
    auto wheel_RB = viper.GetWheel(ViperWheelID::V_RB)->GetBody();
    auto Body_1 = viper.GetChassis()->GetBody();
    

    // ----------------------------------------- //
    // nonresponsive rocks and decorated grounds //
    // ----------------------------------------- //
    
    //// set up decorated grounds ////
    printf("Loading decorated ground into system ...\n");
    auto decorated_ground_mat = chrono_types::make_shared<ChVisualMaterial>();
    auto decorated_ground_mesh_side = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
    auto decorated_ground_mesh_front = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
    // auto decorated_ground_mesh_back = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
    auto decorated_ground_body_left = chrono_types::make_shared<ChBodyAuxRef>();
    auto decorated_ground_body_right = chrono_types::make_shared<ChBodyAuxRef>();
    // auto decorated_ground_body_front = chrono_types::make_shared<ChBodyAuxRef>();
    auto decorated_ground_body_front = chrono_types::make_shared<ChBodyEasyBox>(6.5, 3.8, 0.2, 1000, true, false);
    // auto decorated_ground_body_back = chrono_types::make_shared<ChBodyAuxRef>();
    auto decorated_ground_body_back = chrono_types::make_shared<ChBodyEasyBox>(7.0, 3.8, 0.2, 1000, true, false);
    


    // double terrain_scale_ratio = 1.0;
    // printf("Loading Terrain 6 into system ...\n");
    
    // set up decorated-ground material
    decorated_ground_mat->SetAmbientColor({0.0, 0.0, 0.0});
    decorated_ground_mat->SetDiffuseColor({0.7, 0.7, 0.7});
    decorated_ground_mat->SetSpecularColor({1.0, 1.0, 1.0});
    decorated_ground_mat->SetUseSpecularWorkflow(true);
    decorated_ground_mat->SetRoughness(0.8f);
    decorated_ground_mat->SetAnisotropy(1.f);
	if (enable_hapke) {
		decorated_ground_mat->SetBSDF((unsigned int)BSDFType::HAPKE);
		decorated_ground_mat->SetHapkeParameters(0.32357f, 0.23955f, 0.30452f, 1.80238f, 0.07145f, 0.3f, 23.4f * (CH_PI / 180));
	}

    // set up side ground meshes
    // load mesh from obj file
    std::string decorated_ground_mesh_path = GetChronoDataFile("robot/curiosity/rocks/Terrain06/terrain06_ground_decimate_005.obj");;
    auto decorated_ground_mesh_side_loader = ChTriangleMeshConnected::CreateFromWavefrontFile(
        decorated_ground_mesh_path, false, false
    );
    decorated_ground_mesh_side_loader->Transform(
        ChVector3d(0, 0, 0),
        ChMatrix33<>(
            {7.0, 1.0, 1.0}, // scale to a different size
            {0., 0., 0.}
        ) 
    );
    decorated_ground_mesh_side_loader->RepairDuplicateVertexes(1e-9); // if meshes are not watertight
    decorated_ground_mesh_side->SetMesh(decorated_ground_mesh_side_loader);
    decorated_ground_mesh_side->SetBackfaceCull(true);
    
    // set up side decorated-grounds
    decorated_ground_body_left->AddVisualShape(decorated_ground_mesh_side);
    decorated_ground_body_left->GetVisualShape(0)->SetMaterial(0, decorated_ground_mat);
    decorated_ground_body_left->SetFixed(true);
    decorated_ground_body_left->SetPos({0., -3.7, -0.32}); // [m]
    decorated_ground_body_left->SetRot(QuatFromAngleAxis(-0.5 * (CH_PI / 180), {0, 1, 0}));
    sys.Add(decorated_ground_body_left);
    
    decorated_ground_body_right->AddVisualShape(decorated_ground_mesh_side);
    decorated_ground_body_right->GetVisualShape(0)->SetMaterial(0, decorated_ground_mat);
    decorated_ground_body_right->SetFixed(true);
    decorated_ground_body_right->SetPos({0.,  3.7, -0.39}); // [m]
    decorated_ground_body_right->SetRot(QuatFromAngleAxis(-0.5 * (CH_PI / 180), {0, 1, 0}));
    sys.Add(decorated_ground_body_right);
    
    // set up front ground mesh
    /*
    // load front mesh from obj file
    auto decorated_ground_mesh_front_loader = ChTriangleMeshConnected::CreateFromWavefrontFile(
        decorated_ground_mesh_path, false, false
    );
    decorated_ground_mesh_front_loader->Transform(
        ChVector3d(0, 0, 0),
        ChMatrix33<>(
            {1.75, 1.00, 1.0}, // scale to a different size
            // {std::atof(argv[7]), std::atof(argv[8]), 1.0}, // calibrating
            {0., 0., 0.}
        ) 
    );
    decorated_ground_mesh_front_loader->RepairDuplicateVertexes(1e-9); // if meshes are not watertight
    decorated_ground_mesh_front->SetMesh(decorated_ground_mesh_front_loader);
    decorated_ground_mesh_front->SetBackfaceCull(true);

    // set up front decorated-ground
    decorated_ground_body_front->AddVisualShape(decorated_ground_mesh_front);
    decorated_ground_body_front->GetVisualShape(0)->SetMaterial(0, decorated_ground_mat);
    */
    decorated_ground_body_front->SetFixed(true);
    decorated_ground_body_front->SetPos({-9.48, 0.10, -0.505}); // [m]
    // decorated_ground_body_front->SetPos({std::atof(argv[4]), std::atof(argv[5]), std::atof(argv[6])}); // calibrating
    decorated_ground_body_front->SetRot(QuatFromAngleAxis(-1.0 * (CH_PI/180), {0, 1, 0}));
    {
        auto shape = decorated_ground_body_front->GetVisualModel()->GetShapeInstances()[0].first;
        if (shape->GetNumMaterials() == 0) {
            shape->AddMaterial(decorated_ground_mat);
        } else {
            shape->GetMaterials()[0] = decorated_ground_mat;
        }
    }
    
    sys.Add(decorated_ground_body_front);

    // set up back ground mesh
    /*
    load back mesh from obj file
    auto decorated_ground_mesh_back_loader = ChTriangleMeshConnected::CreateFromWavefrontFile(
        decorated_ground_mesh_path, false, false
    );
    decorated_ground_mesh_back_loader->Transform(
        ChVector3d(0, 0, 0),
        ChMatrix33<>(
            {1.0, 1.0, 1.0}, // scale to a different size
            {0., 0., 0.}
        ) 
    );
    decorated_ground_mesh_back_loader->RepairDuplicateVertexes(1e-9); // if meshes are not watertight
    decorated_ground_mesh_back->SetMesh(decorated_ground_mesh_back_loader);
    decorated_ground_mesh_back->SetBackfaceCull(true);

    // set up back decorated-ground
    decorated_ground_body_back->AddVisualShape(decorated_ground_mesh_back);
    decorated_ground_body_back->GetVisualShape(0)->SetMaterial(0, decorated_ground_mat);
    decorated_ground_body_back->SetFixed(true);
    decorated_ground_body_back->SetPos({6.0, 0., -0.32}); // [m]
    decorated_ground_body_back->SetRot(QuatFromAngleAxis(-0.5 * (CH_PI/180), {0, 1, 0}));
    */
    decorated_ground_body_back->SetPos({9.5, 0, -0.38});
    decorated_ground_body_back->SetFixed(true);
    {
        auto shape = decorated_ground_body_back->GetVisualModel()->GetShapeInstances()[0].first;
        if (shape->GetNumMaterials() == 0) {
            shape->AddMaterial(decorated_ground_mat);
        } else {
            shape->GetMaterials()[0] = decorated_ground_mat;
        }
    }
    sys.Add(decorated_ground_body_back);

    printf("Finished Loading decorated ground into system\n");

    //// set up rocks ////
    
    std::vector<std::string> terrains = {"04", "01", "11"};
    // std::vector<std::string> terrains = {"04", "01"};
    // std::vector<std::string> terrains = {"11"};
     // Looping through all available terrains
    for (std::string terrainID : terrains) {
        std::string rock_mesh_dir = GetChronoDataFile("robot/curiosity/rocks/Terrain" + terrainID + "/");
        size_t rock_num = 0;
        float terrain_x_offset = 0., terrain_y_offset = 0., terrain_z_offset = 0.;
        float terrain_x_rot = 0.;
        
        if (terrainID == "04") { // Terrain 4
            rock_num = 27;
            // rock_num = 14; // debug
            // terrain_x_offset = 2.6 * 1.5; // for scale 1.5
            // terrain_y_offset = -0.1; // for scale 1.5
            // terrain_z_offset = -0.2 - 0.23 - 0.2 - 0.175; // for scale 1.5
            terrain_x_offset = 2.9 - 0.34 - 0.01 - 0.10; // [m], for v10
            terrain_y_offset = 0.04 - 0.04; // [m], for v10
            // terrain_z_offset = -0.525 - 0.017 + 0.08 - 0.080; // [m], for v10
            terrain_z_offset = -0.525 - 0.017 + 0.08 - 0.080 + 0.015; // [m], for v12
            terrain_x_rot = -0.4 * CH_PI/180.0; // [rad]
        }
        else if (terrainID == "01") { // Terrain 01
            rock_num = 15;
            terrain_x_offset = 2.9 - 0.34 - 3.56 + 0.08 - 0.02 - 0.13; // [m], for v10
            terrain_y_offset = 0.04 - 0.04; // [m], for v10
            // terrain_z_offset = -0.525 + 0.23 - 0.009 + 0.015 + 0.07 - 0.080; // [m], for v10
            terrain_z_offset = -0.525 + 0.23 - 0.009 + 0.015 + 0.07 - 0.080 + 0.025; // [m], for v12
            terrain_x_rot = -0.1 * CH_PI/180.0; // [rad]
        }
        else if (terrainID == "11") { // Terrain 11
            rock_num = 9;
            terrain_x_offset = 2.9 - 0.34 - 3.56 - 3.35 + 0.06 + 0.07; // [m], for v10
            terrain_y_offset = 0.04 + 0.26 - 0.08; // [m], for v10
            terrain_z_offset = -0.525 + 0.23 + 0.04 + 0.02 + 0.09 + 0.02 - 0.095; // [m], for v12
            terrain_x_rot = 2.5 * CH_PI/180.0; // [rad]
        }
        else {
            std::cout << "unknown terrain ID\n";
            exit(1);
        }
        
        std::vector<std::shared_ptr<ChVisualShapeTriangleMesh>> rock_meshes(rock_num);
        std::vector<std::shared_ptr<ChBodyAuxRef>> rock_bodies(rock_num);
        std::string rock_mesh_path = "";
        std::shared_ptr<ChContactMaterial> rock_contact_mat = ChContactMaterial::DefaultMaterial(sys.GetContactMethod());
        double mmass;
        ChVector3d mcog;
        ChMatrix33<> minertia;
        double rock_density = 8000;  // [kg/m^3];
        ChMatrix33<> principal_inertia_rot;
        ChVector3d principal_I;

        // initilize visual material
        auto vis_mat2 = chrono_types::make_shared<ChVisualMaterial>();
        vis_mat2->SetAmbientColor({1,1,1}); //0.65f,0.65f,0.65f
        vis_mat2->SetDiffuseColor({1,1,1});
        vis_mat2->SetSpecularColor({1,1,1});
        vis_mat2->SetUseSpecularWorkflow(true);
        vis_mat2->SetRoughness(1.0f);
		if (enable_hapke) {
			vis_mat2->SetBSDF((unsigned int)BSDFType::HAPKE);
    		vis_mat2->SetHapkeParameters(0.32357f, 0.23955f, 0.30452f, 1.80238f, 0.07145f, 0.3f, 23.4f * (CH_PI / 180));
		}

        // ------------------------ //
        // Visible/collidable rocks //
        // ------------------------ //
        for (size_t rock_idx = 0; rock_idx < rock_num; ++rock_idx) {
            // set up rock mesh
            rock_meshes[rock_idx] = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
            
            // load mesh from obj file
            if (terrainID == "test" && rock_num > 0) {
                rock_mesh_path = GetChronoDataFile("robot/curiosity/rocks/Terrain" + std::string(argv[3]) + "/terrain" + std::string(argv[3]) + "_rock" + std::to_string(rock_idx+1) + ".obj"); // test
            }
            else {
                rock_mesh_path = rock_mesh_dir + "terrain" + terrainID + "_rock" + std::to_string(rock_idx+1) + ".obj";
                // rock_mesh_path = rock_mesh_dir + "terrain" + terrainID + "_rock" + std::to_string(rock_idx+1) + "_decimate-005.obj"; // debug
            }

            // printf("loading Rock %zu mesh from %s ...\n", rock_idx + 1, rock_mesh_path.c_str());
            auto rock_mesh_loader = ChTriangleMeshConnected::CreateFromWavefrontFile(rock_mesh_path, false, true);
            rock_mesh_loader->Transform(
                ChVector3d(terrain_x_offset, terrain_y_offset, terrain_z_offset),
                ChMatrix33<>(terrain_scale_ratio) * ChMatrix33<>(QuatFromAngleAxis(0., {1, 0, 0})) // scale to a different size
            );
            rock_mesh_loader->RepairDuplicateVertexes(1e-9); // if meshes are not watertight
            rock_meshes[rock_idx]->SetMesh(rock_mesh_loader);
            rock_meshes[rock_idx]->SetBackfaceCull(true);
            
            // set up rocks
            rock_bodies[rock_idx] = chrono_types::make_shared<ChBodyAuxRef>();

            // compute mass inertia from mesh
            // rock_mesh_loader->ComputeMassProperties(true, mmass, mcog, minertia);
            // ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

            // set rock appearance
            rock_bodies[rock_idx]->AddVisualShape(rock_meshes[rock_idx]);
            rock_bodies[rock_idx]->GetVisualShape(0)->SetMaterial(0, polar_mat);

            // set the abs orientation, position and velocity
            // rock_bodies[rock_idx]->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));
            // rock_bodies[rock_idx]->SetMass(mmass * rock_density);  // mmass * mdensity
            // rock_bodies[rock_idx]->SetInertiaXX(rock_density * principal_I);
            //rock_bodies[rock_idx]->SetFrame_REF_to_abs(ChFrame<>(ChVector3d(rock_pos), ChQuaternion<>(rock_rot))); // ???

            sys.Add(rock_bodies[rock_idx]);
            
            // Rock Collision Property for a few rocks
            // || (terrainID == "01" && rock_idx == 6)
            // if ((terrainID == "04" && rock_idx ==  6) ||
            //     (terrainID == "01" && rock_idx ==  6) ||
            //     (terrainID == "04" && rock_idx == 11) ||
            //     (terrainID == "01" && rock_idx ==  7)) {
			std::string rock_mesh_path_decimate = rock_mesh_dir + "terrain" + terrainID + "_rock" + std::to_string(rock_idx+1) + "_decimate-005.obj";
			auto rock_mesh_loader_decimate = ChTriangleMeshConnected::CreateFromWavefrontFile(rock_mesh_path_decimate, false, true);
			rock_mesh_loader_decimate->Transform(
				ChVector3d(terrain_x_offset, terrain_y_offset, terrain_z_offset),
				ChMatrix33<>(terrain_scale_ratio) * ChMatrix33<>(QuatFromAngleAxis(0., {1, 0, 0})) // scale to a different size
			);
			rock_mesh_loader_decimate->RepairDuplicateVertexes(1e-9); // if meshes are not watertight

			// rock_bodies[rock_idx]->GetCollisionModel()->ClearModel();
			
			auto rock_ct_shape = chrono_types::make_shared<ChCollisionShapeTriangleMesh>(
				rock_contact_mat, rock_mesh_loader_decimate, false, false, 0.005
			);
			rock_bodies[rock_idx]->AddCollisionShape(rock_ct_shape);
			rock_bodies[rock_idx]->EnableCollision(true);
            // }

            rock_bodies[rock_idx]->SetFixed(true);
            rock_bodies[rock_idx]->SetPos({0., 0., 0.}); // [m]
            rock_bodies[rock_idx]->SetRot(ChMatrix33<>(terrain_x_rot, {1., 0., 0.}));
            // rock_bodies[rock_idx]->SetOffsetPose(chrono::ChFrame<double>(
            //     {0., 0., 0.},
            //     QuatFromAngleAxis(terrain_x_rot, {1., 0., 0.})
            // ));
            

            //ground.AddMovingPatch(rocks[i], ChVector3d(0, 0, 0), ChVector3d(0.5, 0.5, 0.5));

            printf("Rock %zu of Terrain %s added to system\n", rock_idx + 1, terrainID.c_str());
        }

    }
    

    // ---------- //
    // The ground //
    // ---------- //
    // std::string ground_mesh_path = GetChronoDataFile("robot/curiosity/rocks/big_ground_scale1_v12.obj"); // for MAES version
    std::string ground_mesh_path = GetChronoDataFile("robot/curiosity/rocks/big_ground_scale1_v15.obj");
    // std::string ground_mesh_path = "/home/sbel/blender/2024ICRA_PolarDigitize_material/big_ground_scale1.obj";
    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0025);
    // collision::ChCollisionModel::SetDefaultSuggestedMargin(0.0025);

    //// Create the rigid ground ////
    /*
    auto ground_mesh = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
    auto ground_mesh_loader = ChTriangleMeshConnected::CreateFromWavefrontFile(ground_mesh_path, false, false);
    ground_mesh_loader->Transform(ChVector3d(0, 0, 0), ChMatrix33<>(1.0));  // scale to a different size
    ground_mesh_loader->RepairDuplicateVertexes(1e-9); // if meshes are not watertight
    
    // set terrain mesh
    ground_mesh->SetMesh(ground_mesh_loader);
    ground_mesh->SetBackfaceCull(true);
    
    auto ground_contact_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    chrono::vehicle::RigidTerrain ground(&sys);
    auto mesh = ground.AddPatch(ground_contact_mat, ChCoordsys<>(ChVector3d(0, 0, 0), QUNIT), ground_mesh_path);
    {
       // printf("OK!\n");
        mesh->GetGroundBody()->AddVisualShape(ground_mesh);
        auto shape = mesh->GetGroundBody()->GetVisualModel()->GetShapes()[0].first;
        if(shape->GetNumMaterials() == 0){
            shape->AddMaterial(polar_mat);
        }
        else{
            shape->GetMaterials()[0] = polar_mat;
        }
    }
    ground.Initialize();
    */

    //// Create the deformable ground ////
    vehicle::SCMTerrain ground(&sys);
    
    // load mesh from obj file
    // std::string ground_mesh_path = terrain_mesh_dir + "terrain" + terrainID + "_ground" + std::to_string(deform_ground_indices[ground_idx]+1) + ".obj";
    // std::string ground_mesh_path = terrain_mesh_dir + "big_ground_scale2.obj";
    // std::string ground_mesh_path = GetChronoDataFile("robot/curiosity/rocks/terrain11_ground_all_scale2.obj");
    
    
    ground.Initialize(ground_mesh_path, mesh_resolution);
    ground.SetMeshWireframe(false);
    auto mesh = ground.GetMesh();
    {
        if(mesh->GetNumMaterials() == 0){
            mesh->AddMaterial(polar_mat);
        }
        else{
            mesh->GetMaterials()[0] = polar_mat;
        }
    }

    // Set the soil terramechanical parameters
    if (var_params) { // Here we use the soil callback defined at the beginning of the code
        auto my_params = chrono_types::make_shared<MySoilParams>();
        ground.RegisterSoilParametersCallback(my_params);
    } 
    else { // If var_params is set to be false, these parameters will be used
        ground.SetSoilParameters(
            0.2e6,  // Bekker Kphi
            0,      // Bekker Kc
            1.1,    // Bekker n exponent
            0,      // Mohr cohesive limit (Pa)
            30,     // Mohr friction limit (degrees)
            0.01,   // Janosi shear coefficient (m)
            4e7,    // Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
            3e2     // Damping (Pa s/m), proportional to negative vertical speed (optional)
        );
    }

    // Set up bulldozing effect factors
    if (enable_bulldozing) {
        ground.EnableBulldozing(true);  // inflate soil at the border of the rut
        ground.SetBulldozingParameters(
            55,     // angle of friction for erosion of displaced material at the border of the rut
            1,      // displaced material vs downward pressed material.
            5,      // number of erosion refinements per timestep
            6       // number of concentric vertex selections subject to erosion
        );
    }

    // We need to add a moving patch under every wheel on the ground
    // Or we can define a large moving patch at the pos of the rover body
    if (enable_moving_patch) {
        // for VIPER moving on the ground
        
        ground.AddMovingPatch(wheel_LF, ChVector3d(0, 0, 0), ChVector3d(0.5, 2 * wheel_range, 2 * wheel_range));
        ground.AddMovingPatch(wheel_RF, ChVector3d(0, 0, 0), ChVector3d(0.5, 2 * wheel_range, 2 * wheel_range));
        ground.AddMovingPatch(wheel_LB, ChVector3d(0, 0, 0), ChVector3d(0.5, 2 * wheel_range, 2 * wheel_range));
        ground.AddMovingPatch(wheel_RB, ChVector3d(0, 0, 0), ChVector3d(0.5, 2 * wheel_range, 2 * wheel_range));
        

        // for obstacle rocks standing on the ground
        // for (int i = 0; i < 1; i++) {
        //     ground.AddMovingPatch(rocks[i], ChVector3d(0, 0, 0), ChVector3d(0.5, 0.5, 0.5));
        // }
       // ground.AddMovingPatch(rock_bodies[0], ChVector3d(0, 0, 0), ChVector3d(0.5, 0.5, 0.5));
    }
    

    // Set some visualization parameters: either with a texture, or with falsecolor plot, etc.
    //ground.SetPlotType(vehicle::SCMTerrain::PLOT_PRESSURE, 0, 20000);

    // ground.SetMeshWireframe(true);
    // deform_grounds.push_back(ground);

    // create a fixed origin cube to attach camera
    auto origin_cube = chrono_types::make_shared<ChBodyEasyBox>(1, 1, 1, 1000, false, false);
    origin_cube->SetPos({0, 0, 0});
    origin_cube->SetFixed(true);
    sys.Add(origin_cube);

    // ------------------------ //
    // background manager setup //
    // ------------------------ //
    auto manager = chrono_types::make_shared<ChSensorManager>(&sys);
    manager->scene->AddPointLight(sun_pose, {1.0, 1.0, 1.0}, 2000.0f);
    Background bgd;
    bgd.mode = BackgroundMode::ENVIRONMENT_MAP;
    bgd.env_tex = GetChronoDataFile("sensor/textures/starmap_2020_4k_darkened.hdr");
    // bgd.mode = BackgroundMode::SOLID_COLOR;
    // bgd.color_zenith = {0, 0, 0};
    manager->scene->SetBackground(bgd);

    // ------------------------- //
    // cameras and viewers setup //
    // ------------------------- //
    std::vector<float> expsr_a_arr(3, 0.); // exposure correction coefficients of a-term
    std::vector<float> expsr_b_arr(3, 0.); // exposure correction coefficients of b-term
    expsr_a_arr = {0., 0., 1.0}; // simple Default model
    if (enable_hapke) {
        expsr_b_arr = {0.698932456, -0.862405122, -7.530271991}; // simple Hapke model, avg of Terrains 01, 11, and 04
    }
    else {
        expsr_b_arr = {0.698938172, -0.918033419, -7.751329574}; // simple Default model, avg of Terrains 01, 11, and 04
    }
    
    chrono::ChFrame<double> wheel_poses[4] = {
        chrono::ChFrame<double>({-1.0, -1.5, 0.5}, QuatFromAngleAxis(.2, {-2, 3,  9.75})), 
        chrono::ChFrame<double>({ 1.0, -1.5, 0.5}, QuatFromAngleAxis(.2, {-2, 3,  9.75})),
        chrono::ChFrame<double>({ 1.0,  1.5, 0.5}, QuatFromAngleAxis(.2, { 2, 3, -9.75})),
        chrono::ChFrame<double>({-1.0,  1.5, 0.5}, QuatFromAngleAxis(.2, { 2, 3, -9.75}))
    };

	std::vector<std::string> wheel_cam_names = {"RightBack", "RightFront", "LeftFront", "LeftBack"};

    // auto rf_wheel = chrono::ChFrame<double>({1.0, 1.5, 0.5}, QuatFromAngleAxis(.2, {2, 3, -9.75}));
    for (int i = 0; i < 4; i++) {
        auto wheel_cam = chrono_types::make_shared<ChCameraSensor>(
			viper.GetChassis()->GetBody(),	// body camera is attached to
			update_rate,					// update rate in Hz
			wheel_poses[i],					// offset pose
			image_width,					// image width
			image_height,					// image height
			viewer_fov,						// camera's horizontal field of view
			alias_factor,					// supersample factor for antialiasing
			lens_model,						// lens model
			false							// use global illumination or not
		);
        wheel_cam->SetName("Wheel Camera " + wheel_cam_names[i]);
        wheel_cam->SetLag(lag);
        wheel_cam->SetCollectionWindow(0.f); // would cause dynamic blur effect
        if (exposure_correction_switch == true) {
            wheel_cam->PushFilter(chrono_types::make_shared<ChFilterCameraExposureCorrect>(
				expsr_a_arr[0], expsr_a_arr[1], expsr_a_arr[2],
				expsr_b_arr[0], expsr_b_arr[1], expsr_b_arr[2], expsr_time + 0.100f
			));
            printf("exposure time: %f sec\n", wheel_cam->GetCollectionWindow());
        }
        if (vis)
            wheel_cam->PushFilter(chrono_types::make_shared<ChFilterVisualize>(
				image_width, image_height, "Wheel Camera " + wheel_cam_names[i]
			));
        if (save) {
            wheel_cam->PushFilter(chrono_types::make_shared<ChFilterSave>(
				out_dir + "WheelCam_" + wheel_cam_names[i] + "_" + argv[1] + "_" + brdf_type + "_" + expsr_time_str.str() + "/"
			));
		}
        manager->AddSensor(wheel_cam);
    }
    
    // Set up left bird-eye-view camera
    chrono::ChFrame<double> cam_birdview_left_pose({0, -5.5, 2.0}, QuatFromAngleAxis(CH_PI/2, {0, 0, 1}));
    // chrono::ChFrame<double> cam_birdview_left_pose({-5.0, -7.0, 2.0}, QuatFromAngleAxis(CH_PI/2, {0, 0, 1})); // calibrating
    cam_birdview_left_pose.SetRot(cam_birdview_left_pose.GetRot() * QuatFromAngleAxis(20. * CH_PI/180, {0, 1, 0}));
    auto cam_birdview_left = chrono_types::make_shared<ChCameraSensor>(
        // viper.GetChassis()->GetBody(),  // body camera is attached to
        origin_cube,            // body that camera is attached to
        update_rate,                    // update rate in Hz
        cam_birdview_left_pose,         // offset pose
        image_width,                    // image width
        image_height,                   // image height
        viewer_fov,                     // camera's horizontal field of view
        alias_factor,                   // supersample factor for antialiasing
        lens_model,                     // lens model
        false							// use global illumination or not
    );
    cam_birdview_left->SetName("Left Bird Viewer");
    cam_birdview_left->SetLag(lag);
    cam_birdview_left->SetCollectionWindow(0.0f); // would cause dynamic blur effect
    if (exposure_correction_switch == true) {
        std::cout << expsr_a_arr[0] << std::endl;
        cam_birdview_left->PushFilter(chrono_types::make_shared<ChFilterCameraExposureCorrect>(
            expsr_a_arr[0], expsr_a_arr[1], expsr_a_arr[2],
            expsr_b_arr[0], expsr_b_arr[1], expsr_b_arr[2], expsr_time)
        );
    }
    if (vis) {
        cam_birdview_left->PushFilter(chrono_types::make_shared<ChFilterVisualize>(
            image_width, image_height, "Left bird view"
        ));
    }
    if (save) {
        cam_birdview_left->PushFilter(chrono_types::make_shared<ChFilterSave>(
            out_dir + "CamBirdViewLeft_" + argv[1] + "_" + brdf_type + "_" + expsr_time_str.str() + "/"
        ));
    }
    manager->AddSensor(cam_birdview_left);

    // Set up right bird-eye-view camera
    chrono::ChFrame<double> cam_birdview_right_pose({0, 5.5, 2.0}, QuatFromAngleAxis(-CH_PI/2, {0, 0, 1}));
    // chrono::ChFrame<double> cam_birdview_right_pose({-5.0, 8.0, 2.0}, QuatFromAngleAxis(-CH_PI/2, {0, 0, 1})); // calibrating
    cam_birdview_right_pose.SetRot(cam_birdview_right_pose.GetRot() * QuatFromAngleAxis(20. * CH_PI/180, {0, 1, 0}));
    auto cam_birdview_right = chrono_types::make_shared<ChCameraSensor>(
        // viper.GetChassis()->GetBody(),  // body camera is attached to
        origin_cube,                    // body that camera is attached to
        update_rate,                    // update rate in Hz
        cam_birdview_right_pose,         // offset pose
        image_width,                    // image width
        image_height,                   // image height
        viewer_fov,                     // camera's horizontal field of view
        alias_factor,                   // supersample factor for antialiasing
        lens_model,                     // lens model
        false                           // use global illumination or not
    );
    cam_birdview_right->SetName("Right Bird Viewer");
    cam_birdview_right->SetLag(lag);
    cam_birdview_right->SetCollectionWindow(0.0f); // would cause dynamic blur effect
    if (exposure_correction_switch == true) {
        std::cout << expsr_a_arr[0] << std::endl;
        cam_birdview_right->PushFilter(chrono_types::make_shared<ChFilterCameraExposureCorrect>(
            expsr_a_arr[0], expsr_a_arr[1], expsr_a_arr[2],
            expsr_b_arr[0], expsr_b_arr[1], expsr_b_arr[2], expsr_time)
        );
    }
    if (vis) {
        cam_birdview_right->PushFilter(chrono_types::make_shared<ChFilterVisualize>(
            image_width, image_height, "Right bird view"
        ));
    }
    if (save) {
        cam_birdview_right->PushFilter(chrono_types::make_shared<ChFilterSave>(
            out_dir + "CamBirdViewRight_" + argv[1] + "_" + brdf_type + "_" + expsr_time_str.str() + "/"
        ));
    }
    manager->AddSensor(cam_birdview_right);

    // Add front-end camera mounted on the Viper
    char front_cam_posi_idx = 'A';
    char front_cam_LR_idx = 'R';
    ChVector3f front_cam_posi = camera_posi_map[front_cam_posi_idx][front_cam_LR_idx];
    front_cam_posi.Mul(front_cam_posi, terrain_scale_ratio);
    front_cam_posi.x() += -3.3;
    front_cam_posi.z() += 0.30;
    front_cam_posi.y() += 0.50;
    //front_cam_posi.x() -= 4 ;
    chrono::ChFrame<double> front_cam_pose(
        front_cam_posi,
        QuatFromAngleAxis(30 * (CH_PI / 180), {0, 1, 0})
    );
    // chrono::ChFrame<double> offset_pose6( // debug
    //     ChVector3d({1.76913 + front_cam_posi.x(), -0.503446 + front_cam_posi.y(), 0.5 + front_cam_posi.z()}),
    //     QuatFromAngleAxis(CH_PI, {0, 0, 1}) * QuatFromAngleAxis(20 * (CH_PI / 180), {0, 1, 0}) 
    // );
    auto front_end_cam = chrono_types::make_shared<ChCameraSensor>(
        viper.GetChassis()->GetBody(), // body that camera is attached to
        // origin_cube,            // body that camera is attached to, debug
        update_rate,            // update rate in Hz
        front_cam_pose,         // offset pose
        // offset_pose6,           // offset pose, debug
        front_camera_width,     // image width
        front_camera_height,    // image height
        front_camera_fov,       // camera's horizontal field of view
        alias_factor,           // supersample factor for antialiasing
        lens_model,             // lens model
        false                  // use global illumination or not
    );
    front_end_cam->SetName("Front-end Camera");
    
    front_end_cam->SetLag(lag);
    // cam->SetUpdateRate(1000.0f / float(exposure_time)); // [Hz]
    
    front_end_cam->SetCollectionWindow(0.0f); // [sec]
    if (exposure_correction_switch == true) {
        front_end_cam->PushFilter(chrono_types::make_shared<ChFilterCameraExposureCorrect>(
            expsr_a_arr[0], expsr_a_arr[1], expsr_a_arr[2],
            expsr_b_arr[0], expsr_b_arr[1], expsr_b_arr[2], expsr_time + 0.300f
        ));
    }
    if (vis) {
        front_end_cam->PushFilter(chrono_types::make_shared<ChFilterVisualize>(
            front_camera_width, front_camera_height, "Front-end Camera"
        ));
    }
    if (save) {
        front_end_cam->PushFilter(chrono_types::make_shared<ChFilterSave>(
            out_dir + "front_end_cam_" + argv[1] + "_" + brdf_type + "_" + expsr_time_str.str() + "/"
        ));
    }
    manager->AddSensor(front_end_cam);

    // // Create the Irrlicht visualization sys
    // auto vis = chrono_types::make_shared<ChVisualSystemIrrlicht>();Speed
    // vis->AddCamera(ChVector3d(2.0, 0.0, 1.4), ChVector3d(0, 0, wheel_range));
    // vis->AddTypicalLights();
    // vis->AddLightWithShadow(ChVector3d(-5.0, -0.5, 8.0), ChVector3d(-1, 0, 0), 100, 1, 35, 85, 512,
    //                         ChColor(0.8f, 0.8f, 0.8f));
    // vis->EnableShadows();
        // std::chrono::high_resolution_clock::time_point tm1 = std::chrono::high_resolution_clock::now();
    
    ChTimer timer; // [sec], wall time in real world
    float ch_time = 0.0; // [sec], simulation time in Chrono
    float cout_timer = 0.1; // [sec], time to print out something
    float orbit_radius = 100.0f;
    float orbit_rate = 0.5f;
	double max_steering = CH_PI / 3;
	double steering = 0;
    while (ch_time < end_time) {
    // while (vis->Run()) {   
    
    //     vis->BeginScene();
    //     vis->GetActiveCamera()->setTarget(core::vector3dfCH(Body_1->GetPos()));
    //     vis->Render();
    //     tools::drawColorbar(vis.get(), 0, 20000, "Pressure yield [Pa]", 1180);
    //     vis->EndScene();# Make ssh dir
        
        
        ch_time = (float)sys.GetChTime();
        // std::cout << "Sim time: " << ch_time << " RTF: " << timer() / ch_time << std::endl; 
        if (ch_time > cout_timer) {
			// time_t timenow = time(NULL);
			auto timenow = std::chrono::system_clock::to_time_t(std::chrono::high_resolution_clock::now());
			char* time_str = ctime(&timenow);
    		time_str[strlen(time_str) - 1] = '\0'; // remove the '\n' character
            std::cout << "[" << time_str << "] " << "Sim. time: " << ch_time << " [sec], RTF: " << timer() / ch_time << std::endl;
            // std::cout << viper.GetChassis()->GetBody()->GetPos() << std::endl; // debug
            cout_timer += 0.1;
        }
        
        // --- Comment these lines line if you don't want the "sun" moving
        // PointLight p = {{orbit_radius * cos(ch_time * orbit_rate), orbit_radius * sin(ch_time * orbit_rate), 15.0f} , {1.0f,1.0f,1.0f}, 2000.0f};
        // manager->scene->ModifyPointLight(0,p);
        
        timer.start();
        if (debug_terrain == false) {
            sys.DoStepDynamics(2.5e-4);
            // sys.DoStepDynamics(2.5e-3); // debug
        }
        timer.stop();

        manager->Update();
        
        // std::chrono::high_resolution_clock::time_point tm1 = std::chrono::high_resolution_clock::now();
        
        //// steering policy ////
        // open-loop
        // if (11 < ch_time && ch_time < 13) {
        //     steering = 0.25f * max_steering * (13 - ch_time);
        // }
		// else {
		// 	steering = 0;
		// }
		// P-control
		double y_offset = viper.GetChassis()->GetBody()->GetPos().y();
		// steering = 0.25f * max_steering * std::max(-1.0, std::min(y_offset, 1.0)); // all_collidable_v1
		steering = 1.0 * max_steering * std::max(-1.0, std::min(y_offset, 1.0)); // v2

        driver->SetSteering(steering); // debug, stop VIPER
        viper.Update(); // debug, stop VIPER
        

        // std::chrono::high_resolution_clock::time_point tm2 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> wall_time_m = std::chrono::duration_cast<std::chrono::duration<double>>(tm2 - tm1);
        // std::cout << "wall time: " << wall_time_m.count() << "s.\n";

        ////terrain.PrintStepStatistics(std::cout);
    }

    return 0;
}

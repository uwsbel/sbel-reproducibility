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
// Authors: Huzaifa Mustafa Unjhawala and Khailanii Slaton
// =============================================================================
//
// Demo code for a lunar lander simulation on CRM terrain
// - Cylinder body with 4 rigidly attached legs and footpads
// - Drops onto CRM (SPH) terrain under configurable planetary gravity
// - Optional randomized heightmap terrain with embedded rocks
// - Visualized with VSG
//
// =============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "model/Lander.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChLinkLock.h"
#include "chrono/physics/ChContactMaterialNSC.h"
#include "chrono/core/ChRealtimeStep.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/input_output/ChUtilsInputOutput.h"
#include "chrono/assets/ChVisualSystem.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/assets/ChVisualShapeTriangleMesh.h"
#include "chrono/collision/ChCollisionShapeTriangleMesh.h"
#include "chrono_postprocess/ChBlender.h"

#include "chrono_fsi/sph/ChFsiSystemSPH.h"
#include "chrono_fsi/sph/ChFsiProblemSPH.h"
#include "chrono_fsi/sph/ChFsiDefinitionsSPH.h"
#include "chrono_fsi/sph/visualization/ChSphVisualizationVSG.h"

#include "chrono_vehicle/ChVehicleDataPath.h"

#include "chrono/utils/ChUtils.h"
#include "chrono_thirdparty/stb/stb.h"
#include "chrono_thirdparty/cxxopts/ChCLI.h"

#ifdef CHRONO_VSG
    #include "chrono_vsg/ChVisualSystemVSG.h"
using namespace chrono::vsg3d;
#endif

#include "demos/SetChronoSolver.h"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fsi::sph;
using namespace chrono::vehicle;
namespace filesystem = std::filesystem;

// -----------------------------------------------------------------------------
// Global constants
// -----------------------------------------------------------------------------

// Gravity acceleration (m/s^2)
const double LUNAR_GRAVITY = 1.62;
const double EARTH_GRAVITY = 9.81;
const double MARS_GRAVITY = 3.71;

// Platform parameters
const double TERRAIN_SIZE_X = 6.0;  // meters
const double TERRAIN_SIZE_Y = 6.0;  // meters
const double TERRAIN_SIZE_Z = 0.3;  // meters (soil depth under surface)
const int HEIGHTMAP_RESOLUTION = 256;
const int MAX_CRATERS = 8;                           // total features (craters + bumps)
const double MAX_CRATER_DIAMETER = 2.0;              // meters
const double MAX_CRATER_DEPTH = 0.3;                 // meters
const double TERRAIN_HEIGHT_MIN = 0.0;               // meters (black)
const double TERRAIN_HEIGHT_MAX = MAX_CRATER_DEPTH;  // meters (white)
const bool USE_NOISE_LIB = false;

// Rock parameters
const int NUM_ROCKS = 4;
const bool EMBED_ROCKS = true;
const double ROCK_EMBED_FRACTION = 0.4;  // fraction of height below surface when embedded
const double ROCK_DENSITY = 2500.0;      // kg/m^3
const double ROCK_SCALE_MIN = 0.6;
const double ROCK_SCALE_MAX = 1.0;
const double ROCK_CLEARANCE = 0.05;         // meters above surface
const double ROCK_FOOTPAD_CLEARANCE = 0.1;  // meters

// Footpad parameters
const double FOOTPAD_DIAMETER = 0.5;
const double FOOTPAD_HEIGHT = 0.1;
const double FOOTPAD_OFFSET = 0.02;
const double FOOTPAD_MASS = 2.0;

// Drop height (2m above the legs) - used to calculate initial velocity if USE_DROP_HEIGHT_MODE is true
const double DROP_HEIGHT = 2.0;

// Initial velocity mode switch: true = calculate from drop height (v = sqrt(2*g*h)), false = use INITIAL_VELOCITY
// directly
const bool USE_DROP_HEIGHT_MODE = false;  // Set to false to use INITIAL_VELOCITY directly

// Initial velocity (m/s) - used when USE_DROP_HEIGHT_MODE is false (negative = downward)
const double INITIAL_VELOCITY = -1.0;  // m/s downward

// Clearance above ground when starting simulation (m)
const double GROUND_CLEARANCE = 0.05;  // meters
const double t_end = 1.2;
const bool USE_ACTIVE_DOMAIN = true;

// Output parameters
const double output_fps = 100.0;  // Output FPS for CSV and snapshots
const bool snapshots = true;      // Enable snapshot saving

const bool ADD_LANDERBODY_BCE = true;
const bool ADD_LANDERLEGS_BCE = true;

// -----------------------------------------------------------------------------
// Small utility functions
// -----------------------------------------------------------------------------

bool EnsureDirectory(const filesystem::path& dir) {
    try {
        filesystem::create_directories(dir);

        if (!filesystem::exists(dir)) {
            std::cerr << "Directory was not created: " << filesystem::absolute(dir) << std::endl;
            return false;
        }

        if (!filesystem::is_directory(dir)) {
            std::cerr << "Path exists but is not a directory: " << filesystem::absolute(dir) << std::endl;
            return false;
        }
    } catch (const filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error while creating directory: " << filesystem::absolute(dir) << "\n"
                  << e.what() << std::endl;
        return false;
    }

    return true;
}

std::string BuildOutputDirectory(const std::string& rheology_model_crm,
                                 const std::string& gravity_planet,
                                 double gravity_polar_deg,
                                 double gravity_azimuth_deg,
                                 double pre_pressure_scale,
                                 double kappa,
                                 double lambda) {
    std::stringstream ss;
    ss << std::fixed;
    ss << GetChronoOutputPath() << "ROBOT_Lander_CRM";
    ss << "_" << rheology_model_crm;
    ss << "_gravity_planet_" << gravity_planet;
    ss << "_gravity_polar_deg_" << gravity_polar_deg;
    ss << "_gravity_azimuth_deg_" << gravity_azimuth_deg;
    if (rheology_model_crm == "MCC") {
        ss << "_pre_pressure_scale_" << std::setprecision(2) << pre_pressure_scale;
        ss << "_kappa_" << std::setprecision(2) << kappa;
        ss << "_lambda_" << std::setprecision(2) << lambda;
    }
    ss << "/";
    return ss.str();
}

// -----------------------------------------------------------------------------
// Terrain / rock helpers
// -----------------------------------------------------------------------------

class SPHPropertiesCallbackWithPressureScale : public ChFsiProblemSPH::ParticlePropertiesCallback {
  public:
    SPHPropertiesCallbackWithPressureScale(double zero_height, double pre_pressure_scale)
        : ParticlePropertiesCallback(), zero_height(zero_height), pre_pressure_scale(pre_pressure_scale) {}

    virtual void set(const ChFsiFluidSystemSPH& sysSPH, const ChVector3d& pos) override {
        double gz = std::abs(sysSPH.GetGravitationalAcceleration().z());
        p0 = sysSPH.GetDensity() * gz * (zero_height - pos.z());
        rho0 = sysSPH.GetDensity();
        mu0 = sysSPH.GetViscosity();
        v0 = ChVector3d(0, 0, 0);
        pre_pressure_scale0 = pre_pressure_scale;
    }

    double zero_height;
    double pre_pressure_scale;
};

struct HeightmapData {
    int nx = 0;
    int ny = 0;
    double range = 0.0;
    std::vector<unsigned short> gmap;
};

bool LoadHeightmap(const std::string& heightmap_file, HeightmapData& data) {
    STB cmap;
    if (!cmap.ReadFromFile(heightmap_file, 1)) {
        std::cerr << "Cannot open height map image file " << heightmap_file << std::endl;
        return false;
    }

    data.nx = cmap.GetWidth();
    data.ny = cmap.GetHeight();
    data.range = cmap.GetRange();
    data.gmap.resize(data.nx * data.ny);

    for (int ix = 0; ix < data.nx; ix++) {
        for (int iy = 0; iy < data.ny; iy++) {
            data.gmap[ix * data.ny + iy] = cmap.Gray(ix, iy);
        }
    }

    return true;
}

double SampleHeightmap(const HeightmapData& data,
                       double length,
                       double width,
                       const ChVector2d& height_range,
                       double x,
                       double y) {
    double x_local = std::clamp(x + length / 2.0, 0.0, length);
    double y_local = std::clamp(width / 2.0 - y, 0.0, width);

    double dx = length / (data.nx - 1);
    double dy = width / (data.ny - 1);

    int ix1 = (x_local >= length) ? data.nx - 2 : (int)std::floor(x_local / dx);
    int ix2 = (x_local <= 0) ? 1 : (int)std::ceil(x_local / dx);
    double wx = (ix2 * dx - x_local) / dx;

    int iy1 = (y_local >= width) ? data.ny - 2 : (int)std::floor(y_local / dy);
    int iy2 = (y_local <= 0) ? 1 : (int)std::ceil(y_local / dy);
    double wy = (iy2 * dy - y_local) / dy;

    auto g11 = data.gmap[ix1 * data.ny + iy1];
    auto g12 = data.gmap[ix1 * data.ny + iy2];
    auto g21 = data.gmap[ix2 * data.ny + iy1];
    auto g22 = data.gmap[ix2 * data.ny + iy2];

    auto h1 = wx * g11 + (1 - wx) * g21;
    auto h2 = wx * g12 + (1 - wx) * g22;
    double h_scale = (height_range[1] - height_range[0]) / data.range;

    return height_range[0] + (wy * h1 + (1 - wy) * h2) * h_scale;
}

ChQuaternion<> RandomQuaternion(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 2.0 * CH_PI);
    return QuatFromAngleX(dist(rng)) * QuatFromAngleY(dist(rng)) * QuatFromAngleZ(dist(rng));
}

bool GenerateHeightmap(const std::string& script_file,
                       const std::string& output_file,
                       int resolution,
                       double length,
                       double width,
                       double height_min,
                       double height_max,
                       int max_craters,
                       double max_crater_diameter,
                       double max_crater_depth,
                       bool use_noise_lib,
                       unsigned int seed) {
    auto run_generator = [&](const std::string& python_cmd) {
        std::ostringstream command;
        command << python_cmd << " " << std::quoted(script_file) << " --output " << std::quoted(output_file)
                << " --resolution " << resolution << " --length " << length << " --width " << width << " --height-min "
                << height_min << " --height-max " << height_max << " --max-craters " << max_craters
                << " --max-crater-diameter " << max_crater_diameter << " --max-crater-depth " << max_crater_depth
                << " --seed " << seed;
        if (use_noise_lib) {
            command << " --use-noise-lib";
        }
        return std::system(command.str().c_str());
    };

    int result = run_generator("python3");
    if (result != 0) {
        result = run_generator("python");
    }
    return result == 0;
}

int main(int argc, char* argv[]) {
    std::cout << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << std::endl;

    // Parse command line arguments
    ChCLI cli(argv[0], "Lunar Lander on CRM Terrain Demo");

    // Default values
    std::string rheology_model_crm = "MU_OF_I";  // "MU_OF_I" or "MCC"
    double kappa = 0.01;
    double lambda = 0.04;
    double pre_pressure_scale = 2.0;
    bool no_vis = false;
    bool flat_terrain = false;
    bool blender_output = false;
    bool particle_output = false;
    bool add_landerbody_bce = ADD_LANDERBODY_BCE;
    bool add_landerlegs_bce = ADD_LANDERLEGS_BCE;
    double gravity_polar_deg = 0.0;
    double gravity_azimuth_deg = 0.0;
    double gravity_magnitude = LUNAR_GRAVITY;
    std::string gravity_planet = "moon";

    cli.AddOption<std::string>("Physics", "rheology_model_crm", "Rheology model (MU_OF_I/MCC)", rheology_model_crm);
    cli.AddOption<double>("Physics", "pre_pressure_scale", "Pre-pressure scale", std::to_string(pre_pressure_scale));
    cli.AddOption<double>("Physics", "kappa", "kappa", std::to_string(kappa));
    cli.AddOption<double>("Physics", "lambda", "lambda", std::to_string(lambda));
    cli.AddOption<std::string>("Visualization", "no_vis", "Disable visualization", "false");
    cli.AddOption<std::string>("Terrain", "flat_terrain", "Use a flat 6 x 6 x 0.3 m terrain patch", "false");
    cli.AddOption<std::string>("Output", "blender_output", "Save Blender assets and per-frame state files", "false");
    cli.AddOption<std::string>("Output", "particle_output", "Save CRM particle and rigid-body CSV output", "false");
    cli.AddOption<std::string>("FSI", "lander_body_bce", "Add BCE markers for the lander body",
                               add_landerbody_bce ? "true" : "false");
    cli.AddOption<std::string>("FSI", "lander_legs_bce", "Add BCE markers for the lander legs",
                               add_landerlegs_bce ? "true" : "false");
    cli.AddOption<double>("Physics", "gravity_polar_deg", "Gravity polar angle (degrees)",
                          std::to_string(gravity_polar_deg));
    cli.AddOption<double>("Physics", "gravity_azimuth_deg", "Gravity azimuth angle (degrees)",
                          std::to_string(gravity_azimuth_deg));
    cli.AddOption<std::string>("Physics", "gravity_planet", "Gravity planet (earth/mars/moon)", gravity_planet);
    if (!cli.Parse(argc, argv))
        return 1;

    rheology_model_crm = cli.GetAsType<std::string>("rheology_model_crm");
    pre_pressure_scale = cli.GetAsType<double>("pre_pressure_scale");
    kappa = cli.GetAsType<double>("kappa");
    lambda = cli.GetAsType<double>("lambda");
    no_vis = parse_bool(cli.GetAsType<std::string>("no_vis"));
    flat_terrain = parse_bool(cli.GetAsType<std::string>("flat_terrain"));
    blender_output = parse_bool(cli.GetAsType<std::string>("blender_output"));
    particle_output = parse_bool(cli.GetAsType<std::string>("particle_output"));
    add_landerbody_bce = parse_bool(cli.GetAsType<std::string>("lander_body_bce"));
    add_landerlegs_bce = parse_bool(cli.GetAsType<std::string>("lander_legs_bce"));
    gravity_polar_deg = cli.GetAsType<double>("gravity_polar_deg");
    gravity_azimuth_deg = cli.GetAsType<double>("gravity_azimuth_deg");
    gravity_planet = cli.GetAsType<std::string>("gravity_planet");
    bool enable_vis = !no_vis;
    double terrain_surface_top = flat_terrain ? TERRAIN_SIZE_Z : TERRAIN_HEIGHT_MAX;

    // Convert to lowercase for case-insensitive comparison
    std::transform(gravity_planet.begin(), gravity_planet.end(), gravity_planet.begin(), ::tolower);

    if (gravity_planet == "earth") {
        gravity_magnitude = EARTH_GRAVITY;
    } else if (gravity_planet == "mars") {
        gravity_magnitude = MARS_GRAVITY;
    } else if (gravity_planet == "moon") {
        gravity_magnitude = LUNAR_GRAVITY;
    }

    // Create the physical system
    ChSystemNSC sys;
    double step_size = 5e-4;
    double mu_s = 0.6;
    double initial_spacing = 0.02;
    ChVector3d active_box_dim(FOOTPAD_DIAMETER * 4.0, FOOTPAD_DIAMETER * 4.0, FOOTPAD_HEIGHT * 10.0);
    sys.SetCollisionSystemType(ChCollisionSystem::Type::BULLET);
    SetChronoSolver(sys, ChSolver::Type::BARZILAIBORWEIN, ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);

    // Set gravity (downward in Z direction, tilted by polar/azimuth angles)
    double gravity_x = -gravity_magnitude * std::sin(gravity_polar_deg * CH_PI / 180.0) *
                       std::cos(gravity_azimuth_deg * CH_PI / 180.0);
    double gravity_y = -gravity_magnitude * std::sin(gravity_polar_deg * CH_PI / 180.0) *
                       std::sin(gravity_azimuth_deg * CH_PI / 180.0);
    double gravity_z = -gravity_magnitude * std::cos(gravity_polar_deg * CH_PI / 180.0);
    sys.SetGravitationalAcceleration(ChVector3d(gravity_x, gravity_y, gravity_z));

    // Create contact material
    auto contact_material = chrono_types::make_shared<ChContactMaterialNSC>();
    contact_material->SetFriction(0.7f);
    contact_material->SetRestitution(0.05f);

    // =============================================================================
    // Create the lander using the Lander class
    // =============================================================================
    Lander lander(&sys, contact_material);

    // Position the lander: place it just above the ground (GROUND_CLEARANCE)
    // The lowest point (footpad or leg end) will be at platform_top + GROUND_CLEARANCE
    double platform_top = terrain_surface_top;
    ChVector3d lowest_point_pos(0, 0, platform_top);
    ChFrame<> lander_pos(lowest_point_pos, QUNIT);
    lander.SetFootpadParameters(FOOTPAD_DIAMETER, FOOTPAD_HEIGHT, FOOTPAD_OFFSET, FOOTPAD_MASS);
    lander.SetUseFootpads(true);
    lander.SetUseSphericalJoint(false);

    // Initialize the lander at the specified position
    lander.Initialize(lander_pos, GROUND_CLEARANCE);

    // =============================================================================
    // Set initial velocity to simulate drop from height
    // =============================================================================
    if (USE_DROP_HEIGHT_MODE) {
        // Calculate velocity from drop height: v = sqrt(2 * g * h)
        lander.SetInitialVelocityFromDropHeight(DROP_HEIGHT, LUNAR_GRAVITY);
    } else {
        // Use direct velocity value (should be negative for downward)
        lander.SetInitialVelocity(ChVector3d(0, 0, INITIAL_VELOCITY));
    }

    // ================================================================
    // Create the FSI problem
    // ================================================================
    ChFsiProblemCartesian fsi(initial_spacing, &sys);
    fsi.SetVerbose(true);
    auto sysFSI = fsi.GetFsiSystemSPH();

    // Set gravitational acceleration
    fsi.SetGravitationalAcceleration(ChVector3d(gravity_x, gravity_y, gravity_z));

    // Set integration step size
    fsi.SetStepSizeCFD(step_size);
    fsi.SetStepsizeMBD(step_size);

    // Set soil properties
    ChFsiFluidSystemSPH::ElasticMaterialProperties mat_props;
    mat_props.density = 1700;
    mat_props.Young_modulus = 1e6;
    mat_props.Poisson_ratio = 0.3;
    if (rheology_model_crm == "MU_OF_I") {
        mat_props.rheology_model = RheologyCRM::MU_OF_I;
        mat_props.mu_I0 = 0.04;
        mat_props.mu_fric_s = mu_s;
        mat_props.mu_fric_2 = mu_s;
        mat_props.average_diam = 0.005;
        mat_props.cohesion_coeff = 0;  // default
    } else {
        mat_props.rheology_model = RheologyCRM::MCC;
        double angle_mus = std::atan(mu_s);
        mat_props.mcc_M = (6 * std::sin(angle_mus)) / (3 - std::sin(angle_mus));
        std::cout << "MCC M: " << mat_props.mcc_M << std::endl;
        mat_props.mcc_kappa = kappa;
        mat_props.mcc_lambda = lambda;
    }
    fsi.SetElasticSPH(mat_props);

    // Set SPH solution parameters
    ChFsiFluidSystemSPH::SPHParameters sph_params;
    sph_params.num_bce_layers = 3;
    sph_params.integration_scheme = IntegrationScheme::RK2;
    sph_params.initial_spacing = initial_spacing;
    sph_params.d0_multiplier = 1.3;
    sph_params.artificial_viscosity = 0.2;
    sph_params.viscosity_method = ViscosityMethod::ARTIFICIAL_BILATERAL;
    sph_params.kernel_type = KernelType::WENDLAND;
    sph_params.boundary_method = BoundaryMethod::HOLMES;
    sph_params.shifting_method = ShiftingMethod::PPST_XSPH;
    sph_params.shifting_xsph_eps = 0.25;
    sph_params.shifting_ppst_pull = 1.0;
    sph_params.shifting_ppst_push = 3.0;
    sph_params.free_surface_threshold = 2.0;
    sph_params.num_proximity_search_steps = 1;
    sph_params.use_variable_time_step = true;
    sph_params.use_delta_sph = true;
    fsi.SetSPHParameters(sph_params);
    fsi.SetOutputLevel(OutputLevel::STATE);

    fsi.RegisterParticlePropertiesCallback(
        chrono_types::make_shared<SPHPropertiesCallbackWithPressureScale>(terrain_surface_top, pre_pressure_scale));

    // Add the footpads as FSI bodies
    auto footpads = lander.GetFootpads();

    auto geometry = chrono_types::make_shared<utils::ChBodyGeometry>();
    geometry->coll_cylinders.push_back(utils::ChBodyGeometry::CylinderShape(ChVector3d(0, 0, 0), ChVector3d(0, 0, 1),
                                                                            FOOTPAD_DIAMETER / 2.0, FOOTPAD_HEIGHT));
    for (auto& footpad : footpads) {
        fsi.AddRigidBody(footpad, geometry, false);
    }
    if (add_landerlegs_bce) {
        auto leg_length = lander.GetLegLength();
        auto leg_radius = lander.GetLegRadius();
        auto leg_geometry = chrono_types::make_shared<utils::ChBodyGeometry>();
        leg_geometry->coll_cylinders.push_back(
            utils::ChBodyGeometry::CylinderShape(ChVector3d(0, 0, 0), ChVector3d(0, 0, 1), leg_radius, leg_length));
        auto legs = lander.GetLegs();
        for (auto& leg : legs) {
            fsi.AddRigidBody(leg, leg_geometry, false);
        }
    }
    if (add_landerbody_bce) {
        auto cylinder_length = lander.GetCylinderLength();
        auto cylinder_radius = lander.GetCylinderRadius();
        auto cylinder_geometry = chrono_types::make_shared<utils::ChBodyGeometry>();
        cylinder_geometry->coll_cylinders.push_back(utils::ChBodyGeometry::CylinderShape(
            ChVector3d(0, 0, 0), ChVector3d(0, 0, 1), cylinder_radius, cylinder_length));
        auto lander_body = lander.GetBody();
        fsi.AddRigidBody(lander_body, cylinder_geometry, false);
    }
    if (USE_ACTIVE_DOMAIN) {
        if (add_landerbody_bce) {
            // Make box bigger
            auto cylinder_length = lander.GetCylinderLength();
            active_box_dim.z() = cylinder_length;
            fsi.SetActiveDomain(active_box_dim);
        } else {
            fsi.SetActiveDomain(active_box_dim);
        }
    }

    std::cout << "Constructing terrain..." << std::endl;
    if (flat_terrain) {
        fsi.Construct(ChVector3d(TERRAIN_SIZE_X, TERRAIN_SIZE_Y, TERRAIN_SIZE_Z), ChVector3d(0, 0, 0), BoxSide::Z_NEG);
    } else {
        // NOTE: This demo generates a randomized heightmap at runtime.
        // Dependencies: Python 3 + numpy + Pillow (optional: noise for faster Perlin).
        const std::string heightmap_script = std::string(DEMO_SOURCE_DIR) + "/generate_heightmap.py";
        const std::string heightmap_file = GetChronoDataFile("robot/lander/terrain_heightmap.png");
        const auto heightmap_dir = filesystem::path(heightmap_file).parent_path();
        if (!EnsureDirectory(heightmap_dir)) {
            return 1;
        }
        if (!filesystem::exists(filesystem::path(heightmap_script))) {
            std::cerr << "Heightmap generator script not found at " << heightmap_script << std::endl;
            return 1;
        }

        std::random_device rd;
        unsigned int heightmap_seed = rd();
        std::cout << "Generating heightmap (seed=" << heightmap_seed << ")..." << std::endl;
        if (!GenerateHeightmap(heightmap_script, heightmap_file, HEIGHTMAP_RESOLUTION, TERRAIN_SIZE_X, TERRAIN_SIZE_Y,
                               TERRAIN_HEIGHT_MIN, TERRAIN_HEIGHT_MAX, MAX_CRATERS, MAX_CRATER_DIAMETER,
                               MAX_CRATER_DEPTH, USE_NOISE_LIB, heightmap_seed)) {
            std::cerr << "Heightmap generation failed. Check that Python and dependencies are available." << std::endl;
            return 1;
        }

        HeightmapData heightmap_data;
        if (!LoadHeightmap(heightmap_file, heightmap_data)) {
            return 1;
        }

        // Spawn randomized rocks using mesh libraries from the data folder.
        const std::array<std::string, 8> rock_mesh_library = {
            "robot/curiosity/rocks/rock1.obj", "robot/curiosity/rocks/rock2.obj", "robot/curiosity/rocks/rock3.obj",
            "sensor/offroad/rock1.obj",        "sensor/offroad/rock2.obj",        "sensor/offroad/rock3.obj",
            "sensor/offroad/rock4.obj",        "sensor/offroad/rock5.obj",
        };

        std::mt19937 rock_rng(heightmap_seed);
        std::uniform_int_distribution<int> rock_mesh_dist(0, static_cast<int>(rock_mesh_library.size()) - 1);
        std::uniform_real_distribution<double> rock_scale_dist(ROCK_SCALE_MIN, ROCK_SCALE_MAX);
        const double half_x = TERRAIN_SIZE_X * 0.5;
        const double half_y = TERRAIN_SIZE_Y * 0.5;
        const double footpad_radius = lander.GetFootpadRadius();
        std::vector<ChVector3d> footpad_positions;
        footpad_positions.reserve(4);
        for (const auto& footpad : lander.GetFootpads()) {
            if (footpad)
                footpad_positions.push_back(footpad->GetPos());
        }

        for (int i = 0; i < NUM_ROCKS; i++) {
            const auto& rock_file = rock_mesh_library[rock_mesh_dist(rock_rng)];
            const std::string rock_mesh_file = GetChronoDataFile(rock_file);
            double rock_scale = rock_scale_dist(rock_rng);

            auto trimesh = ChTriangleMeshConnected::CreateFromWavefrontFile(rock_mesh_file, false, false);
            if (!trimesh) {
                std::cerr << "Cannot load rock mesh " << rock_mesh_file << std::endl;
                return 1;
            }

            auto aabb_unscaled = trimesh->GetBoundingBox();
            ChVector3d size_unscaled = aabb_unscaled.max - aabb_unscaled.min;
            double max_dim = std::max({size_unscaled.x(), size_unscaled.y(), size_unscaled.z(), 1e-6});
            double scale_base = 1.0 / max_dim;
            double scale_factor = rock_scale;
            if (scale_factor > 1.0) {
                scale_factor = 1.0;
            }
            rock_scale = scale_base * scale_factor;

            trimesh->Transform(VNULL, ChMatrix33<>(rock_scale));
            auto aabb = trimesh->GetBoundingBox();
            ChVector3d size = aabb.max - aabb.min;

            double margin_x = std::min(half_x - 0.01, 0.5 * size.x());
            double margin_y = std::min(half_y - 0.01, 0.5 * size.y());
            std::uniform_real_distribution<double> rock_x_dist(-half_x + margin_x, half_x - margin_x);
            std::uniform_real_distribution<double> rock_y_dist(-half_y + margin_y, half_y - margin_y);

            double rock_x = 0;
            double rock_y = 0;
            bool placed = false;
            const double rock_xy_radius = 0.5 * std::sqrt(size.x() * size.x() + size.y() * size.y());
            for (int attempt = 0; attempt < 50; attempt++) {
                rock_x = rock_x_dist(rock_rng);
                rock_y = rock_y_dist(rock_rng);
                bool overlaps_footpad = false;
                for (const auto& pos : footpad_positions) {
                    double dx = rock_x - pos.x();
                    double dy = rock_y - pos.y();
                    double min_sep = rock_xy_radius + footpad_radius + ROCK_FOOTPAD_CLEARANCE;
                    if (dx * dx + dy * dy < min_sep * min_sep) {
                        overlaps_footpad = true;
                        break;
                    }
                }
                if (!overlaps_footpad) {
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                std::cerr << "Failed to place rock " << (i + 1) << " without overlapping footpads" << std::endl;
            }
            double rock_z_surface = SampleHeightmap(heightmap_data, TERRAIN_SIZE_X, TERRAIN_SIZE_Y,
                                                    {TERRAIN_HEIGHT_MIN, TERRAIN_HEIGHT_MAX}, rock_x, rock_y);
            double rock_z = rock_z_surface + 0.5 * size.z() + ROCK_CLEARANCE;
            if (EMBED_ROCKS) {
                double embed_fraction = std::clamp(ROCK_EMBED_FRACTION, 0.0, 1.0);
                rock_z = rock_z_surface + (0.5 - embed_fraction) * size.z();
            }

            // Approximate inertia using the rock's AABB.
            double volume = std::max(1e-6, size.x() * size.y() * size.z());
            double mass = ROCK_DENSITY * volume;
            double Ixx = mass * (size.y() * size.y() + size.z() * size.z()) / 12.0;
            double Iyy = mass * (size.x() * size.x() + size.z() * size.z()) / 12.0;
            double Izz = mass * (size.x() * size.x() + size.y() * size.y()) / 12.0;

            auto rock_body = chrono_types::make_shared<ChBody>();
            rock_body->SetName("Rock" + std::to_string(i + 1));
            rock_body->SetPos(ChVector3d(rock_x, rock_y, rock_z));
            rock_body->SetRot(RandomQuaternion(rock_rng));
            rock_body->SetMass(mass);
            rock_body->SetInertiaXX(ChVector3d(Ixx, Iyy, Izz));
            rock_body->SetFixed(false);
            rock_body->EnableCollision(true);

            auto rock_coll =
                chrono_types::make_shared<ChCollisionShapeTriangleMesh>(contact_material, trimesh, false, false);
            rock_body->AddCollisionShape(rock_coll);

            auto rock_vis = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
            rock_vis->SetMesh(trimesh, false);
            rock_body->AddVisualShape(rock_vis);

            sys.AddBody(rock_body);
            ChVector3d interior_point = 0.5 * (aabb.min + aabb.max);
            auto rock_geometry = chrono_types::make_shared<utils::ChBodyGeometry>();
            rock_geometry->coll_meshes.push_back(
                utils::ChBodyGeometry::TrimeshShape(VNULL, QUNIT, trimesh, interior_point, 1.0, 0.0, 0));
            fsi.AddRigidBody(rock_body, rock_geometry, EMBED_ROCKS);
        }

        fsi.Construct(heightmap_file,                            // height map image file
                      TERRAIN_SIZE_X, TERRAIN_SIZE_Y,            // length (X) and width (Y)
                      {TERRAIN_HEIGHT_MIN, TERRAIN_HEIGHT_MAX},  // height range
                      TERRAIN_SIZE_Z,                            // depth
                      true,                                      // uniform depth
                      ChVector3d(0, 0, 0),                       // patch center
                      BoxSide::ALL & ~BoxSide::Z_POS             // bottom wall
        );
    }

    fsi.Initialize();

    // =============================================================================
    // Create output directories
    // =============================================================================
    std::string out_dir = BuildOutputDirectory(rheology_model_crm, gravity_planet, gravity_polar_deg,
                                               gravity_azimuth_deg, pre_pressure_scale, kappa, lambda);
    if (!EnsureDirectory(out_dir)) {
        return 1;
    }

    // Create snapshots directory if enabled and visualization is enabled
    if (snapshots && enable_vis) {
        if (!EnsureDirectory(out_dir + "snapshots")) {
            return 1;
        }
    }

    std::string particles_dir;
    std::string fsi_dir;
    if (particle_output) {
        particles_dir = out_dir + "particles";
        fsi_dir = out_dir + "fsi";
        if (!EnsureDirectory(particles_dir) || !EnsureDirectory(fsi_dir)) {
            return 1;
        }
    }

    const ChVector3d camera_target = lander.GetBody()->GetPos();
    const ChVector3d camera_position = camera_target + ChVector3d(13.5, -13.5, 0.0);

    std::string blender_dir;
    postprocess::ChBlender blender_exporter(&sys);
    if (blender_output) {
        blender_dir = out_dir + "blender";
        if (!EnsureDirectory(blender_dir)) {
            return 1;
        }

        blender_exporter.SetBlenderUp_is_ChronoZ();
        blender_exporter.SetBasePath(blender_dir);
        blender_exporter.SetCamera(camera_position, camera_target, 50);
        blender_exporter.SetPictureSize(1920, 1080);
        blender_exporter.AddAll();
        blender_exporter.ExportScript();
    }

    // Get SPH system for saving particles and markers
    auto sysSPH = fsi.GetFluidSystemSPH();

    // Open CSV file for lander body data
    std::string csv_file = out_dir + "lander_body_data.csv";
    std::ofstream csv_output(csv_file);
    csv_output << "time,px,py,pz,vx,vy,vz,wx,wy,wz" << std::endl;

    // =============================================================================
    // Create VSG visualization
    // =============================================================================
    std::shared_ptr<ChVisualSystem> vis;
    if (enable_vis) {
#ifdef CHRONO_VSG
        // FSI plugin
        auto col_callback = chrono_types::make_shared<ParticleHeightColorCallback>(
            TERRAIN_HEIGHT_MIN, terrain_surface_top + FOOTPAD_HEIGHT);

        auto visFSI = chrono_types::make_shared<ChSphVisualizationVSG>(sysFSI.get());
        visFSI->EnableFluidMarkers(true);
        visFSI->EnableBoundaryMarkers(true);
        visFSI->EnableRigidBodyMarkers(false);
        visFSI->SetSPHColorCallback(col_callback);

        // VSG visual system (attach visFSI as plugin)
        auto visVSG = chrono_types::make_shared<vsg3d::ChVisualSystemVSG>();
        visVSG->AttachPlugin(visFSI);
        visVSG->AttachSystem(&sys);
        visVSG->SetWindowTitle("Lunar Lander Simulation");
        visVSG->SetWindowSize(ChVector2i(1920, 1080));
        visVSG->SetWindowPosition(100, 100);
        visVSG->SetCameraVertical(CameraVerticalDir::Z);
        visVSG->AddCamera(camera_position, camera_target);
        visVSG->SetLightIntensity(1.0f);
        visVSG->SetLightDirection(1.5 * CH_PI_2, CH_PI_4);
        visVSG->SetBackgroundColor(ChColor(0.1f, 0.1f, 0.15f));  // Dark blue-gray (space-like)
        visVSG->Initialize();
        // Always enable active domain visualization after initialization
        if (USE_ACTIVE_DOMAIN) {
            visFSI->SetActiveBoxVisibility(false, -1);  // -1 means all active boxes
        }
        vis = visVSG;
#else
        std::cout << "VSG not available. Cannot visualize." << std::endl;
        return 1;
#endif
    }

    std::cout << "\nStarting simulation..." << std::endl;
    if (!enable_vis) {
        std::cout << "Visualization: DISABLED (--no_vis flag set)" << std::endl;
    }
    std::cout << "Terrain mode: " << (flat_terrain ? "Flat box" : "Heightmap with rocks") << std::endl;
    std::cout << "Blender output: " << (blender_output ? "ENABLED" : "disabled") << std::endl;
    std::cout << "CRM particle output: " << (particle_output ? "ENABLED" : "disabled") << std::endl;
    std::cout << "Lander body BCE: " << (add_landerbody_bce ? "ENABLED" : "disabled") << std::endl;
    std::cout << "Lander legs BCE: " << (add_landerlegs_bce ? "ENABLED" : "disabled") << std::endl;
    std::cout << "Lander mass: " << lander.GetLanderMass() << " kg" << std::endl;
    std::cout << "Cylinder: length=" << lander.GetCylinderLength() << " m, diameter=" << lander.GetCylinderDiameter()
              << " m" << std::endl;
    std::cout << "Lunar gravity: " << LUNAR_GRAVITY << " m/s^2" << std::endl;
    std::cout << "Initial position: " << GROUND_CLEARANCE << " m above platform" << std::endl;
    if (USE_DROP_HEIGHT_MODE) {
        std::cout << "Velocity mode: Calculated from drop height" << std::endl;
        std::cout << "Drop height: " << DROP_HEIGHT << " m (simulated)" << std::endl;
        double initial_velocity_z = -std::sqrt(2.0 * LUNAR_GRAVITY * DROP_HEIGHT);
        std::cout << "Initial velocity: " << initial_velocity_z << " m/s (downward)" << std::endl;
    } else {
        std::cout << "Velocity mode: Direct specification" << std::endl;
        std::cout << "Initial velocity: " << INITIAL_VELOCITY << " m/s (downward)" << std::endl;
    }
    std::cout << "Footpad joint type: "
              << (lander.GetUseSphericalJoint() ? "Spherical (rotating)" : "Rigid lock (fixed)") << std::endl;

    // Simulation loop variables
    double t = 0;
    int render_frame = 0;
    int output_frame = 1;

    auto write_output = [&](double time, bool export_blender_frame) {
        ChVector3d pos = lander.GetBody()->GetPos();
        ChVector3d vel = lander.GetBody()->GetPosDt();
        ChVector3d ang_vel = lander.GetBody()->GetAngVelParent();

        csv_output << time << "," << pos.x() << "," << pos.y() << "," << pos.z() << "," << vel.x() << "," << vel.y()
                   << "," << vel.z() << "," << ang_vel.x() << "," << ang_vel.y() << "," << ang_vel.z() << std::endl;

        if (particle_output) {
            sysSPH->SaveParticleData(particles_dir);
            sysSPH->SaveSolidData(fsi_dir, time);
        }

        if (export_blender_frame) {
            blender_exporter.ExportData();
        }

        std::cout << "Time: " << sys.GetChTime() << " s, "
                  << "Position: (" << pos.x() << ", " << pos.y() << ", " << pos.z() << ") m, "
                  << "Velocity z: " << vel.z() << " m/s" << std::endl;
    };

    write_output(t, false);

    while (t < t_end) {
        if (t >= output_frame / output_fps) {
            write_output(t, blender_output);
            output_frame++;
        }

        // Render and save snapshots
        if (enable_vis && vis && t >= render_frame / output_fps) {
            if (!vis->Run())
                break;
            vis->Render();

            // Save snapshots at output FPS
            if (snapshots) {
                std::cout << " -- Snapshot frame " << render_frame << " at t = " << t << std::endl;
                std::ostringstream filename;
                filename << out_dir << "snapshots/img_" << std::setw(5) << std::setfill('0') << render_frame + 1
                         << ".bmp";
                vis->WriteImageToFile(filename.str());
            }
            render_frame++;
        }

        fsi.DoStepDynamics(step_size);
        t += step_size;
    }

    csv_output.close();
    if (snapshots && enable_vis) {
        std::cout << "Snapshots saved to: " << out_dir << "snapshots/" << std::endl;
    }
    std::cout << "CSV data saved to: " << csv_file << std::endl;
    if (blender_output) {
        std::cout << "Blender data saved to: " << blender_dir << std::endl;
    }
    if (particle_output) {
        std::cout << "CRM particle data saved to: " << particles_dir << std::endl;
    }

    if (particle_output) {
        // Particle CSV writers run in detached worker threads inside the FSI module.
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}
// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Pei Li, Wei Hu
// =============================================================================

#include <assert.h>
#include <stdlib.h>
#include <ctime>

#include "chrono/physics/ChSystemSMC.h"

#ifdef CHRONO_PARDISO_MKL
    #include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#endif

#include "chrono/solver/ChIterativeSolverLS.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"

#include "chrono/fea/ChElementShellANCF_3423.h"
#include "chrono/fea/ChLinkDirFrame.h"
#include "chrono/fea/ChLinkPointFrame.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshExporter.h"
#include "chrono/fea/ChBuilderBeam.h"

#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/core/ChCubicSpline.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"

// Chrono namespaces
using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;
using namespace chrono::fsi;
using namespace chrono::geometry;

// Set the output directory
const std::string out_dir = GetChronoOutputPath() + "FSI_Flexible_Four_Tires/";
std::string MESH_CONNECTIVITY = out_dir + "Flex_MESH.vtk";

// Dimension of the domain
double smalldis = 1.0e-9;
double bxDim = 20.0 + smalldis;
double byDim = 3.0 + smalldis;
double bzDim = 0.3 + smalldis;

// Dimension of the fluid domain
double fxDim = 20.0 + smalldis;
double fyDim = 3.0 + smalldis;
double fzDim = 0.2 + smalldis;
bool flexible_elem_1D = false;

// Size of the wheel
double wheel_radius = 0.35;
double wheel_slip = 0.0;
double wheel_AngVel = 1.0;
double total_mass = 105.22;

//distance of the wheel
double dis_Front_back = 3.3;
double dis_Left_right = 1.83;

// -----------------------------------------------------------------------------
// Static variables  ANCFtire
// -----------------------------------------------------------------------------
const double m_tire_radius = 0.4673;
const double m_rim_radius = 0.2683;
const double m_rim_width = 0.254;

const double m_alpha = 0.005;
const double m_default_pressure = 200e3;

const double m_rho_0 = 0.1e4;
const ChVector<> m_E_0(0.756e10, 0.474e8, 0.474e8);
const ChVector<> m_nu_0(0.45, 0.45, 0.45);
const ChVector<> m_G_0(0.1634e8, 0.1634e8, 0.1634e8);
const double m_rho_1 = 0.2639e4;
const ChVector<> m_E_1(0.18e12, 0.474e8, 0.474e8);
const ChVector<> m_nu_1(0.45, 0.45, 0.45);
const ChVector<> m_G_1(0.1634e8, 0.1634e8, 0.1634e8);
const double m_rho_2 = 0.11e4;
const ChVector<> m_E_2(0.474e8, 0.474e8, 0.474e8);
const ChVector<> m_nu_2(0.45, 0.45, 0.45);
const ChVector<> m_G_2(0.1634e8, 0.1634e8, 0.1634e8);

const unsigned int m_num_elements_bead = 2;
const unsigned int m_num_layers_bead = 3;
const std::vector<double> m_layer_thickness_bead{{0.5e-03, 0.5e-02, 0.5e-03}};
const std::vector<double> m_ply_angle_bead{{90, 0, 90}};
const std::vector<int> m_material_id_bead{{0, 2, 0}};

const unsigned int m_num_elements_sidewall = 4;
const unsigned int m_num_layers_sidewall = 3;
const std::vector<double> m_layer_thickness_sidewall{{0.5e-03, 0.1e-03, 0.5e-03}};
const std::vector<double> m_ply_angle_sidewall{{90, 0, 90}};
const std::vector<int> m_material_id_sidewall{{0, 2, 0}};

const unsigned int m_num_elements_tread = 6;
const unsigned int m_num_layers_tread = 4;
const std::vector<double> m_layer_thickness_tread{{0.1e-02, 0.3e-03, 0.3e-03, 0.5e-03}};
const std::vector<double> m_ply_angle_tread{{0, -20, 20, 90}};
const std::vector<int> m_material_id_tread{{2, 1, 1, 0}};

const int m_div_circumference = 30;

const float m_friction = 0.9f;
const float m_restitution = 0.1f;
const float m_Young = 2.0e6f;
const float m_Poisson = 0.3f;
const float m_kn = 2.0e6f;
const float m_gn = 1.3e1f;
const float m_kt = 1.0e6f;
const float m_gt = 0;

const int m_num_points = 71;
const double m_profile[71][3] = {
    {0.000000E+00, 0.000000E+00, -1.150000E-01}, {1.428571E-02, 1.166670E-02, -1.164180E-01},
    {2.857143E-02, 2.333330E-02, -1.192300E-01}, {4.285714E-02, 3.500000E-02, -1.230200E-01},
    {5.714286E-02, 4.666670E-02, -1.273710E-01}, {7.142857E-02, 5.833330E-02, -1.318700E-01},
    {8.571429E-02, 7.000000E-02, -1.361330E-01}, {1.000000E-01, 8.166670E-02, -1.399910E-01},
    {1.142857E-01, 9.333330E-02, -1.433510E-01}, {1.285714E-01, 1.050000E-01, -1.461240E-01},
    {1.428571E-01, 1.166670E-01, -1.482160E-01}, {1.571429E-01, 1.283330E-01, -1.495390E-01},
    {1.714286E-01, 1.400000E-01, -1.500000E-01}, {1.857143E-01, 1.475000E-01, -1.486380E-01},
    {2.000000E-01, 1.550000E-01, -1.457860E-01}, {2.142857E-01, 1.625000E-01, -1.419760E-01},
    {2.285714E-01, 1.700000E-01, -1.360000E-01}, {2.428571E-01, 1.768970E-01, -1.288420E-01},
    {2.571429E-01, 1.831090E-01, -1.216840E-01}, {2.714286E-01, 1.883940E-01, -1.145260E-01},
    {2.857143E-01, 1.925100E-01, -1.073680E-01}, {3.000000E-01, 1.953230E-01, -1.002110E-01},
    {3.142857E-01, 1.970380E-01, -9.305260E-02}, {3.285714E-01, 1.979260E-01, -8.589470E-02},
    {3.428571E-01, 1.982580E-01, -7.873680E-02}, {3.571429E-01, 1.983020E-01, -7.157890E-02},
    {3.714286E-01, 1.983090E-01, -6.442110E-02}, {3.857143E-01, 1.983540E-01, -5.726320E-02},
    {4.000000E-01, 1.984290E-01, -5.010530E-02}, {4.142857E-01, 1.985240E-01, -4.294740E-02},
    {4.285714E-01, 1.986300E-01, -3.578950E-02}, {4.428571E-01, 1.987380E-01, -2.863160E-02},
    {4.571429E-01, 1.988390E-01, -2.147370E-02}, {4.714286E-01, 1.989220E-01, -1.431580E-02},
    {4.857143E-01, 1.989790E-01, -7.157890E-03}, {5.000000E-01, 1.990000E-01, 0.000000E+00},
    {5.142857E-01, 1.989790E-01, 7.157890E-03},  {5.285714E-01, 1.989220E-01, 1.431580E-02},
    {5.428571E-01, 1.988390E-01, 2.147370E-02},  {5.571429E-01, 1.987380E-01, 2.863160E-02},
    {5.714286E-01, 1.986300E-01, 3.578950E-02},  {5.857143E-01, 1.985240E-01, 4.294740E-02},
    {6.000000E-01, 1.984290E-01, 5.010530E-02},  {6.142857E-01, 1.983540E-01, 5.726320E-02},
    {6.285714E-01, 1.983090E-01, 6.442110E-02},  {6.428571E-01, 1.983020E-01, 7.157890E-02},
    {6.571429E-01, 1.982580E-01, 7.873680E-02},  {6.714286E-01, 1.979260E-01, 8.589470E-02},
    {6.857143E-01, 1.970380E-01, 9.305260E-02},  {7.000000E-01, 1.953230E-01, 1.002110E-01},
    {7.142857E-01, 1.925100E-01, 1.073680E-01},  {7.285714E-01, 1.883940E-01, 1.145260E-01},
    {7.428571E-01, 1.831090E-01, 1.216840E-01},  {7.571429E-01, 1.768970E-01, 1.288420E-01},
    {7.714286E-01, 1.700000E-01, 1.360000E-01},  {7.857143E-01, 1.625000E-01, 1.419760E-01},
    {8.000000E-01, 1.550000E-01, 1.457860E-01},  {8.142857E-01, 1.475000E-01, 1.486380E-01},
    {8.285714E-01, 1.400000E-01, 1.500000E-01},  {8.428571E-01, 1.283330E-01, 1.495390E-01},
    {8.571429E-01, 1.166670E-01, 1.482160E-01},  {8.714286E-01, 1.050000E-01, 1.461240E-01},
    {8.857143E-01, 9.333330E-02, 1.433510E-01},  {9.000000E-01, 8.166670E-02, 1.399910E-01},
    {9.142857E-01, 7.000000E-02, 1.361330E-01},  {9.285714E-01, 5.833330E-02, 1.318700E-01},
    {9.428571E-01, 4.666670E-02, 1.273710E-01},  {9.571429E-01, 3.500000E-02, 1.230200E-01},
    {9.714286E-01, 2.333330E-02, 1.192300E-01},  {9.857143E-01, 1.166670E-02, 1.164180E-01},
    {1.000000E+00, 0.000000E+00, 1.150000E-01}};


// -----------------------------------------------------------------------------

// Initial Position of wheel
ChVector<> wheel1_IniPos(-bxDim / 2 + 1.5 * wheel_radius, -dis_Left_right / 2, 1.5 * wheel_radius + bzDim);
ChVector<> wheel2_IniPos(-bxDim / 2 + 1.5 * wheel_radius + dis_Front_back, -dis_Left_right / 2, 1.5 * wheel_radius + bzDim);
ChVector<> wheel3_IniPos(-bxDim / 2 + 1.5 * wheel_radius,  dis_Left_right / 2, 1.5 * wheel_radius + bzDim);
ChVector<> wheel4_IniPos(-bxDim / 2 + 1.5 * wheel_radius + dis_Front_back,  dis_Left_right / 2, 1.5 * wheel_radius + bzDim);
ChVector<> wheel_IniVel(0.0, 0.0, 0.0);

// Simulation time and stepsize
double t_end = 20.0;
double dT = 5e-4;

// Output frequency
bool output = true;
double out_fps = 20;

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = true;
float render_fps = 100;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
auto motor1 = chrono_types::make_shared<ChLinkMotorRotationAngle>();
auto motor2 = chrono_types::make_shared<ChLinkMotorRotationAngle>();
auto motor3 = chrono_types::make_shared<ChLinkMotorRotationAngle>();
auto motor4 = chrono_types::make_shared<ChLinkMotorRotationAngle>();

std::vector<std::vector<int>> NodeNeighborElement_mesh;

void Create_MB_FE(ChSystemSMC& sysMBS, ChSystemFsi& sysFSI);

int main(int argc, char* argv[]) {
    // Create oputput directories
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cerr << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(out_dir + "/particles"))) {
        std::cerr << "Error creating directory " << out_dir + "/particles" << std::endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(out_dir + "/vtk"))) {
        std::cerr << "Error creating directory " << out_dir + "/vtk" << std::endl;
        return 1;
    }

    // Create a physics system and an FSI system
    ChSystemSMC sysMBS;
    ChSystemFsi sysFSI(sysMBS);

    // Use the default input file or you may enter your input parameters as a command line argument
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_Flexible_Elements_Explicit.json");
    if (argc == 1) {
        std::cout << "Use the default JSON file" << std::endl;
    } else if (argc == 2) {
        std::cout << "Use the specified JSON file" << std::endl;
        std::string my_inputJson = std::string(argv[1]);
        inputJson = my_inputJson;
    } else {
        std::cout << "usage: ./demo_FSI_Flexible_Elements <json_file>" << std::endl;
        return 1;
    }
    sysFSI.ReadParametersFromFile(inputJson);

    sysFSI.SetContainerDim(ChVector<>(bxDim, byDim, bzDim));

    auto initSpace0 = sysFSI.GetInitialSpacing();
    ChVector<> cMin = ChVector<>(-5 * bxDim, -byDim / 2.0 - initSpace0 / 2.0, -5 * bzDim );
    ChVector<> cMax = ChVector<>( 5 * bxDim,  byDim / 2.0 + initSpace0 / 2.0,  10 * bzDim );
    sysFSI.SetBoundaries(cMin, cMax);

    // Setup the output directory for FSI data
    sysFSI.SetOutputDirectory(out_dir);

    // Create SPH particles of fluid region
    chrono::utils::GridSampler<> sampler(initSpace0);
    ChVector<> boxCenter(-bxDim / 2 + fxDim / 2, 0, fzDim / 2 + 1 * initSpace0);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);
    chrono::utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);
    size_t numPart = points.size();
    for (int i = 0; i < numPart; i++) {
        sysFSI.AddSPHParticle(points[i]);
    }

    // Create solids
    Create_MB_FE(sysMBS, sysFSI);
    sysFSI.Initialize();
    auto my_mesh = sysFSI.GetFsiMesh();

    // Create a run-tme visualizer
    ChVisualizationFsi fsi_vis(&sysFSI);
    if (render) {
        fsi_vis.SetTitle("Chrono::FSI flexible element demo");
        fsi_vis.SetCameraPosition(ChVector<>(bxDim / 8, -3, 0.25), ChVector<>(bxDim / 8, 0.0, 0.25));
        fsi_vis.SetCameraMoveScale(1.0f);
        fsi_vis.EnableBoundaryMarkers(false);
        fsi_vis.Initialize();
    }

    // Set MBS solver
#ifdef CHRONO_PARDISO_MKL
    auto mkl_solver = chrono_types::make_shared<ChSolverPardisoMKL>();
    mkl_solver->LockSparsityPattern(true);
    sysMBS.SetSolver(mkl_solver);
#else
    auto solver = chrono_types::make_shared<ChSolverMINRES>();
    sysMBS.SetSolver(solver);
    solver->SetMaxIterations(2000);
    solver->SetTolerance(1e-10);
    solver->EnableDiagonalPreconditioner(true);
    solver->SetVerbose(false);
    sysMBS.SetSolverForceTolerance(1e-10);
#endif

    // Simulation loop
    double dT = sysFSI.GetStepSize();

    unsigned int output_steps = (unsigned int)(1 / (out_fps * dT));
    unsigned int render_steps = (unsigned int)(1 / (render_fps * dT));

    double time = 0.0;
    int current_step = 0;

    ChTimer<> timer;
    timer.start();
    while (time < t_end) {
        std::cout << current_step << " time: " << time << std::endl;

        if (output && current_step % output_steps == 0) {
            std::cout << "-------- Output" << std::endl;
            sysFSI.PrintParticleToFile(out_dir + "/particles");
            static int counter = 0;
            std::string filename = out_dir + "/vtk/flex_body." + std::to_string(counter) + ".vtk";
            fea::ChMeshExporter::writeFrame(my_mesh, (char*)filename.c_str(), MESH_CONNECTIVITY);

            // save the chassis to obj files
            auto body = sysMBS.Get_bodylist()[5];
            ChVector<> body_pos = body->GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body->GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "../data/vehicle/hmmwv/hmmwv_chassis.obj";
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 
            mmesh->Transform(body_pos + ChVector<>(0, 0, 0.1), ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

            std::string chassisname = out_dir + "/vtk/chassis." + std::to_string(counter++) + ".vtk";
            std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };

            std::ofstream file;
            file.open(chassisname);
            file << "# vtk DataFile Version 2.0" << std::endl;
            file << "VTK from simulation" << std::endl;
            file << "ASCII" << std::endl;
            file << "DATASET UNSTRUCTURED_GRID" << std::endl;
            int nv = mmesh->getCoordsVertices().size();
            file << "POINTS " << nv << " " << "float" << std::endl;
            for (auto& v : mmesh->getCoordsVertices()) { 
                file << v.x() << " " << v.y() << " " << v.z() << std::endl;
            }
            int nf = mmesh->getIndicesVertexes().size();
            file << "CELLS " << nf << " " << 4*nf << std::endl;
            for (auto& f : mmesh->getIndicesVertexes()) {
                file <<  "3 " << f.x()  << " " << f.y() << " " << f.z()  << std::endl; 
            }
            file << "CELL_TYPES " << nf << std::endl;
            for (auto& f : mmesh->getIndicesVertexes()) {
                file <<  "5 " << std::endl; 
            }
            file.close();
        }

        // Render SPH particles
        if (render && current_step % render_steps == 0) {
            if (!fsi_vis.Render())
                break;
        }

        sysFSI.DoStepDynamics_FSI();

        time += dT;
        current_step++;
    }
    timer.stop();
    std::cout << "\nSimulation time: " << timer() << " seconds\n" << std::endl;

    return 0;
}

//--------------------------------------------------------------------
// Create the objects of the MBD system. Rigid/flexible bodies, and if 
// fsi, their bce representation are created and added to the systems
void Create_MB_FE(ChSystemSMC& sysMBS, ChSystemFsi& sysFSI) {
    sysMBS.Set_G_acc(ChVector<>(0, 0, -9.81));
    sysFSI.Set_G_acc(ChVector<>(0, 0, -9.81));
    
    // Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mysurfmaterial->SetYoungModulus(6e4);
    mysurfmaterial->SetFriction(0.3f);
    mysurfmaterial->SetRestitution(0.2f);
    mysurfmaterial->SetAdhesion(0);

    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetIdentifier(-1);
    ground->SetBodyFixed(true);
    ground->SetCollide(true);

    ground->GetCollisionModel()->ClearModel();
    auto initSpace0 = sysFSI.GetInitialSpacing();

    // Bottom and top wall
    ChVector<> size_XY(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> pos_zp(0, 0, bzDim + 2 * initSpace0);
    ChVector<> pos_zn(0, 0, -2 * initSpace0);

    // left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 1 * initSpace0);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 1 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 1 * initSpace0);

    // MBD representation of walls
    chrono::utils::AddBoxGeometry(ground.get(), mysurfmaterial, size_XY, pos_zn, QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), mysurfmaterial, size_YZ, pos_xp, QUNIT, true);
    chrono::utils::AddBoxGeometry(ground.get(), mysurfmaterial, size_YZ, pos_xn, QUNIT, true);
    // chrono::utils::AddBoxGeometry(ground.get(), mysurfmaterial, size_XZ, pos_yp, QUNIT, true);
    // chrono::utils::AddBoxGeometry(ground.get(), mysurfmaterial, size_XZ, pos_yn, QUNIT, true);
    sysMBS.AddBody(ground);

    // Fluid representation of walls
    sysFSI.AddBoxBCE(ground, pos_zn, QUNIT, size_XY, 12);
    // sysFSI.AddBoxBCE(ground, pos_zp, QUNIT, size_XY, 12);
    sysFSI.AddBoxBCE(ground, pos_xp, QUNIT, size_YZ, 23);
    sysFSI.AddBoxBCE(ground, pos_xn, QUNIT, size_YZ, 23);
    // sysFSI.AddBoxBCE(ground, pos_yp, QUNIT, size_XZ, 13);
    // sysFSI.AddBoxBCE(ground, pos_yn, QUNIT, size_XZ, 13);

    // ******************************* Rigid bodies ***********************************
    // set the abs orientation, position and velocity
    auto wheel1 = chrono_types::make_shared<ChBodyAuxRef>();
    auto wheel2 = chrono_types::make_shared<ChBodyAuxRef>();
    auto wheel3 = chrono_types::make_shared<ChBodyAuxRef>();
    auto wheel4 = chrono_types::make_shared<ChBodyAuxRef>();
    ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 0));
    ChVector<> Body_vel = wheel_IniVel;
    ChVector<> Body_pos1 = wheel1_IniPos;
    ChVector<> Body_pos2 = wheel2_IniPos;
    ChVector<> Body_pos3 = wheel3_IniPos;
    ChVector<> Body_pos4 = wheel4_IniPos;

    // Set inertia
    wheel1->SetMass(total_mass * 1.0 / 2.0);
    wheel1->SetInertiaXX(ChVector<>(60, 60, 60));
    wheel1->SetPos_dt(Body_vel);
    wheel1->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0));  // set an initial anular velocity (rad/s)

    wheel2->SetMass(total_mass * 1.0 / 2.0);
    wheel2->SetInertiaXX(ChVector<>(60, 60, 60));
    wheel2->SetPos_dt(Body_vel);
    wheel2->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0));  // set an initial anular velocity (rad/s)

    wheel3->SetMass(total_mass * 1.0 / 2.0);
    wheel3->SetInertiaXX(ChVector<>(60, 60, 60));
    wheel3->SetPos_dt(Body_vel);
    wheel3->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0));  // set an initial anular velocity (rad/s)

    wheel4->SetMass(total_mass * 1.0 / 2.0);
    wheel4->SetInertiaXX(ChVector<>(60, 60, 60));
    wheel4->SetPos_dt(Body_vel);
    wheel4->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0));  // set an initial anular velocity (rad/s)

    // Set the absolute position of the body:
    wheel1->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos1), ChQuaternion<>(Body_rot)));
    wheel1->SetBodyFixed(false);
    wheel1->SetCollide(false);
    sysMBS.AddBody(wheel1);

    wheel2->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos2), ChQuaternion<>(Body_rot)));
    wheel2->SetBodyFixed(false);
    wheel2->SetCollide(false);
    sysMBS.AddBody(wheel2);

    wheel3->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos3), ChQuaternion<>(Body_rot)));
    wheel3->SetBodyFixed(false);
    wheel3->SetCollide(false);
    sysMBS.AddBody(wheel3);

    wheel4->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos4), ChQuaternion<>(Body_rot)));
    wheel4->SetBodyFixed(false);
    wheel4->SetCollide(false);
    sysMBS.AddBody(wheel4);
    // Create the chassis 
    auto chassis = chrono_types::make_shared<ChBody>();
    chassis->SetMass(total_mass * 1.0 / 2.0);
    chassis->SetPos(0.25 * (wheel1->GetPos() + wheel2->GetPos() + wheel3->GetPos() + wheel4->GetPos()));
    chassis->SetCollide(false);
    chassis->SetBodyFixed(false);
    sysMBS.AddBody(chassis);

    // // Create the axle
    // auto axle = chrono_types::make_shared<ChBody>();
    // axle->SetMass(total_mass * 1.0 / 2.0);
    // axle->SetPos(wheel->GetPos());
    // axle->SetCollide(false);
    // axle->SetBodyFixed(false);
    // sysMBS.AddBody(axle);

    // Connect the chassis to the ground through a translational joint and create a linear actuator.
    // auto prismatic1 = chrono_types::make_shared<ChLinkLockPrismatic>();
    // prismatic1->Initialize(ground, chassis, ChCoordsys<>(chassis->GetPos(), Q_from_AngY(CH_C_PI_2)));
    // prismatic1->SetName("prismatic_chassis_ground");
    // sysMBS.AddLink(prismatic1);

    // double velocity = wheel_AngVel * wheel_radius * (1.0 - wheel_slip);
    // auto actuator_fun = chrono_types::make_shared<ChFunction_Ramp>(0.0, velocity);

    // actuator->Initialize(ground, chassis, false, ChCoordsys<>(chassis->GetPos(), QUNIT),
    //     ChCoordsys<>(chassis->GetPos() + ChVector<>(1, 0, 0), QUNIT));
    // actuator->SetName("actuator");
    // actuator->SetDistanceOffset(1);
    // actuator->SetActuatorFunction(actuator_fun);
    // sysMBS.AddLink(actuator);

    // // Connect the axle to the chassis through a vertical translational joint.
    // auto prismatic2 = chrono_types::make_shared<ChLinkLockPrismatic>();
    // prismatic2->Initialize(chassis, axle, ChCoordsys<>(chassis->GetPos(), QUNIT));
    // prismatic2->SetName("prismatic_axle_chassis");
    // sysMBS.AddLink(prismatic2);

    // // Connect the wheel to the axle through a engine joint.
    // motor->SetName("engine_wheel_axle");
    // motor->Initialize(wheel, axle, ChFrame<>(wheel->GetPos(), 
    //     chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    // motor->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    // sysMBS.AddLink(motor);

    // Connect the wheel to the chassis through a engine joint.
    motor1->SetName("engine_wheel_chassis");
    motor1->Initialize(wheel1, chassis, ChFrame<>(wheel1->GetPos(), 
        chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor1->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    sysMBS.AddLink(motor1);

    motor2->SetName("engine_wheel_chassis");
    motor2->Initialize(wheel2, chassis, ChFrame<>(wheel2->GetPos(), 
        chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor2->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    sysMBS.AddLink(motor2);

    motor3->SetName("engine_wheel_chassis");
    motor3->Initialize(wheel3, chassis, ChFrame<>(wheel3->GetPos(), 
        chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor3->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    sysMBS.AddLink(motor3);

    motor4->SetName("engine_wheel_chassis");
    motor4->Initialize(wheel4, chassis, ChFrame<>(wheel4->GetPos(), 
        chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor4->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    sysMBS.AddLink(motor4);

    // ******************************* Flexible bodies ***********************************
    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = chrono_types::make_shared<fea::ChMesh>();
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    std::vector<std::vector<int>> _2D_elementsNodes_mesh;

    // Add the tire
    {
        auto mat = chrono_types::make_shared<ChMaterialShellANCF>(2000, 2.0e7, 0.3);

        int m_div_width = 2 * (m_num_elements_bead + m_num_elements_sidewall + m_num_elements_tread);
        
        // Set the profile
        std::vector<double> m_profile_t;
        std::vector<double> m_profile_x;
        std::vector<double> m_profile_y;
        m_profile_t.resize(m_num_points);
        m_profile_x.resize(m_num_points);
        m_profile_y.resize(m_num_points);
        for ( int i = 0; i < m_num_points; i++) {
            m_profile_t[i] = m_profile[i][0];
            m_profile_x[i] = m_profile[i][1];
            m_profile_y[i] = m_profile[i][2];
        }

        //   x - radial direction
        //   y - transversal direction
        ChCubicSpline splineX(m_profile_t, m_profile_x);
        ChCubicSpline splineY(m_profile_t, m_profile_y);

        // Create the mesh nodes.
        // The nodes are first created in the wheel local frame, assuming Y as the tire axis,
        // and are then transformed to the global frame.
        //first tire
        for (int i = 0; i < m_div_circumference; i++) {
            double phi = (CH_C_2PI * i) / m_div_circumference;
            ChVector<> nrm(-std::sin(phi), 0, std::cos(phi));

            for (int j = 0; j <= m_div_width; j++) {
                double t_prf = double(j) / m_div_width;
                double x_prf, xp_prf, xpp_prf;
                double y_prf, yp_prf, ypp_prf;
                splineX.Evaluate(t_prf, x_prf, xp_prf, xpp_prf);
                splineY.Evaluate(t_prf, y_prf, yp_prf, ypp_prf);

                // Node position with respect to rim center
                double x = (m_rim_radius + x_prf) * std::cos(phi) + wheel1_IniPos.x();
                double y = y_prf+ wheel1_IniPos.y();
                double z = (m_rim_radius + x_prf) * std::sin(phi)+ wheel1_IniPos.z();
                // Node position in global frame (actual coordinate values)
                ChVector<> loc = ChVector<>(x, y, z);

                // Node direction
                ChVector<> tan_prf(std::cos(phi) * xp_prf, yp_prf, std::sin(phi) * xp_prf);
                ChVector<> nrm_prf = Vcross(tan_prf, nrm).GetNormalized();
                ChVector<> dir = nrm_prf;

                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(loc, dir);
                node->SetMass(0);
                my_mesh->AddNode(node);
                
                // Fix the edge node on the wheel
                if (j == 0 || j == m_div_width){
                    auto mlink1 = chrono_types::make_shared<ChLinkPointFrame>();
                    mlink1->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node), wheel1);
                    sysMBS.Add(mlink1);
                }
            }
        }
        // second tire
        for (int i = 0; i < m_div_circumference; i++) {
            double phi = (CH_C_2PI * i) / m_div_circumference;
            ChVector<> nrm(-std::sin(phi), 0, std::cos(phi));

            for (int j = 0; j <= m_div_width; j++) {
                double t_prf = double(j) / m_div_width;
                double x_prf, xp_prf, xpp_prf;
                double y_prf, yp_prf, ypp_prf;
                splineX.Evaluate(t_prf, x_prf, xp_prf, xpp_prf);
                splineY.Evaluate(t_prf, y_prf, yp_prf, ypp_prf);

                // Node position with respect to rim center
                double x = (m_rim_radius + x_prf) * std::cos(phi) + wheel2_IniPos.x();
                double y = y_prf+ wheel2_IniPos.y();
                double z = (m_rim_radius + x_prf) * std::sin(phi)+ wheel2_IniPos.z();
                // Node position in global frame (actual coordinate values)
                ChVector<> loc = ChVector<>(x, y, z);

                // Node direction
                ChVector<> tan_prf(std::cos(phi) * xp_prf, yp_prf, std::sin(phi) * xp_prf);
                ChVector<> nrm_prf = Vcross(tan_prf, nrm).GetNormalized();
                ChVector<> dir = nrm_prf;

                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(loc, dir);
                node->SetMass(0);
                my_mesh->AddNode(node);
                
                // Fix the edge node on the wheel
                if (j == 0 || j == m_div_width){
                    auto mlink2 = chrono_types::make_shared<ChLinkPointFrame>();
                    mlink2->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node), wheel2);
                    sysMBS.Add(mlink2);
                }
            }
        }
        // third tire
        for (int i = 0; i < m_div_circumference; i++) {
            double phi = (CH_C_2PI * i) / m_div_circumference;
            ChVector<> nrm(-std::sin(phi), 0, std::cos(phi));

            for (int j = 0; j <= m_div_width; j++) {
                double t_prf = double(j) / m_div_width;
                double x_prf, xp_prf, xpp_prf;
                double y_prf, yp_prf, ypp_prf;
                splineX.Evaluate(t_prf, x_prf, xp_prf, xpp_prf);
                splineY.Evaluate(t_prf, y_prf, yp_prf, ypp_prf);

                // Node position with respect to rim center
                double x = (m_rim_radius + x_prf) * std::cos(phi) + wheel3_IniPos.x();
                double y = y_prf+ wheel3_IniPos.y();
                double z = (m_rim_radius + x_prf) * std::sin(phi)+ wheel3_IniPos.z();
                // Node position in global frame (actual coordinate values)
                ChVector<> loc = ChVector<>(x, y, z);

                // Node direction
                ChVector<> tan_prf(std::cos(phi) * xp_prf, yp_prf, std::sin(phi) * xp_prf);
                ChVector<> nrm_prf = Vcross(tan_prf, nrm).GetNormalized();
                ChVector<> dir = nrm_prf;

                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(loc, dir);
                node->SetMass(0);
                my_mesh->AddNode(node);
                
                // Fix the edge node on the wheel
                if (j == 0 || j == m_div_width){
                    auto mlink3 = chrono_types::make_shared<ChLinkPointFrame>();
                    mlink3->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node), wheel3);
                    sysMBS.Add(mlink3);
                }
            }
        }
        // fourth tire
        for (int i = 0; i < m_div_circumference; i++) {
            double phi = (CH_C_2PI * i) / m_div_circumference;
            ChVector<> nrm(-std::sin(phi), 0, std::cos(phi));

            for (int j = 0; j <= m_div_width; j++) {
                double t_prf = double(j) / m_div_width;
                double x_prf, xp_prf, xpp_prf;
                double y_prf, yp_prf, ypp_prf;
                splineX.Evaluate(t_prf, x_prf, xp_prf, xpp_prf);
                splineY.Evaluate(t_prf, y_prf, yp_prf, ypp_prf);

                // Node position with respect to rim center
                double x = (m_rim_radius + x_prf) * std::cos(phi) + wheel4_IniPos.x();
                double y = y_prf+ wheel4_IniPos.y();
                double z = (m_rim_radius + x_prf) * std::sin(phi)+ wheel4_IniPos.z();
                // Node position in global frame (actual coordinate values)
                ChVector<> loc = ChVector<>(x, y, z);

                // Node direction
                ChVector<> tan_prf(std::cos(phi) * xp_prf, yp_prf, std::sin(phi) * xp_prf);
                ChVector<> nrm_prf = Vcross(tan_prf, nrm).GetNormalized();
                ChVector<> dir = nrm_prf;

                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(loc, dir);
                node->SetMass(0);
                my_mesh->AddNode(node);
                
                // Fix the edge node on the wheel
                if (j == 0 || j == m_div_width){
                    auto mlink4 = chrono_types::make_shared<ChLinkPointFrame>();
                    mlink4->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node), wheel4);
                    sysMBS.Add(mlink4);
                }
            }
        }
    
        int TotalNumElements = m_div_circumference * m_div_width * 4; //my_mesh->GetNelements();
        int TotalNumNodes = my_mesh->GetNnodes();
        int NumNodes_OneTire = TotalNumNodes / 4;

        _2D_elementsNodes_mesh.resize(TotalNumElements);
        NodeNeighborElement_mesh.resize(TotalNumNodes);


        // Create the ANCF shell elements
        int num_elem = 0;
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < m_div_circumference; i++) {
                for (int j = 0; j < m_div_width; j++) {
                    // Adjacent nodes
                    int inode0, inode1, inode2, inode3;
                    inode1 = j + i * (m_div_width + 1) + k * NumNodes_OneTire;
                    inode2 = j + 1 + i * (m_div_width + 1) + k * NumNodes_OneTire;
                    if (i == m_div_circumference - 1) {
                        inode0 = j + k * NumNodes_OneTire;
                        inode3 = j + 1 + k * NumNodes_OneTire;
                    } else {
                        inode0 = j + (i + 1) * (m_div_width + 1) + k * NumNodes_OneTire;
                        inode3 = j + 1 + (i + 1) * (m_div_width + 1) + k * NumNodes_OneTire;
                    }

                    auto node0 = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(inode0));
                    auto node1 = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(inode1));
                    auto node2 = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(inode2));
                    auto node3 = std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(inode3));

                    _2D_elementsNodes_mesh[num_elem].push_back(inode0);
                    _2D_elementsNodes_mesh[num_elem].push_back(inode1);
                    _2D_elementsNodes_mesh[num_elem].push_back(inode2);
                    _2D_elementsNodes_mesh[num_elem].push_back(inode3);
                    NodeNeighborElement_mesh[inode0].push_back(num_elem);
                    NodeNeighborElement_mesh[inode1].push_back(num_elem);
                    NodeNeighborElement_mesh[inode2].push_back(num_elem);
                    NodeNeighborElement_mesh[inode3].push_back(num_elem);

                    // Create the element and set its nodes.
                    auto element = chrono_types::make_shared<ChElementShellANCF_3423>();
                    element->SetNodes(node0, node1, node2, node3);

                    // Element dimensions
                    double len_circumference =
                        0.5 * ((node1->GetPos() - node0->GetPos()).Length() + (node3->GetPos() - node2->GetPos()).Length());
                    double len_width = 
                        0.5 * ((node2->GetPos() - node1->GetPos()).Length() + (node3->GetPos() - node0->GetPos()).Length());

                    element->SetDimensions(len_circumference, len_width);

                    std::cout << len_circumference  << " SetDimensions: " << len_width << std::endl;

                    element->AddLayer(0.03, 0 , mat);

                    // Set other element properties
                    element->SetAlphaDamp(m_alpha);

                    // Add element to mesh
                    my_mesh->AddElement(element);

                    ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                                element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
                    std::cout << "Adding element" << num_elem << "  with center:  " << center.x() << " " << center.y()
                            << " " << center.z() << std::endl;

                    num_elem++;
                }
            }
        }
    }

    // Add the mesh to the system
    sysMBS.Add(my_mesh);

    // fluid representation of flexible bodies
    bool multilayer = true;
    bool removeMiddleLayer = true;
    bool add1DElem = flexible_elem_1D;
    bool add2DElem = !flexible_elem_1D;
    sysFSI.AddFEAmeshBCE(my_mesh, NodeNeighborElement_mesh, _1D_elementsNodes_mesh, 
        _2D_elementsNodes_mesh, add1DElem, add2DElem, multilayer, removeMiddleLayer, 0, 0);

    if (flexible_elem_1D)
        sysFSI.SetCableElementsNodes(_1D_elementsNodes_mesh);
    else
        sysFSI.SetShellElementsNodes(_2D_elementsNodes_mesh);

    sysFSI.SetFsiMesh(my_mesh);
    fea::ChMeshExporter::writeMesh(my_mesh, MESH_CONNECTIVITY);
}
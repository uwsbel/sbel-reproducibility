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
// Author: Wei Hu, Radu Serban
// =============================================================================

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "chrono/physics/ChSystemSMC.h"

#ifdef CHRONO_PARDISO_MKL
    #include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#endif

#include "chrono/ChConfig.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/core/ChTimer.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"

#include "chrono_thirdparty/filesystem/path.h"

#include "chrono/solver/ChIterativeSolverLS.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/fea/ChElementShellANCF_3423.h"
#include "chrono/fea/ChLinkDirFrame.h"
#include "chrono/fea/ChLinkPointFrame.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshExporter.h"
#include "chrono/fea/ChBuilderBeam.h"


// Chrono namespaces
using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fea;
using namespace chrono::geometry;
using namespace chrono::collision;

// Physical properties of terrain particles
double iniSpacing = 0.02;
double kernelLength = 0.02;
double density = 1700.0;

// Dimension of the terrain container
double smalldis = 1.0e-9;
double bxDim = 5.0 + smalldis;
double byDim = 0.8 + smalldis;
double bzDim = 0.2 + smalldis;

// Size of the wheel
double wheel_radius = 0.47;
double wheel_slip = 0.0;
double wheel_AngVel = 1.0;
double total_mass = 105.22;
std::string wheel_obj = "vehicle/hmmwv/hmmwv_tire_coarse_closed.obj";

double m_rim_radius = 0.35;
double m_height = 0.195;
double m_thickness = 0.014;
int m_div_circumference = 30;
int m_div_width = 6;
double m_alpha = 0.15;

bool flexible_elem_1D = false;

// Initial Position of wheel
ChVector<> wheel_IniPos(-bxDim / 2 + 1.5 * wheel_radius, 0.0, 1.5 * wheel_radius + bzDim);
ChVector<> Tire_IniPos(bxDim / 2 - 1.5 * wheel_radius, 0.0, 1.5 * wheel_radius + bzDim);
ChVector<> wheel_IniVel(0.0, 0.0, 0.0);

// Simulation time and stepsize
double total_time = 5.0;
double dT = 5e-5;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();

std::vector<std::vector<int>> NodeNeighborElement_mesh;

// Save data as csv files to see the results off-line using Paraview
bool output = true;
int out_fps = 20;

// Output directories and settings
const std::string out_dir = GetChronoOutputPath() + "FSI_Single_Wheel_With_Tire/";
std::string MESH_CONNECTIVITY = out_dir + "Flex_MESH.vtk";

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = true;
float render_fps = 100;

// Verbose terminal output
bool verbose_fsi = true;
bool verbose = true;

//------------------------------------------------------------------
// Function to save wheel to Paraview VTK files
//------------------------------------------------------------------
void WriteWheelVTK(const std::string& filename,
                   ChTriangleMeshConnected& mesh,
                   const ChFrame<>& frame) {
    std::ofstream outf;
    outf.open(filename);
    outf << "# vtk DataFile Version 2.0" << std::endl;
    outf << "VTK from simulation" << std::endl;
    outf << "ASCII" << std::endl;
    outf << "DATASET UNSTRUCTURED_GRID" << std::endl;
    outf << "POINTS " << mesh.getCoordsVertices().size() << " "
         << "float" << std::endl;
    for (auto& v : mesh.getCoordsVertices()) {
        auto w = frame.TransformPointLocalToParent(v);
        outf << w.x() << " " << w.y() << " " << w.z() << std::endl;
    }
    auto nf = mesh.getIndicesVertexes().size();
    outf << "CELLS " << nf << " " << 4 * nf << std::endl;
    for (auto& f : mesh.getIndicesVertexes()) {
        outf << "3 " << f.x() << " " << f.y() << " " << f.z() << std::endl;
    }
    outf << "CELL_TYPES " << nf << std::endl;
    for (int i = 0; i < nf; i++) {
        outf << "5 " << std::endl;
    }
    outf.close();
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if FSI,
// their BCE representation are created and added to the systems
//------------------------------------------------------------------
void CreateSolidPhase(ChSystemSMC& sysMBS, ChSystemFsi& sysFSI) {
    // Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mysurfmaterial->SetYoungModulus(1e8);
    mysurfmaterial->SetFriction(0.9f);
    mysurfmaterial->SetRestitution(0.4f);
    mysurfmaterial->SetAdhesion(0);

    // Create a container -- always FIRST body in the system
    auto ground = chrono_types::make_shared<ChBodyEasyBox>(100, 100, 0.02, 1000, false, true, mysurfmaterial);
    ground->SetPos(ChVector<>(0.0, 0.0, 0.0));
    ground->SetCollide(true);
    ground->SetBodyFixed(true);
    sysMBS.AddBody(ground);

    // Bottom wall
    ChVector<> size_XY(bxDim / 2 + 3 * iniSpacing, byDim / 2 + 0 * iniSpacing, 2 * iniSpacing);
    ChVector<> pos_zn(0, 0, -3 * iniSpacing);
    ChVector<> pos_zp(0, 0, bzDim + 2 * iniSpacing);

    // Left and right Wall
    ChVector<> size_YZ(2 * iniSpacing, byDim / 2 + 0 * iniSpacing, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + iniSpacing, 0.0, bzDim / 2 + 0 * iniSpacing);
    ChVector<> pos_xn(-bxDim / 2 - 3 * iniSpacing, 0.0, bzDim / 2 + 0 * iniSpacing);

    // Front and back Wall
    ChVector<> size_XZ(bxDim / 2 + 3 * iniSpacing, 2 * iniSpacing, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + iniSpacing, bzDim / 2 + 0 * iniSpacing);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * iniSpacing, bzDim / 2 + 0 * iniSpacing);

    // Add BCE particles attached on the walls into FSI system
    // sysFSI.AddBoxBCE(ground, pos_zp, QUNIT, size_XY, 12);
    sysFSI.AddBoxBCE(ground, pos_zn, QUNIT, size_XY, 12);
    sysFSI.AddBoxBCE(ground, pos_xp, QUNIT, size_YZ, 23);
    sysFSI.AddBoxBCE(ground, pos_xn, QUNIT, size_YZ, 23);
    ////sysFSI.AddBoxBCE(ground, pos_yp, QUNIT, size_XZ, 13);
    ////sysFSI.AddBoxBCE(ground, pos_yn, QUNIT, size_XZ, 13);

    // ==================== Rigid body system ====================

    // Create the wheel -- always SECOND body in the system
    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    double scale_ratio = 1.0;
    mmesh->LoadWavefrontMesh(GetChronoDataFile(wheel_obj), false, true);
    ////mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(body_rot));       // rotate the mesh if needed
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));  // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                              // if meshes are not watertight

    // compute mass inertia from mesh
    double mmass;
    double mdensity = 1500.0;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);
    mcog = ChVector<>(0.0, 0.0, 0.0);

    // set the abs orientation, position and velocity
    auto wheel = chrono_types::make_shared<ChBodyAuxRef>();
    ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 0));
    ChVector<> Body_pos = wheel_IniPos;
    ChVector<> Body_vel = wheel_IniVel;

    // Set the COG coordinates to barycenter, without displacing the REF reference.
    // Make the COG frame a principal frame.
    wheel->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    wheel->SetMass(total_mass * 1.0 / 2.0);
    wheel->SetInertiaXX(mdensity * principal_I);
    wheel->SetPos_dt(Body_vel);
    wheel->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0));  // set an initial anular velocity (rad/s)

    // Set the absolute position of the body:
    wheel->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos), ChQuaternion<>(Body_rot)));
    sysMBS.AddBody(wheel);

    wheel->SetBodyFixed(false);
    wheel->GetCollisionModel()->ClearModel();
    wheel->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh, false, false, VNULL, ChMatrix33<>(1), 0.005);
    wheel->GetCollisionModel()->BuildModel();
    wheel->SetCollide(true);

    // Add this body to the FSI system
    std::vector<ChVector<>> BCE_par_rock;
    sysFSI.CreateMeshPoints(mmesh, iniSpacing, BCE_par_rock);
    sysFSI.AddPointsBCE(wheel, BCE_par_rock, ChVector<>(0.0), QUNIT);
    sysFSI.AddFsiBody(wheel);

    // Create the chassis -- always THIRD body in the system
    // Initially, the chassis is fixed to ground.
    // It is released after the settling phase.
    auto chassis = chrono_types::make_shared<ChBody>();
    // chassis->SetIdentifier(Id_chassis);
    chassis->SetMass(total_mass * 1.0 / 2.0);
    chassis->SetPos(wheel->GetPos());
    chassis->SetCollide(false);
    chassis->SetBodyFixed(false);

    // Add geometry of the chassis.
    chassis->GetCollisionModel()->ClearModel();
    chrono::utils::AddBoxGeometry(chassis.get(), mysurfmaterial, ChVector<>(0.1, 0.1, 0.1), ChVector<>(0, 0, 0));
    chassis->GetCollisionModel()->BuildModel();
    sysMBS.AddBody(chassis);

    // Create the axle -- always FOURTH body in the system
    auto axle = chrono_types::make_shared<ChBody>();
    // axle->SetIdentifier(Id_axle);
    axle->SetMass(total_mass * 1.0 / 2.0);
    axle->SetPos(wheel->GetPos());
    axle->SetCollide(false);
    axle->SetBodyFixed(false);
    // Add geometry of the axle.
    axle->GetCollisionModel()->ClearModel();
    chrono::utils::AddSphereGeometry(axle.get(), mysurfmaterial, 0.5, ChVector<>(0, 0, 0));
    axle->GetCollisionModel()->BuildModel();
    sysMBS.AddBody(axle);

    // Connect the chassis to the containing bin (ground) through a translational joint and create a linear actuator.
    auto prismatic1 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic1->Initialize(ground, chassis, ChCoordsys<>(chassis->GetPos(), Q_from_AngY(CH_C_PI_2)));
    prismatic1->SetName("prismatic_chassis_ground");
    sysMBS.AddLink(prismatic1);

    double velocity = wheel_AngVel * wheel_radius * (1.0 - wheel_slip);
    auto actuator_fun = chrono_types::make_shared<ChFunction_Ramp>(0.0, velocity);

    actuator->Initialize(ground, chassis, false, ChCoordsys<>(chassis->GetPos(), QUNIT),
                         ChCoordsys<>(chassis->GetPos() + ChVector<>(1, 0, 0), QUNIT));
    actuator->SetName("actuator");
    actuator->SetDistanceOffset(1);
    actuator->SetActuatorFunction(actuator_fun);
    sysMBS.AddLink(actuator);

    // Connect the axle to the chassis through a vertical translational joint.
    auto prismatic2 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic2->Initialize(chassis, axle, ChCoordsys<>(chassis->GetPos(), QUNIT));
    prismatic2->SetName("prismatic_axle_chassis");
    sysMBS.AddLink(prismatic2);

    // Connect the wheel to the axle through a engine joint.
    motor->SetName("engine_wheel_axle");
    motor->Initialize(wheel, axle, ChFrame<>(wheel->GetPos(), 
        chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, wheel_AngVel));
    sysMBS.AddLink(motor);

    // ==================== Flexible body system ====================

    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = chrono_types::make_shared<fea::ChMesh>();
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    std::vector<std::vector<int>> _2D_elementsNodes_mesh;

    // Add the tire
    {
        auto mat = chrono_types::make_shared<ChMaterialShellANCF>(2000, 1.0e7, 0.3);

        // Create the mesh nodes.
        // The nodes are first created in the wheel local frame, assuming Y as the tire axis,
        // and are then transformed to the global frame.
        for (int i = 0; i < m_div_circumference; i++) {
            double phi = (CH_C_2PI * i) / m_div_circumference;
            for (int j = 0; j <= m_div_width; j++) {
                double theta = -CH_C_PI_2 + (CH_C_PI * j) / m_div_width;

                double x = (m_rim_radius + m_height * cos(theta)) * cos(phi) + Tire_IniPos.x();
                double y = m_height * sin(theta) + Tire_IniPos.y();
                double z = (m_rim_radius + m_height * cos(theta)) * sin(phi) + Tire_IniPos.z();
                ChVector<> loc = ChVector<>(x, y, z);

                double nx = cos(theta) * cos(phi);
                double ny = sin(theta);
                double nz = cos(theta) * sin(phi);
                ChVector<> dir = ChVector<>(nx, ny, nz);

                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(loc, dir);
                node->SetMass(0.0);
                my_mesh->AddNode(node);

                // Fix the edge node on the wheel
                // if (j == 0 || j == m_div_width){
                //     auto mlink = chrono_types::make_shared<ChLinkPointFrame>();
                //     mlink->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node), wheel);
                //     sysMBS.Add(mlink);
                // }
            }
        }

        int TotalNumElements = m_div_circumference * m_div_width; //my_mesh->GetNelements();
        int TotalNumNodes = my_mesh->GetNnodes();

        _2D_elementsNodes_mesh.resize(TotalNumElements);
        NodeNeighborElement_mesh.resize(TotalNumNodes);

        // Element dimensions
        double dz = m_thickness;
        double dx = CH_C_2PI * (m_rim_radius + m_height) / (1 * m_div_circumference);
        double dy = CH_C_PI * m_height / m_div_width;

        std::cout << dx  << " dx dy: " << dy << std::endl;

        // Create the ANCF shell elements
        int num_elem = 0;
        for (int i = 0; i < m_div_circumference; i++) {
            for (int j = 0; j < m_div_width; j++) {

                // Adjacent nodes
                int inode0, inode1, inode2, inode3;
                inode1 = j + i * (m_div_width + 1);
                inode2 = j + 1 + i * (m_div_width + 1);
                if (i == m_div_circumference - 1) {
                    inode0 = j;
                    inode3 = j + 1;
                } else {
                    inode0 = j + (i + 1) * (m_div_width + 1);
                    inode3 = j + 1 + (i + 1) * (m_div_width + 1);
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

                // Set element dimensions
                element->SetDimensions(dx, dy);

                // Add a single layers with a fiber angle of 0 degrees.
                element->AddLayer(dz, 0 * CH_C_DEG_TO_RAD, mat);

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

    // Add the mesh to the system
    sysMBS.Add(my_mesh);

    // fluid representation of flexible bodies
    bool multilayer = false;
    bool removeMiddleLayer = false;
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

// =============================================================================
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

    // Create the MBS and FSI systems
    ChSystemSMC sysMBS;
    ChSystemFsi sysFSI(sysMBS);

    ChVector<> gravity = ChVector<>(0, 0, -9.81);
    sysMBS.Set_G_acc(gravity);
    sysFSI.Set_G_acc(gravity);

    sysFSI.SetVerbose(verbose_fsi);

    // Use the default input file or you may enter your input parameters as a command line argument
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_SingleWheelTest.json");
    if (argc == 3) {
        inputJson = std::string(argv[1]);
        wheel_slip = std::stod(argv[2]);
    } else if (argc != 1) {
        std::cout << "usage: ./FSI_Single_Wheel_With_Tire <json_file> <wheel_slip>" << std::endl;
        std::cout << "or to use default input parameters ./FSI_Single_Wheel_With_Tire " << std::endl;
        return 1;
    }

    sysFSI.ReadParametersFromFile(inputJson);

    sysFSI.SetInitialSpacing(iniSpacing);
    sysFSI.SetKernelLength(kernelLength);
    sysFSI.SetStepSize(dT);

    // Set the terrain container size
    sysFSI.SetContainerDim(ChVector<>(bxDim, byDim, bzDim));

    // Set SPH discretization type, consistent or inconsistent
    sysFSI.SetDiscreType(true, true);

    // Set wall boundary condition
    sysFSI.SetWallBC(BceVersion::ADAMI);

    // Set rigid body boundary condition
    sysFSI.SetRigidBodyBC(BceVersion::ADAMI);

    // Set cohsion of the granular material
    sysFSI.SetCohesionForce(0);

    // Setup the SPH method
    sysFSI.SetSPHMethod(FluidDynamics::WCSPH);

    // Set up the periodic boundary condition (if not, set relative larger values)
    ChVector<> cMin(-bxDim / 2 * 10, -byDim / 2 - 0.5 * iniSpacing, -bzDim * 10);
    ChVector<> cMax(bxDim / 2 * 10, byDim / 2 + 0.5 * iniSpacing, bzDim * 20);
    sysFSI.SetBoundaries(cMin, cMax);

    // Initialize the SPH particles
    ChVector<> boxCenter(0.0, 0.0, bzDim / 2);
    ChVector<> boxHalfDim(bxDim / 2, byDim / 2, bzDim / 2);
    sysFSI.AddBoxSPH(iniSpacing, kernelLength, boxCenter, boxHalfDim);

    // Create Solid region and attach BCE SPH particles
    CreateSolidPhase(sysMBS, sysFSI);

    /// Setup the output directory for FSI data
    sysFSI.SetOutputDirectory(out_dir);

    // Set simulation data output length
    sysFSI.SetOutputLength(0);

    // Construction of the FSI system must be finalized before running
    sysFSI.Initialize();
    auto my_mesh = sysFSI.GetFsiMesh();

    auto wheel = sysMBS.Get_bodylist()[1];
    ChVector<> force = actuator->Get_react_force();
    ChVector<> torque = motor->Get_react_torque();
    ChVector<> w_pos = wheel->GetPos();
    ChVector<> w_vel = wheel->GetPos_dt();
    ChVector<> angvel = wheel->GetWvel_loc();

    // Save wheel mesh
    ChTriangleMeshConnected wheel_mesh;
    wheel_mesh.LoadWavefrontMesh(GetChronoDataFile(wheel_obj), false, true);
    wheel_mesh.RepairDuplicateVertexes(1e-9);

    // Write the information into a txt file
    std::ofstream myFile;
    if (output)
        myFile.open(out_dir + "/results.txt", std::ios::trunc);

    // Create a run-tme visualizer
    ChVisualizationFsi fsi_vis(&sysFSI);
    if (render) {
        fsi_vis.SetTitle("Chrono::FSI single wheel with tire demo");
        fsi_vis.SetCameraPosition(ChVector<>(0, -5 * byDim, 5 * bzDim), ChVector<>(0, 0, 0));
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

    // Start the simulation
    unsigned int output_steps = (unsigned int)(1 / (out_fps * dT));
    unsigned int render_steps = (unsigned int)(1 / (render_fps * dT));

    double time = 0.0;
    int current_step = 0;

    ChTimer<> timer;
    timer.start();
    while (time < total_time) {
        // Get the infomation of the wheel
        force = actuator->Get_react_force();
        torque = motor->Get_react_torque();
        w_pos = wheel->GetPos();
        w_vel = wheel->GetPos_dt();
        angvel = wheel->GetWvel_loc();

        if (verbose) {
            std::cout << "time: " << time << std::endl;
            std::cout << "  wheel position:         " << w_pos << std::endl;
            std::cout << "  wheel linear velocity:  " << w_vel << std::endl;
            std::cout << "  wheel angular velocity: " << angvel << std::endl;
            std::cout << "  drawbar pull:           " << force << std::endl;
            std::cout << "  wheel torque:           " << torque << std::endl;
        }

        if (output) {
            myFile << time << "\t" << w_pos.x() << "\t" << w_pos.y() << "\t" << w_pos.z() << "\t" << w_vel.x() << "\t"
                   << w_vel.y() << "\t" << w_vel.z() << "\t" << angvel.x() << "\t" << angvel.y() << "\t" << angvel.z()
                   << "\t" << force.x() << "\t" << force.y() << "\t" << force.z() << "\t" << torque.x() << "\t"
                   << torque.y() << "\t" << torque.z() << "\n";
        }

        if (output && current_step % output_steps == 0) {
            std::cout << "-------- Output" << std::endl;
            sysFSI.PrintParticleToFile(out_dir + "/particles");
            static int counter = 0;
            std::string filename = out_dir + "/vtk/wheel." + std::to_string(counter++) + ".vtk";
            WriteWheelVTK(filename, wheel_mesh, wheel->GetFrame_REF_to_abs());
            std::string filename_flex = out_dir + "/vtk/flex_body." + std::to_string(counter++) + ".vtk";
            fea::ChMeshExporter::writeFrame(my_mesh, (char*)filename_flex.c_str(), MESH_CONNECTIVITY);
        }

        // Render SPH particles
        if (render && current_step % render_steps == 0) {
            if (!fsi_vis.Render())
                break;
        }

        // Call the FSI solver
        sysFSI.DoStepDynamics_FSI();
        time += dT;
        current_step++;
    }
    timer.stop();
    std::cout << "\nSimulation time: " << timer() << " seconds\n" << std::endl;

    if (output)
        myFile.close();

    return 0;
}

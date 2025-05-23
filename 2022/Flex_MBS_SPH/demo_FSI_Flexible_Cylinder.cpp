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
#include <cmath>

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

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono_thirdparty/filesystem/path.h"

#include "chrono_vehicle/wheeled_vehicle/tire/ANCFToroidalTire.h"

#include "chrono/physics/ChIndexedNodes.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChLinkPointPoint.h"

// Chrono namespaces
using namespace chrono;
using namespace chrono::fea;
using namespace chrono::collision;
using namespace chrono::fsi;
using namespace chrono::vehicle;

// Set the output directory
const std::string out_dir = GetChronoOutputPath() + "FSI_Flexible_Cylinder/";
std::string MESH_CONNECTIVITY = out_dir + "Flex_MESH.vtk";

// Dimension of the domain
double smalldis = 1.0e-9;
double bxDim = 3.0 + smalldis;
double byDim = 0.4 + smalldis;
double bzDim = 2.0 + smalldis;

// Dimension of the fluid domain
double fxDim = 1.0 + smalldis;
double fyDim = 0.4 + smalldis;
double fzDim = 1.0 + smalldis;
bool flexible_elem_1D = false;

// Output frequency
bool output = true;
double out_fps = 20;

// Final simulation time
double t_end = 10.0;

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = true;
float render_fps = 100;

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
    ChVector<> cMin = ChVector<>(-5 * bxDim, -byDim / 2.0 - initSpace0 / 2.0, -10 * bzDim );
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
            std::string filename = out_dir + "/vtk/flex_body." + std::to_string(counter++) + ".vtk";
            fea::ChMeshExporter::writeFrame(my_mesh, (char*)filename.c_str(), MESH_CONNECTIVITY);
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

    // ******************************* Flexible bodies ***********************************
    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = chrono_types::make_shared<fea::ChMesh>();
    std::vector<std::vector<int>> _1D_elementsNodes_mesh;
    std::vector<std::vector<int>> _2D_elementsNodes_mesh;

   {
        // Geometry of the plate
        double plate_lenght_x = 0.02;
        double plate_lenght_y = byDim / 2;
        double plate_lenght_z = 40 * initSpace0 ;

        double plate_radius = plate_lenght_z / (2.0 * CH_C_PI);

        ChVector<> center_plate(-bxDim / 2 + fxDim / 2, 0, fzDim + plate_radius * 2 + 1 * initSpace0);

        // Specification of the mesh, numDiv_z is a multiple of 4
        int numDiv_x = 1;
        int numDiv_y = 5;
        int numDiv_z = 20;
        int N_y = numDiv_y + 1;
        int N_z = numDiv_z;
        
        
        // number of elements in the circle
        int num_cir_layer = 1;
        int numDiv_circle = numDiv_z * num_cir_layer + numDiv_z * numDiv_z / 16;
        // number of nodes  in the circle
        int numNodes_circle = N_z * num_cir_layer + (numDiv_z/4 + 1) * (numDiv_z/4 + 1);

        // Number of elements in the z direction is considered as 1
        int TotalNumElements = numDiv_y * numDiv_z + numDiv_circle * 2;
        int TotalNumNodes = N_y * N_z + numNodes_circle * 2;

        _2D_elementsNodes_mesh.resize(TotalNumElements);
        NodeNeighborElement_mesh.resize(TotalNumNodes);

        // For uniform mesh
        double dx = plate_lenght_x / numDiv_x;
        double dy = plate_lenght_y / numDiv_y;
        double dz = plate_lenght_z / numDiv_z;
        // Uniform length of squre in the circle is dl
        double multiple11 = 1 / sqrt(2);
        double dl = plate_radius * 2 / ( multiple11 * 2 * num_cir_layer + numDiv_z / 4 * sqrt(2) );

        std::vector<int> index_squre_x={1,1,-1,-1};
        std::vector<int> index_squre_z={1,-1,-1,1};

        std::vector<double> loc_nodes_squre;
        loc_nodes_squre.resize(N_z * 3);
        // location of nodes in the squre 
        loc_nodes_squre[0] = - dl * numDiv_z / 4 / sqrt(2) + center_plate.x();
        loc_nodes_squre[1]= center_plate.y();
        loc_nodes_squre[2]= center_plate.z();
        int num_num = 3;
        for (int k = 0; k < 4; k++) {
            int j_k = numDiv_z / 4;
            if ( k == 3) {
                j_k = numDiv_z / 4 - 1;
            }
            for (int j = 0; j < j_k; j++) {
                loc_nodes_squre[ num_num ]=
                    loc_nodes_squre[ num_num - 3 ] + dl / sqrt(2) * index_squre_x[k];
                loc_nodes_squre[ num_num + 1 ] =
                    loc_nodes_squre[ num_num + 1 - 3 ];
                loc_nodes_squre[ num_num + 2 ] =
                    loc_nodes_squre[ num_num + 2 - 3 ] + dl / sqrt(2) * index_squre_z[k];
                num_num = num_num + 3;
            }
        }

        // uniform angle
        double d_angle = CH_C_PI * 2 / numDiv_z;        

        // Global definitions
        // =============================================================================
        // Create and add the nodes
        for (int k = 0; k < N_z; k++) {
            for (int j = 0; j < N_y; j++) {
                double loc_x = - plate_radius * cos(d_angle * k) + center_plate.x();
                double loc_y = j * dy - plate_lenght_y / 2 + center_plate.y();
                double loc_z = plate_radius * sin(d_angle * k) + center_plate.z();
                // Node direction
                double dir_x = cos(d_angle * k);
                double dir_y = 0;
                double dir_z = -sin(d_angle * k);

                // Create the node
                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                    ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));

                node->SetMass(0.0);

                // Add node to mesh
                my_mesh->AddNode(node);
            }
        }

        std::vector<int> index_m={-1,1};
        for (int m = 0; m < 2; m++) {
            double loc_x, loc_y, loc_z;
            // Node direction
            double dir_x = 0;
            double dir_y = 1.0;
            double dir_z = 0;
            
            for (int k = 0; k < num_cir_layer + 1; k++) {
                for (int j = 0; j < N_z; j++) {
                    double loc_x_A = - plate_radius * cos(d_angle * j) + center_plate.x();
                    double loc_y_A = index_m[m] * plate_lenght_y / 2 + center_plate.y();
                    double loc_z_A = plate_radius * sin(d_angle * j) + center_plate.z();

                    double loc_x_B = loc_nodes_squre[ j * 3 ];
                    double loc_y_B = index_m[m] * plate_lenght_y / 2 + loc_nodes_squre[ j * 3 + 1 ];
                    double loc_z_B = loc_nodes_squre[ j * 3 + 2 ];

                    loc_x = loc_x_A + (loc_x_B - loc_x_A) / num_cir_layer * k;
                    loc_y = loc_y_A + (loc_y_B - loc_y_A) / num_cir_layer * k;
                    loc_z = loc_z_A + (loc_z_B - loc_z_A) / num_cir_layer * k;
                    
                    // Create the node
                    auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                        ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));

                    node->SetMass(0.0);
                    // Add node to mesh
                    my_mesh->AddNode(node);
                    // Fix the edge nodes on the flat plate and the circle
                    if (k == 0) {
                        auto mlink = chrono_types::make_shared<ChLinkPointPoint>();
                        mlink->Initialize(std::dynamic_pointer_cast<ChNodeFEAxyzD>(node),
                                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(j * N_y + m * (N_y - 1))));
                        sysMBS.Add(mlink);
                    }
                }
            }
            
            for (int l = 0; l < numDiv_z / 4 / 2 - 1; l++) {
                loc_x = loc_x + dl / sqrt(2);
                loc_z = loc_z + dl / sqrt(2);
                auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                    ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));
                node->SetMass(0.0);
                my_mesh->AddNode(node);
                for (int k = 0; k < 4; k++) {
                    int j_k = numDiv_z / 4;
                    if ( k == 3) {
                        j_k = numDiv_z / 4 - 1;
                    }
                    for (int j = 0; j < j_k - 2 * l - 2; j++) {
                        loc_x = loc_x + dl / sqrt(2) * index_squre_x[k];
                        loc_z = loc_z + dl / sqrt(2) * index_squre_z[k];                   
                        auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                            ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));
                        node->SetMass(0.0);
                        my_mesh->AddNode(node);
                    }
                }
            }
            loc_x = loc_x + dl / sqrt(2);
            loc_z = loc_z + dl / sqrt(2);
            auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));
            node->SetMass(0.0);
            my_mesh->AddNode(node);
            if ( ( ( numDiv_z / 4 ) % 2 ) == 1 ) {
                for (int k = 0; k < 3; k++) {
                    loc_x = loc_x + dl / sqrt(2) * index_squre_x[k];
                    loc_z = loc_z + dl / sqrt(2) * index_squre_z[k];                   
                    auto node = chrono_types::make_shared<ChNodeFEAxyzD>(
                        ChVector<>(loc_x, loc_y, loc_z), ChVector<>(dir_x, dir_y, dir_z));
                    node->SetMass(0.0);
                    my_mesh->AddNode(node);
                }
            }
        }

        // Create an isotropic material.
        // All layers for all elements share the same material.
        double rho = 2000;
        double E = 1e7;
        double nu = 0.3;
        auto mat = chrono_types::make_shared<ChMaterialShellANCF>(rho, E, nu);
        // Create the elements
        int num_elem = 0;
        for (int k = 0; k < numDiv_z; k++) {
            for (int j = 0; j < numDiv_y; j++) {
                int node0 = (j + 0) + N_y * (k + 0);
                int node1 = (j + 1) + N_y * (k + 0);
                int node2 = (j + 1) + N_y * (k + 1);
                int node3 = (j + 0) + N_y * (k + 1);
                if ( k == numDiv_z-1 ){
                    node2 = (j + 1);
                    node3 = (j + 0);
                }

                _2D_elementsNodes_mesh[num_elem].push_back(node0);
                _2D_elementsNodes_mesh[num_elem].push_back(node1);
                _2D_elementsNodes_mesh[num_elem].push_back(node2);
                _2D_elementsNodes_mesh[num_elem].push_back(node3);
                NodeNeighborElement_mesh[node0].push_back(num_elem);
                NodeNeighborElement_mesh[node1].push_back(num_elem);
                NodeNeighborElement_mesh[node2].push_back(num_elem);
                NodeNeighborElement_mesh[node3].push_back(num_elem);

                // Create the element and set its nodes.
                auto element = chrono_types::make_shared<ChElementShellANCF_3423>();
                element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node0)),
                                  std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                                  std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                                  std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)));

                // Set element dimensions
                element->SetDimensions(dy, dz);

                // Add a single layers with a fiber angle of 0 degrees.
                element->AddLayer(dx, 0 * CH_C_DEG_TO_RAD, mat);

                // Set structural damping for this element
                element->SetAlphaDamp(0.05);

                // Add element to mesh
                my_mesh->AddElement(element);
                ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                            element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
                std::cout << "Adding element" << num_elem << "  with center:  " << center.x() << " " << center.y()
                          << " " << center.z() << std::endl;
                num_elem++;
            }
        }

        for (int m = 0; m < 2; m++) {
            int node0, node1, node2, node3;
            for (int k = 0; k < num_cir_layer; k++) {
                for (int j = 0; j < numDiv_z; j++) {
                    node0 = (j + 0) + N_z * (k + 0) + N_y * N_z + numNodes_circle * m;
                    node1 = (j + 1) + N_z * (k + 0) + N_y * N_z + numNodes_circle * m;
                    node2 = (j + 1) + N_z * (k + 1) + N_y * N_z + numNodes_circle * m;
                    node3 = (j + 0) + N_z * (k + 1) + N_y * N_z + numNodes_circle * m;
                    if ( j == numDiv_z-1 ){
                        node1 = N_z * (k + 0) + N_y * N_z + numNodes_circle * m;
                        node2 = N_z * (k + 1) + N_y * N_z + numNodes_circle * m;
                    }

                    _2D_elementsNodes_mesh[num_elem].push_back(node0);
                    _2D_elementsNodes_mesh[num_elem].push_back(node1);
                    _2D_elementsNodes_mesh[num_elem].push_back(node2);
                    _2D_elementsNodes_mesh[num_elem].push_back(node3);
                    NodeNeighborElement_mesh[node0].push_back(num_elem);
                    NodeNeighborElement_mesh[node1].push_back(num_elem);
                    NodeNeighborElement_mesh[node2].push_back(num_elem);
                    NodeNeighborElement_mesh[node3].push_back(num_elem);

                    // Create the element and set its nodes.
                    auto element = chrono_types::make_shared<ChElementShellANCF_3423>();
                    element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node0)),
                                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                                    std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)));

                    // Set element dimensions
                    double len_height =
                        0.5 * ((element->GetNodeB()->GetPos() - element->GetNodeA()->GetPos()).Length() + 
                               (element->GetNodeD()->GetPos() - element->GetNodeC()->GetPos()).Length());
                    double len_width = 
                        0.5 * ((element->GetNodeC()->GetPos() - element->GetNodeB()->GetPos()).Length() + 
                               (element->GetNodeD()->GetPos() - element->GetNodeA()->GetPos()).Length());
                    element->SetDimensions(len_height, len_width);

                    // Add a single layers with a fiber angle of 0 degrees.
                    element->AddLayer(dx, 0 * CH_C_DEG_TO_RAD, mat);

                    // Set structural damping for this element
                    element->SetAlphaDamp(0.05);

                    // Add element to mesh
                    my_mesh->AddElement(element);
                    ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                                element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
                    std::cout << "Adding element" << num_elem << "  with center:  " << center.x() << " " << center.y()
                            << " " << center.z() << std::endl;

                    num_elem++;
                }
            }

            for (int l = 0; l < (numDiv_z / 4 + 1) / 2; l++) {
                node0 = node2;
                node3 = node3;
                node1 = node0 + 1;
                node2 = node3 + 1;
                if (numDiv_z / 4 - l * 2 - 1 == 0){
                    node2 = node1 + 1;
                }
                
                _2D_elementsNodes_mesh[num_elem].push_back(node0);
                _2D_elementsNodes_mesh[num_elem].push_back(node1);
                _2D_elementsNodes_mesh[num_elem].push_back(node2);
                _2D_elementsNodes_mesh[num_elem].push_back(node3);
                NodeNeighborElement_mesh[node0].push_back(num_elem);
                NodeNeighborElement_mesh[node1].push_back(num_elem);
                NodeNeighborElement_mesh[node2].push_back(num_elem);
                NodeNeighborElement_mesh[node3].push_back(num_elem);

                // Create the element and set its nodes.
                auto element = chrono_types::make_shared<ChElementShellANCF_3423>();
                element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node0)),
                                std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                                std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                                std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)));

                // Set element dimensions
                element->SetDimensions(dl, dl);

                // Add a single layers with a fiber angle of 0 degrees.
                element->AddLayer(dx, 0 * CH_C_DEG_TO_RAD, mat);

                // Set structural damping for this element
                element->SetAlphaDamp(0.05);

                // Add element to mesh
                my_mesh->AddElement(element);
                ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                            element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
                std::cout << "Adding element" << num_elem << "  with center:  " << center.x() << " " << center.y()
                        << " " << center.z() << std::endl;
                num_elem++;

                for (int k = 0; k < 4; k++) {
                    int j_k = numDiv_z / 4 - l * 2 - 1;
                        if ( k == 3) {
                            j_k = numDiv_z / 4 - l * 2 - 2;
                        }
                    for (int j = 0; j < j_k; j++) {
                        if ( j == (numDiv_z / 4 - l * 2 - 2)){
                            node0 = node1;
                            node3 = node2;
                            node1 = node0 + 1;
                            node2 = node0 + 2;
                        }
                        else {
                            node0 = node1;
                            node3 = node2;
                            node1 = node0 + 1;
                            node2 = node3 + 1;
                        }
                        if (k==3 && j == (numDiv_z / 4 - l * 2 - 3)) {
                            node2 = node1 + 1;
                        }
                        
                        _2D_elementsNodes_mesh[num_elem].push_back(node0);
                        _2D_elementsNodes_mesh[num_elem].push_back(node1);
                        _2D_elementsNodes_mesh[num_elem].push_back(node2);
                        _2D_elementsNodes_mesh[num_elem].push_back(node3);
                        NodeNeighborElement_mesh[node0].push_back(num_elem);
                        NodeNeighborElement_mesh[node1].push_back(num_elem);
                        NodeNeighborElement_mesh[node2].push_back(num_elem);
                        NodeNeighborElement_mesh[node3].push_back(num_elem);

                        // Create the element and set its nodes.
                        auto element = chrono_types::make_shared<ChElementShellANCF_3423>();
                        element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node0)),
                                        std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                                        std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                                        std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)));

                        // Set element dimensions
                        element->SetDimensions(dl, dl);

                        // Add a single layers with a fiber angle of 0 degrees.
                        element->AddLayer(dx, 0 * CH_C_DEG_TO_RAD, mat);

                        // Set structural damping for this element
                        element->SetAlphaDamp(0.05);

                        // Add element to mesh
                        my_mesh->AddElement(element);
                        ChVector<> center = 0.25 * (element->GetNodeA()->GetPos() + element->GetNodeB()->GetPos() +
                                                    element->GetNodeC()->GetPos() + element->GetNodeD()->GetPos());
                        std::cout << "Adding element" << num_elem << "  with center:  " << center.x() << " " << center.y()
                                << " " << center.z() << std::endl;
                        num_elem++;
                        if ( j == (numDiv_z / 4 - l * 2 - 2)){
                            int node00 = node1;
                            int node11 = node2;
                            int node22 = node3;
                            int node33 = node0;
                            node0=node00;
                            node1=node11;
                            node2=node22;
                            node3=node33;
                        }
                    }
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

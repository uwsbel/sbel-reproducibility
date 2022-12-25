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
// Author: Wei Hu
// =============================================================================

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "chrono/ChConfig.h"
#include "chrono/core/ChTimer.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"

#include "chrono_thirdparty/filesystem/path.h"

// Chrono namespaces
using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::geometry;

// Physical properties of terrain particles
double iniSpacing = 0.005;
double kernelLength = 0.005;
double density = 1730.0;
double mu_s = 0.5;

// Dimension of the terrain container
double smalldis = 1.0e-9;
double bxDim = 2.0 + smalldis;
double byDim = 0.05 + smalldis;
double bzDim = 1.0 + smalldis;

// Size of the plate
double p_length = 0.2;
double p_width = 0.05;
double p_thick = 0.05;
double linVel = -0.5;
double angVel = 0.0 * CH_C_PI / 180.0;
double total_mass = 25.0;

// Output directories and settings
const std::string out_dir = GetChronoOutputPath() + "FSI_SandClock_60s_dt5e_5_mu05_OneWayAlpha0003_higherIniPos_Surface";

// Cohesion of the soil
double cohesion = 0.0;

// Set the mbs gravity to 0 when pushing down the plate
ChVector<> gravity_mbs = ChVector<>(0, 0, -9.81);
ChVector<> gravity_fsi = ChVector<>(0, 0, -9.81);

// Initial Position of plate
ChVector<> IniPos(0.0, 0.0, bzDim + p_thick / 2.0 + iniSpacing);
ChVector<> IniVel(0.0, 0.0, linVel);

// Simulation time and stepsize
double total_time = 60.0;
double dT = 5.0e-5;

// Save data as csv files to see the results off-line using Paraview
bool output = true;
int out_fps = 25;

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
bool render = true;
float render_fps = 25;

// Verbose terminal output
bool verbose_fsi = true;
bool verbose = true;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();

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

    bzDim = bzDim + 60 * iniSpacing;
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
    bzDim = bzDim - 60 * iniSpacing;

    // Add BCE particles attached on the walls into FSI system
    // sysFSI.AddBoxBCE(ground, pos_zp, QUNIT, size_XY, 12);
    sysFSI.AddBoxBCE(ground, pos_zn, QUNIT, size_XY, 12);
    sysFSI.AddBoxBCE(ground, pos_xp, QUNIT, size_YZ, 23);
    sysFSI.AddBoxBCE(ground, pos_xn, QUNIT, size_YZ, 23);
    // sysFSI.AddBoxBCE(ground, pos_yp, QUNIT, size_XZ, 13);
    // sysFSI.AddBoxBCE(ground, pos_yn, QUNIT, size_XZ, 13);

    ChVector<> pos_shift(0, 0, - 0 * bzDim / 4);
    ChVector<> size_XY_up_L(bxDim / 8, byDim / 2, 2 * iniSpacing);
    ChVector<> size_XY_up_R(bxDim / 8, byDim / 2, 2 * iniSpacing);
    ChVector<> size_YZ_up_L(2 * iniSpacing, byDim / 2, bzDim / 4);
    ChVector<> size_YZ_up_R(2 * iniSpacing, byDim / 2, bzDim / 4);
    ChVector<> pos_up_XY_L(-1.0 * bxDim / 8 - 3 * iniSpacing, 0, bzDim / 2 - 3 * iniSpacing);
    ChVector<> pos_up_XY_R( 1.0 * bxDim / 8 + 3 * iniSpacing, 0, bzDim / 2 - 3 * iniSpacing);
    ChVector<> pos_up_YZ_L(-1.0 * bxDim / 4 - 3 * iniSpacing, 0, 3 * bzDim / 4);
    ChVector<> pos_up_YZ_R( 1.0 * bxDim / 4 + 1 * iniSpacing, 0, 3 * bzDim / 4);
    // ChQuaternion<> rotatio_L = Q_from_AngY( CH_C_PI / 3.0);
    // ChQuaternion<> rotatio_R = Q_from_AngY(-CH_C_PI / 3.0);
    sysFSI.AddBoxBCE(ground, pos_up_XY_L + pos_shift, QUNIT, size_XY_up_L, 12);
    sysFSI.AddBoxBCE(ground, pos_up_XY_R + pos_shift, QUNIT, size_XY_up_R, 12);
    sysFSI.AddBoxBCE(ground, pos_up_YZ_L + pos_shift, QUNIT, size_YZ_up_L, 23);
    sysFSI.AddBoxBCE(ground, pos_up_YZ_R + pos_shift, QUNIT, size_YZ_up_R, 23);

    // Create a plate
    auto plate = chrono_types::make_shared<ChBody>();

    // Set the general properties of the plate
    double mass = total_mass * 1.0 / 2.0;
    // ChQuaternion<> rotatio = Q_from_AngX(CH_C_PI / 2.0);
    // ChVector<> gyration = chrono::utils::CalcCylinderGyration(p_radius, p_length / 2).diagonal();
    plate->SetPos(IniPos);
    plate->SetPos_dt(IniVel);
    plate->SetMass(mass);
    // plate->SetInertiaXX(mass * gyration);
    // plate->SetRot(rotatio);

    // Set the collision type of the plate
    plate->SetCollide(false);
    plate->SetBodyFixed(true);
    plate->GetCollisionModel()->ClearModel();
    // plate->GetCollisionModel()->SetSafeMargin(0.01);
    // chrono::utils::AddCylinderGeometry(plate.get(), mysurfmaterial, 
    //     p_radius, p_length, ChVector<>(0.0), QUNIT);
    plate->GetCollisionModel()->BuildModel();
    sysMBS.AddBody(plate);

    // ChVector<> size_plate(p_length / 2.0 , p_width / 2.0 , p_thick / 2.0);
    // sysFSI.AddBoxBCE(plate, ChVector<>(0.0), QUNIT, size_plate, 123, true);
    // sysFSI.AddFsiBody(plate);
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
    if (!filesystem::create_directory(filesystem::path(out_dir + "/fsi"))) {
        std::cerr << "Error creating directory " << out_dir + "/fsi" << std::endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(out_dir + "/vtk"))) {
        std::cerr << "Error creating directory " << out_dir + "/vtk" << std::endl;
        return 1;
    }

    // Create the MBS and FSI systems
    ChSystemSMC sysMBS;
    ChSystemFsi sysFSI(sysMBS);

    sysMBS.Set_G_acc(gravity_mbs);
    sysFSI.Set_G_acc(gravity_fsi);

    sysFSI.SetVerbose(verbose_fsi);

    // Use the default input file or you may enter your input parameters as a command line argument
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_SingleWheelTest.json");
    if (argc == 2) {
        inputJson = std::string(argv[1]);
    } else if (argc != 1) {
        std::cout << "usage: ./demo_FSI_SingleWheelTest <json_file> <wheel_slip>" << std::endl;
        std::cout << "or to use default input parameters ./demo_FSI_SingleWheelTest " << std::endl;
        return 1;
    }

    sysFSI.ReadParametersFromFile(inputJson);

    sysFSI.SetInitialSpacing(iniSpacing);
    sysFSI.SetKernelLength(kernelLength);
    sysFSI.SetStepSize(dT);
    sysFSI.SetDensity(density);
    sysFSI.SetFriction(mu_s);

    // Set the terrain container size
    sysFSI.SetContainerDim(ChVector<>(bxDim, byDim, bzDim));

    // Set SPH discretization type, consistent or inconsistent
    sysFSI.SetDiscreType(true, false);

    // Set wall boundary condition
    sysFSI.SetWallBC(BceVersion::ADAMI);

    // Set rigid body boundary condition
    sysFSI.SetRigidBodyBC(BceVersion::ADAMI);

    // Set cohsion of the granular material
    sysFSI.SetCohesionForce(cohesion);

    // Setup the SPH method
    sysFSI.SetSPHMethod(FluidDynamics::WCSPH);

    // Set up the periodic boundary condition (if not, set relative larger values)
    ChVector<> cMin(-bxDim / 2 * 5, -byDim / 2 - 0.5 * iniSpacing, -bzDim * 10);
    ChVector<> cMax( bxDim / 2 * 5,  byDim / 2 + 0.5 * iniSpacing,  bzDim * 10);
    sysFSI.SetBoundaries(cMin, cMax);

    // Initialize the SPH particles
    ChVector<> boxCenter(0.0, 0.0, 3 * bzDim / 4);
    ChVector<> boxHalfDim(bxDim / 4, byDim / 2, bzDim / 4);
    sysFSI.AddBoxSPH(iniSpacing, kernelLength, boxCenter, boxHalfDim);

    ChVector<> boxCenterExtra(0.0, 0.0, 3 * iniSpacing);
    ChVector<> boxHalfDimExtra(bxDim / 2, byDim / 2, 3 * iniSpacing);
    sysFSI.AddBoxSPH(iniSpacing, kernelLength, boxCenterExtra, boxHalfDimExtra);

    // Create Solid region and attach BCE SPH particles
    CreateSolidPhase(sysMBS, sysFSI);

    // Set simulation data output length
    sysFSI.SetOutputLength(0);

    // Construction of the FSI system must be finalized before running
    sysFSI.Initialize();

    auto plate = sysMBS.Get_bodylist()[1];
    ChVector<> force = actuator->Get_react_force();
    ChVector<> torque = motor->Get_react_torque();
    ChVector<> pos = plate->GetPos();
    ChVector<> vel = plate->GetPos_dt();
    ChVector<> angvel = plate->GetWvel_loc();
    double sinkage = IniPos.z() - pos.z();

    // Write the information into a txt file
    std::ofstream myFile;
    std::ofstream myDBP;
    std::ofstream myTorque;
    if (output){
        myFile.open(out_dir + "/results.txt", std::ios::trunc);
        myDBP.open(out_dir + "/DBP.txt", std::ios::trunc);
        myTorque.open(out_dir + "/Torque.txt", std::ios::trunc);
    }
        

    // Create a run-tme visualizer
    ChVisualizationFsi fsi_vis(&sysFSI);
    if (render) {
        fsi_vis.SetTitle("Chrono::FSI Bevameter Demo");
        fsi_vis.SetCameraPosition(ChVector<>(0, -5 * byDim, 5 * bzDim), ChVector<>(0, 0, 0));
        fsi_vis.SetCameraMoveScale(1.0f);
        fsi_vis.EnableBoundaryMarkers(false);
        fsi_vis.Initialize();
    }

    // Start the simulation
    unsigned int output_steps = (unsigned int)round(1 / (out_fps * dT));
    unsigned int render_steps = (unsigned int)round(1 / (render_fps * dT));

    double time = 0.0;
    int current_step = 0;

    ChTimer<> timer;
    timer.start();
    while (time < total_time) {
        // Get the infomation of the plate
        force = actuator->Get_react_force();
        torque = motor->Get_react_torque();
        pos = plate->GetPos();
        vel = plate->GetPos_dt();
        angvel = plate->GetWvel_loc();
        sinkage = IniPos.z() - pos.z();

        if (verbose) {
            std::cout << "time: " << time << std::endl;
            std::cout << "  plate position:         " << pos << std::endl;
            std::cout << "  plate linear velocity:  " << vel << std::endl;
            std::cout << "  plate angular velocity: " << angvel << std::endl;
            std::cout << "  plate DBP:              " << force << std::endl;
            std::cout << "  plate torque:           " << torque << std::endl;
        }

        if (output) {
            myFile << time << "\t" << pos.x() << "\t" << pos.y() << "\t" << pos.z() << "\t" << vel.x() << "\t"
                   << vel.y() << "\t" << vel.z() << "\t" << angvel.x() << "\t" << angvel.y() << "\t" << angvel.z()
                   << "\t" << force.x() << "\t" << force.y() << "\t" << force.z() << "\t" << torque.x() << "\t"
                   << torque.y() << "\t" << torque.z() << "\t" << sinkage << "\n";

            myDBP << time << "\t" << sinkage << "\t" 
                  << force.x() << "\t" << force.y() << "\t" << force.z() << "\n";

            myTorque << time << "\t" << sinkage << "\t" 
                     << torque.x() << "\t" << torque.y() << "\t" << torque.z() << "\n";
        }

        if (output && current_step % output_steps == 0) {
            std::cout << "-------- Output" << std::endl;
            sysFSI.PrintParticleToFile(out_dir + "/particles");
            sysFSI.PrintFsiInfoToFile(out_dir + "/fsi", time);
            static int counter = 0;
            std::string filename = out_dir + "/vtk/plate." + std::to_string(counter++) + ".vtk";
            // WriteCylinderVTK(filename, p_radius, p_length, sysFSI.GetFsiBodies()[0]->GetFrame_REF_to_abs(), 100);
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

    if (output){
        myFile.close();
        myDBP.close();
        myTorque.close();
    }
        
    return 0;
}

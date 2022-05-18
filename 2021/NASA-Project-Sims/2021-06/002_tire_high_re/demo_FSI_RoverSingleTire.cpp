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

// General Includes
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <valarray>
#include <string>
#include <sstream>
#include <cmath>

// Chrono includes
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/core/ChTransform.h"

// Chrono fsi includes
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsJSON.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"

#include "chrono/ChConfig.h"
#include "chrono/core/ChStream.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/physics/ChLinkMotorRotationSpeed.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_thirdparty/filesystem/path.h"

// Chrono namespaces
using namespace chrono;
using namespace collision;
using namespace chrono::geometry;

using std::cout;
using std::endl;
std::ofstream simParams;
typedef fsi::Real Real;
 
// Output directories and settings
const std::string out_dir = GetChronoOutputPath() + "FSI_Rover_Single_Tire/";
std::string demo_dir;

// Save data as csv files to see the results off-line using Paraview
bool save_output = true;

// Dimension of the space domain
Real bxDim;
Real byDim;
Real bzDim;
// Dimension of the fluid domain
Real fxDim;
Real fyDim;
Real fzDim;

// Size of the wheel
double wheel_length;
double wheel_radius;
double wheel_slip;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();

// -----------------------------------------------------------------
void ShowUsage() {
    cout << "usage: ./demo_FSI_RoverSingleWheel <json_file>" << endl;
    cout << "or to use default input parameters ./demo_FSI_RoverSingleTire " << endl;
}

//------------------------------------------------------------------
// Function to add walls into Chrono system
//------------------------------------------------------------------
void AddWall(std::shared_ptr<ChBody> body, 
             const ChVector<>& dim, 
             std::shared_ptr<ChMaterialSurface> mat,
             const ChVector<>& loc) {
    body->GetCollisionModel()->AddBox(mat, dim.x(), dim.y(), dim.z(), loc);
    auto box = chrono_types::make_shared<ChBoxShape>();
    box->GetBoxGeometry().Size = dim;
    box->GetBoxGeometry().Pos = loc;
}

//------------------------------------------------------------------
// Function to save wheel to Paraview VTK files
//------------------------------------------------------------------
void WritewheelVTK(ChSystemSMC& mphysicalSystem, 
                  int this_frame,
                  std::shared_ptr<fsi::SimParams> paramsH) {
    auto body = mphysicalSystem.Get_bodylist()[1];
    ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
    ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
    ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    std::string obj_path = "./hmmwv_tire_coarse_closed_small.obj";
    double scale_ratio = 1.0;
    mmesh->LoadWavefrontMesh(obj_path, false, true);
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
    mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

    char filename[4096];
    if(1==0){// save to obj file
        sprintf(filename, "%s/hmmwv_tire_%d.obj", paramsH->demo_dir, this_frame);
        std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
        geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
    }
    if(1==1){// save to vtk file
        sprintf(filename, "%s/hmmwv_tire_%d.vtk", paramsH->demo_dir, this_frame);
        std::ofstream file;
        file.open(filename);
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

}

//------------------------------------------------------------------
// Function to save the paraview files
//------------------------------------------------------------------
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemSMC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int this_frame,
                       double mTime,
                       std::shared_ptr<ChBody> wheel) {
    // Simulation steps between two output frames
    int out_steps = (int)ceil((1.0 / paramsH->dT) / paramsH->out_fps);

    // Simulation time between two output frames
    double frame_time = 1.0 / paramsH->out_fps;

    // Output data to files
    if (save_output && std::abs(mTime - (this_frame)*frame_time) < 1e-9) {
        // save particles to cvs files
        fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2->posRadD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->velMasD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->rhoPresMuD,
                                myFsiSystem.GetDataManager()->fsiGeneralData->sr_tau_I_mu_i,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray_FEA, demo_dir, true);
        // save rigid bodies to vtk files
        char SaveAsRigidObjVTK[256];
        static int RigidCounter = 0;
        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (demo_dir + "/wheel.%d.vtk").c_str(), RigidCounter);
        WritewheelVTK(mphysicalSystem, this_frame, paramsH);
        RigidCounter++;
        cout << "\n--------------------------------\n" << endl;
        cout << "------------ Output Frame:   " << this_frame << endl;
        cout << "------------ Sim Time:       " << mTime << " (s)\n" <<endl;
        cout << "--------------------------------\n" << endl;
    }
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if FSI, 
// their BCE representation are created and added to the systems
//------------------------------------------------------------------
void CreateSolidPhase(ChSystemSMC& mphysicalSystem,
                      fsi::ChSystemFsi& myFsiSystem,
                      std::shared_ptr<fsi::SimParams> paramsH) {
    // Set gravity to the rigid body system in chrono
    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
    mphysicalSystem.Set_G_acc(gravity);

    // Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mysurfmaterial->SetYoungModulus(1e8);
    mysurfmaterial->SetFriction(0.2f);
    mysurfmaterial->SetRestitution(0.05f);
    mysurfmaterial->SetAdhesion(0);

    // Create the geometry of the boundaries
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    /// Bottom wall
    ChVector<> size_XY(bxDim / 2 + 3 * initSpace0, byDim / 2 + 0 * initSpace0, 2 * initSpace0);
    ChVector<> pos_zn(0, 0, -3 * initSpace0);
    ChVector<> pos_zp(0, 0, bzDim + 2 * initSpace0);

    /// left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 0 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);

    /// Front and back Wall
    ChVector<> size_XZ(bxDim / 2 + 3 * initSpace0, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 0 * initSpace0);


    // -----------------------------------------------------
    // Create a container -- always FIRST body in the system
    // -----------------------------------------------------
    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetPos(ChVector<>(0.0, 0.0, 0.0));
    ground->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ground->SetIdentifier(-1);
    ground->SetBodyFixed(true);
    ground->GetCollisionModel()->ClearModel();
    ground->GetCollisionModel()->SetSafeMargin(initSpace0 / 2);

    // Add the walls into chrono system
    // AddWall(ground, size_XY, mysurfmaterial, pos_zp);
    AddWall(ground, size_XY, mysurfmaterial, pos_zn);
    AddWall(ground, size_YZ, mysurfmaterial, pos_xp);
    AddWall(ground, size_YZ, mysurfmaterial, pos_xn);
    // AddWall(ground, size_XZ, mysurfmaterial, pos_yp);
    // AddWall(ground, size_XZ, mysurfmaterial, pos_yn);
    ground->GetCollisionModel()->BuildModel();
    ground->SetCollide(true);
    mphysicalSystem.AddBody(ground);

    // Add BCE particles attached on the walls into FSI system
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_zp, chrono::QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_zn, chrono::QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xp, chrono::QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_xn, chrono::QUNIT, size_YZ, 23);
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yp, chrono::QUNIT, size_XZ, 13);
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, ground, pos_yn, chrono::QUNIT, size_XZ, 13);


    // -----------------------------------------------------
    // Create the wheel -- always SECOND body in the system
    // -----------------------------------------------------
    // auto wheel = chrono_types::make_shared<ChBody>();
    // // Set the general properties of the wheel
    // double volume = utils::CalcCylinderVolume(wheel_radius, wheel_length / 2);
    // double density = paramsH->bodyDensity;
    // double mass = density * volume;
    // ChVector<> wheel_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ + fzDim );
    // ChVector<> wheel_vel = ChVector<>(paramsH->bodyIniVelX, paramsH->bodyIniVelY, paramsH->bodyIniVelZ);
    // ChQuaternion<> wheel_rot = QUNIT;
    // ChVector<> gyration = utils::CalcCylinderGyration(wheel_radius, wheel_length / 2).diagonal();
    // wheel->SetPos(wheel_pos);
    // wheel->SetPos_dt(wheel_vel);
    // wheel->SetMass(mass);
    // wheel->SetInertiaXX(mass * gyration);
    // wheel->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0)); // set an initial anular velocity (rad/s)

    // // Set the collision type of the wheel
    // wheel->SetCollide(true);
    // wheel->SetBodyFixed(false);
    // wheel->GetCollisionModel()->ClearModel();
    // wheel->GetCollisionModel()->SetSafeMargin(initSpace0);
    // utils::AddCylinderGeometry(wheel.get(), mysurfmaterial, wheel_radius, wheel_length, 
    //                            ChVector<>(0.0, 0.0, 0.0),
    //                            ChQuaternion<>(1, 0, 0, 0));
    // wheel->GetCollisionModel()->BuildModel();

    // // Add this body to chrono system
    // mphysicalSystem.AddBody(wheel);

    // // Add this body to the FSI system (only those have inetraction with fluid)
    // myFsiSystem.AddFsiBody(wheel);

    // // Add BCE particles attached on the wheel into FSI system
    // fsi::utils::AddCylinderBce(myFsiSystem.GetDataManager(), paramsH, wheel, ChVector<>(0, 0, 0),
    //                            ChQuaternion<>(1, 0, 0, 0), wheel_radius, wheel_length + initSpace0, 
    //                            paramsH->HSML, false);

    // load mesh from obj file
    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    std::string obj_path = "./hmmwv_tire_coarse_closed_small.obj";
    double scale_ratio = 1.0;
    mmesh->LoadWavefrontMesh(obj_path, false, true);
    // mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(body_rot));       // rotate the mesh if needed
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

    // compute mass inertia from mesh
    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    double mdensity = paramsH->bodyDensity;
    mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);
    mcog = ChVector<>(0.0, 0.0, 0.0);

    // set the abs orientation, position and velocity
    auto wheel = chrono_types::make_shared<ChBodyAuxRef>();
    ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 0));
    ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ + fzDim) ;
    ChVector<> Body_vel = ChVector<>(paramsH->bodyIniVelX, paramsH->bodyIniVelY, paramsH->bodyIniVelZ);

    // Set the COG coordinates to barycenter, without displacing the REF reference.
    // Make the COG frame a principal frame.
    wheel->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    wheel->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    wheel->SetInertiaXX(mdensity * principal_I);
    wheel->SetPos_dt(Body_vel);
    wheel->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0)); // set an initial anular velocity (rad/s)
    
    // Set the absolute position of the body:
    wheel->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos),ChQuaternion<>(Body_rot)));                              
    mphysicalSystem.AddBody(wheel);

    wheel->SetBodyFixed(false);
    wheel->GetCollisionModel()->ClearModel();
    wheel->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
    wheel->GetCollisionModel()->BuildModel();
    wheel->SetCollide(false);

    auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
    masset_mesh->SetMesh(mmesh);
    masset_mesh->SetBackfaceCull(true);
    wheel->AddAsset(masset_mesh);

    // Add this body to the FSI system
    myFsiSystem.AddFsiBody(wheel);
    std::string BCE_path = "./BCE_hmmwv_tire_coarse_closed_small_d5mm.txt";
    fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, wheel, BCE_path, 
                                ChVector<double>(0), QUNIT, scale_ratio);



    // -----------------------------------------------------
    // Create the chassis -- always THIRD body in the system
    // -----------------------------------------------------
    // Initially, the chassis is fixed to ground.
    // It is released after the settling phase.
    auto chassis = chrono_types::make_shared<ChBody>();
    // chassis->SetIdentifier(Id_chassis);
    chassis->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    chassis->SetPos(wheel->GetPos());
    chassis->SetCollide(false);
    chassis->SetBodyFixed(false);

    // Add geometry of the chassis.
    chassis->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(chassis.get(), mysurfmaterial, ChVector<>(0.1, 0.1, 0.1), ChVector<>(0, 0, 0));
    chassis->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(chassis);

    // ---------------------------------------------------
    // Create the axle -- always FOURTH body in the system
    // ---------------------------------------------------
    auto axle = chrono_types::make_shared<ChBody>();
    // axle->SetIdentifier(Id_axle);
    axle->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    axle->SetPos(wheel->GetPos());
    axle->SetCollide(false);
    axle->SetBodyFixed(false);
    // Add geometry of the axle.
    axle->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(axle.get(), mysurfmaterial, 0.5, ChVector<>(0, 0, 0));
    axle->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(axle);


    // =============================================================================
    // Connect the chassis to the containing bin (ground) through a translational
    // joint and create a linear actuator.
    // =============================================================================
    auto prismatic1 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic1->Initialize(ground, chassis, ChCoordsys<>(chassis->GetPos(), Q_from_AngY(CH_C_PI_2)));
    prismatic1->SetName("prismatic_chassis_ground");
    mphysicalSystem.AddLink(prismatic1);

    double angVel = paramsH->bodyIniAngVel;
    double velocity = angVel * wheel_radius * (1.0 - wheel_slip);
    auto actuator_fun = chrono_types::make_shared<ChFunction_Ramp>(0.0, velocity);

    // auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
    actuator->Initialize(ground, chassis, false, ChCoordsys<>(chassis->GetPos(), QUNIT),
                         ChCoordsys<>(chassis->GetPos() + ChVector<>(1, 0, 0), QUNIT));
    actuator->SetName("actuator");
    actuator->Set_lin_offset(1);
    actuator->Set_dist_funct(actuator_fun);
    mphysicalSystem.AddLink(actuator);

    // =============================================================================
    // Connect the axle to the chassis through a vertical translational
    // joint.
    // =============================================================================
    auto prismatic2 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic2->Initialize(chassis, axle, ChCoordsys<>(chassis->GetPos(), QUNIT));
    prismatic2->SetName("prismatic_axle_chassis");
    mphysicalSystem.AddLink(prismatic2);

    // =============================================================================
    // Connect the wheel to the axle through a engine joint.
    // =============================================================================
    // auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();
    motor->SetName("engine_wheel_axle");
    motor->Initialize(wheel, axle, ChFrame<>(wheel->GetPos(), chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, angVel));
    mphysicalSystem.AddLink(motor);
}


// =============================================================================
int main(int argc, char* argv[]) {
     // Create a physics system and an FSI system
    ChSystemSMC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(mphysicalSystem);

    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();

    // Use the default input file or you may enter your input parameters as a command line argument
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_RoverSingleWheel.json");
    if (argc == 3) {
        fsi::utils::ParseJSON(argv[1], paramsH, fsi::mR3(bxDim, byDim, bzDim));
        std::string input_json = std::string(argv[1]);
        inputJson = GetChronoDataFile(input_json);
        wheel_slip = std::stod(argv[2]); 
    } else {
        ShowUsage();
        return 1;
    }

    // Dimension of the space domain
    bxDim = paramsH->boxDimX;
    byDim = paramsH->boxDimY;
    bzDim = paramsH->boxDimZ;
    // Dimension of the fluid domain
    fxDim = paramsH->fluidDimX;
    fyDim = paramsH->fluidDimY;
    fzDim = paramsH->fluidDimZ;

    // Size of the dropping wheel
    wheel_radius = paramsH->bodyRad;
    wheel_length = paramsH->bodyLength;

    // Set up the periodic boundary condition (if not, set relative larger values)
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    paramsH->cMin = chrono::fsi::mR3(-bxDim / 2 * 10, -byDim / 2 - 0.5 * initSpace0, -bzDim * 10 - 10 * initSpace0);
    paramsH->cMax = chrono::fsi::mR3( bxDim / 2 * 10,  byDim / 2 + 0.5 * initSpace0,  bzDim * 10 + 10 * initSpace0);

    // Set the time integration type and the linear solver type (only for ISPH)
    myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);
    myFsiSystem.SetFluidSystemLinearSolver(paramsH->LinearSolver);

    // Call FinalizeDomainCreating to setup the binning for neighbor search
    fsi::utils::FinalizeDomain(paramsH);
    fsi::utils::PrepareOutputDir(paramsH, demo_dir, out_dir, inputJson);

    // Create Fluid region and discretize with SPH particles
    ChVector<> boxCenter(0.0, 0.0, fzDim / 2);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);

    // Use a chrono sampler to create a bucket of points
    utils::GridSampler<> sampler(initSpace0);
    utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    size_t numPart = points.size();
    for (int i = 0; i < numPart; i++) {
        // Calculate the pressure of a steady state (p = rho*g*h)
        Real pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim);
        Real rho_ini = paramsH->rho0 + pre_ini / (paramsH->Cs * paramsH->Cs);
        myFsiSystem.GetDataManager()->AddSphMarker(
            fsi::mR4(points[i].x(), points[i].y(), points[i].z(), paramsH->HSML),
            fsi::mR3(1e-10),
            fsi::mR4(rho_ini, pre_ini, paramsH->mu0, -1));
    }
    size_t numPhases = myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.size();
    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(0, (int)numPart, -1, -1));
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4((int)numPart, (int)numPart, 0, 0));
    }

    // Create Solid region and attach BCE SPH particles
    CreateSolidPhase(mphysicalSystem, myFsiSystem, paramsH);

    // Construction of the FSI system must be finalized before running
    myFsiSystem.Finalize();

    // Set up integrator for the multi-body dynamics system
    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    mystepper->SetAlpha(-0.2);
    mystepper->SetMaxiters(1000);
    mystepper->SetAbsTolerances(1e-6);
    mystepper->SetMode(ChTimestepperHHT::ACCELERATION);
    mystepper->SetScaling(true);

    auto wheel = mphysicalSystem.Get_bodylist()[1];
    ChVector<> force = actuator->Get_react_force();
    ChVector<> torque = motor->Get_react_torque();
    ChVector<> w_pos = wheel->GetPos();
    ChVector<> w_vel = wheel->GetPos_dt();
    ChVector<> angvel = wheel->GetWvel_loc();

    /// write the infomation into file
    std::ofstream myFile;

    // Start the simulation
    Real time = 0;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    double TIMING_sta = clock();
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / paramsH->out_fps;
        int this_frame = (int)floor((time + 1e-9) / frame_time);

        // Get the wheel from the FSI system and Save data of the simulation
        std::vector<std::shared_ptr<ChBody>>& FSI_Bodies = myFsiSystem.GetFsiBodies();

        // auto wheel = FSI_Bodies[0];
        SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, this_frame, time, wheel);

        // Get the infomation of the wheel
        force = actuator->Get_react_force();
        torque = motor->Get_react_torque();
        w_pos = wheel->GetPos();
        w_vel = wheel->GetPos_dt();
        angvel = wheel->GetWvel_loc();
        printf("wheel position = %f,%f,%f\n", w_pos.x(), w_pos.y(), w_pos.z());
        printf("wheel velocity = %f,%f,%f\n", w_vel.x(), w_vel.y(), w_vel.z());
        printf("wheel ang velocity = %f,%f,%f\n", angvel.x(), angvel.y(), angvel.z());
        printf("draw-bar pull = %f,%f,%f\n", force.x(), force.y(), force.z());
        printf("wheel torque = %f,%f,%f\n", torque.x(), torque.y(), torque.z());

        myFile.open(demo_dir + "/results.txt", std::ios::app);
        myFile << time << "\t"
            << w_pos.x() << "\t" << w_pos.y() << "\t" << w_pos.z() << "\t"
            << w_vel.x() << "\t" << w_vel.y() << "\t" << w_vel.z() << "\t"
            << angvel.x() << "\t" << angvel.y() << "\t" << angvel.z() << "\t"
            << force.x() << "\t" << force.y() << "\t" << force.z() << "\t"
            << torque.x() << "\t" << torque.y() << "\t" << torque.z() << "\n";
        myFile.close();
        
        // Call the FSI solver
        myFsiSystem.DoStepDynamics_FSI();
        time += paramsH->dT;
    }

    // Total computational cost
    double TIMING_end = (clock() - TIMING_sta) / (double)CLOCKS_PER_SEC;
    printf("\nSimulation Finished in %f (s)\n", TIMING_end);

    return 0;
}


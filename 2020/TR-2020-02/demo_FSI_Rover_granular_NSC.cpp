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

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/physics/ChParticlesClones.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/physics/ChLinkMotorRotationSpeed.h"

// Chrono fsi includes
#include "chrono_fsi/utils/ChUtilsTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsJSON.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"

#define AddBoundaries

// Chrono namespaces
using namespace chrono;
using namespace collision;
using namespace chrono::geometry;

using std::cout;
using std::endl;
std::ofstream simParams;

//----------------------------
// output directories and settings
//----------------------------
const std::string out_dir = GetChronoOutputPath() + "FSI_Rover/";
std::string demo_dir;
bool pv_output = true;
typedef fsi::Real Real;

Real smalldis = 1.0e-9;
/// Dimension of the space domain
Real bxDim = 0.0 + smalldis;
Real byDim = 0.0 + smalldis;
Real bzDim = 0.0 + smalldis;
/// Dimension of the fluid domain
Real fxDim = 0.0 + smalldis;
Real fyDim = 0.0 + smalldis;
Real fzDim = 0.0 + smalldis;

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/// Forward declaration of helper functions
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemNSC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int tStep,
                       double mTime);

void AddWall(std::shared_ptr<ChBody> body, const ChVector<>& dim, const ChVector<>& loc) {
    body->GetCollisionModel()->AddBox(dim.x(), dim.y(), dim.z(), loc);
    auto box = chrono_types::make_shared<ChBoxShape>();
    box->GetBoxGeometry().Size = dim;
    box->GetBoxGeometry().Pos = loc;
}
//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------

void CreateSolidPhase(ChSystemNSC& mphysicalSystem,
                      fsi::ChSystemFsi& myFsiSystem,
                      std::shared_ptr<fsi::SimParams> paramsH);
void ShowUsage() {
    cout << "usage: ./demo_FSI_Granular_Rover <json_file>" << endl;
}

// =============================================================================
int main(int argc, char* argv[]) {
    // create a physics system
    ChSystemNSC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(mphysicalSystem);
    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_Rover_granular.json");
    if (argc == 1 && fsi::utils::ParseJSON(inputJson.c_str(), paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
    } else if (argc == 2 && fsi::utils::ParseJSON(argv[1], paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
    } else {
        ShowUsage();
        return 1;
    }

    /// Dimension of the space domain
    bxDim = paramsH->boxDimX + smalldis;
    byDim = paramsH->boxDimY + smalldis;
    bzDim = paramsH->boxDimZ + smalldis;
    /// Dimension of the fluid domain
    fxDim = paramsH->fluidDimX + smalldis;
    fyDim = paramsH->fluidDimY + smalldis;
    fzDim = paramsH->fluidDimZ + smalldis;

    myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);
    myFsiSystem.SetFluidSystemLinearSolver(paramsH->LinearSolver);

    // fsi::utils::ParseJSON sets default values to cMin and cMax which may need
    // to be modified depending on the case (e.g periodic BC)
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    paramsH->cMin = fsi::mR3(-bxDim / 2, -byDim / 2, -bzDim / 2 - 5 * initSpace0) * 10 - 4 * initSpace0;
    paramsH->cMax = fsi::mR3(bxDim / 2, byDim / 2, bzDim + 10 * initSpace0) * 10 + 4 * initSpace0;
    // call FinalizeDomainCreating to setup the binning for neighbor search or write your own
    fsi::utils::FinalizeDomainCreating(paramsH);
    fsi::utils::PrepareOutputDir(paramsH, demo_dir, out_dir, argv[1]);

    // ******************************* Create Fluid region ****************************************
    /// Create an initial box of fluid
    utils::GridSampler<> sampler(initSpace0);
    /// Use a chrono sampler to create a bucket of fluid
    ChVector<> boxCenter(0, 0, fzDim / 2 + 0 * initSpace0);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);
    utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);
    /// Add fluid markers from the sampler points to the FSI system
    size_t numPart = points.size();
    for (int i = 0; i < numPart; i++) {
        Real pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim);
        Real rho_ini = paramsH->rho0 + pre_ini / (paramsH->Cs * paramsH->Cs);
        myFsiSystem.GetDataManager()->AddSphMarker(
            fsi::mR4(points[i].x(), points[i].y(), points[i].z(), paramsH->HSML), fsi::mR3(0.0, 0.0, 0.0),
            fsi::mR4(paramsH->rho0, pre_ini, paramsH->mu0, -1),  // initial presssure modified as 0.0
            fsi::mR3(0.0e0),                               // tauxxyyzz
            fsi::mR3(0.0e0));                              // tauxyxzyz
    }

    size_t numPhases = myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.size();

    if (numPhases != 0) {
        std::cout << "Error! numPhases is wrong, thrown from main\n" << std::endl;
        std::cin.get();
        return -1;
    } else {
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(0, numPart, -1, -1));
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(numPart, numPart, 0, 0));
    }

    /// Create MBD or FE model
    CreateSolidPhase(mphysicalSystem, myFsiSystem, paramsH);
    /// Construction of the FSI system must be finalized
    myFsiSystem.Finalize();

    /// Get the body from the FSI system for visualization
    double mTime = 0;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    stepEnd = 1000000;

    /// use the following to write a VTK file of the Rover
    std::vector<std::vector<double>> vCoor;
    std::vector<std::vector<int>> faces;
    std::string RigidConectivity = demo_dir + "RigidConectivity.vtk";

    /// Set up integrator for the MBD
    // mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
    // auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    // mystepper->SetAlpha(-0.2);
    // mystepper->SetMaxiters(1000);
    // mystepper->SetAbsTolerances(1e-6);
    // mystepper->SetMode(ChTimestepperHHT::ACCELERATION);
    // mystepper->SetScaling(true);

    /// Get the body from the FSI system
    std::vector<std::shared_ptr<ChBody>>& FSI_Bodies = myFsiSystem.GetFsiBodies();
    auto Rover = FSI_Bodies[0];
    SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, 0, 0);

    /// write the Penetration into file
    std::ofstream myFile;
    myFile.open("./Rover_position.txt", std::ios::trunc);
    myFile.close();
    myFile.open("./Rover_position.txt", std::ios::app);
    myFile << 0.0 << "\t"
           << Rover->GetPos().x() << "\t" 
           << Rover->GetPos().y() << "\t"
           << Rover->GetPos().z() << "\t"
           << Rover->GetPos_dt().x() << "\t" 
           << Rover->GetPos_dt().y() << "\t"
           << Rover->GetPos_dt().z() << "\n";
    myFile.close();

    Real time = 0;
    Real Global_max_dT = paramsH->dT_Max;
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / paramsH->out_fps;
        int next_frame = (int)floor((time + 1e-6) / frame_time) + 1;
        double next_frame_time = next_frame * frame_time;
        double max_allowable_dt = next_frame_time - time;
        if (max_allowable_dt > 1e-7)
            paramsH->dT_Max = std::min(Global_max_dT, max_allowable_dt);
        else
            paramsH->dT_Max = Global_max_dT;

        myFsiSystem.DoStepDynamics_FSI();
        time += paramsH->dT;
        SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, next_frame, time);

        auto bin = mphysicalSystem.Get_bodylist()[0];
        auto Rover = mphysicalSystem.Get_bodylist()[1];

        printf("bin=%f,%f,%f\n", bin->GetPos().x(), bin->GetPos().y(), bin->GetPos().z());
        printf("Rover=%f,%f,%f\n", Rover->GetPos().x(), Rover->GetPos().y(), Rover->GetPos().z());
        printf("Rover=%f,%f,%f\n", Rover->GetPos_dt().x(), Rover->GetPos_dt().y(), Rover->GetPos_dt().z());
        myFile.open("./Rover_position.txt", std::ios::app);
        myFile << time << "\t"
               << Rover->GetPos().x() << "\t" 
               << Rover->GetPos().y() << "\t"
               << Rover->GetPos().z() << "\t"
               << Rover->GetPos_dt().x() << "\t" 
               << Rover->GetPos_dt().y() << "\t"
               << Rover->GetPos_dt().z() << "\n";
        myFile.close();

        if (time > paramsH->tFinal)
            break;
    }

    return 0;
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------
void CreateSolidPhase(ChSystemNSC& mphysicalSystem,
                      fsi::ChSystemFsi& myFsiSystem,
                      std::shared_ptr<fsi::SimParams> paramsH) {
    Real initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
    mphysicalSystem.Set_G_acc(gravity);

	collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0025);
	collision::ChCollisionModel::SetDefaultSuggestedMargin(0.0025);

    auto mfloor2 = chrono_types::make_shared<ChBodyEasyBox>(10, 10, 0.02, 1000, true, false);
    mfloor2->SetPos(ChVector<>(0, 0, 0));
    mfloor2->SetBodyFixed(true);
    mphysicalSystem.Add(mfloor2);

    /// Bottom wall
    ChVector<> sizeBottom(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> posBottom(0, 0, -3 * initSpace0);
    ChVector<> posTop(0, 0, bzDim + 2 * initSpace0);

    /// left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);

    /// Front and back Wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 0 * initSpace0);

    /// Fluid-Solid Coupling at the walls via Condition Enforcement (BCE) Markers
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor2, posBottom, QUNIT, sizeBottom);
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, bin, posTop, QUNIT, sizeBottom);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor2, pos_xp, QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor2, pos_xn, QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor2, pos_yp, QUNIT, size_XZ, 13);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor2, pos_yn, QUNIT, size_XZ, 13);

    /// Create Rover
    {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./Rover_body.obj";
        double scale_ratio = 0.02;
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

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, -3.14/2));
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ);
        ChVector<> Body_vel = ChVector<>(-0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        Body->SetMass(mmass * mdensity);
        Body->SetInertiaXX(mdensity * principal_I);
        // Body->SetPos(Body_pos);
        Body->SetPos_dt(Body_vel);
        // Body->SetRot(QUNIT);
        
		// Set the absolute position of the body:
        Body->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos),ChQuaternion<>(Body_rot)));                              
        mphysicalSystem.Add(Body);

        Body->SetBodyFixed(false);
        Body->GetCollisionModel()->ClearModel();
        Body->GetCollisionModel()->AddTriangleMesh(mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Body->GetCollisionModel()->BuildModel();
        Body->SetCollide(false);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Body->AddAsset(masset_mesh);

        // Add this body to the FSI system
        // myFsiSystem.AddFsiBody(Body);
        // std::string BCE_path = "./BCE_XXX.txt";
        // fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Body, BCE_path, 
        //                             ChVector<double>(0), QUNIT, scale_ratio);
    }

    /// Create wheels
    ChVector<> Body_Rel_pos_lf = ChVector<>(-1.5, -1.15, 0.4);
    ChVector<> Body_Rel_pos_rf = ChVector<>(-1.5,  1.25, 0.4);
    ChVector<> Body_Rel_pos_lb = ChVector<>( 1.5, -1.15, 0.4);
    ChVector<> Body_Rel_pos_rb = ChVector<>( 1.5,  1.25, 0.4);
    ChVector<> Body_Rel_pos_lf_m = ChVector<>(-0.5, -1.15, 0.4);
    ChVector<> Body_Rel_pos_rf_m = ChVector<>(-0.5,  1.25, 0.4);
    ChVector<> Body_Rel_pos_lb_m = ChVector<>( 0.5, -1.15, 0.4);
    ChVector<> Body_Rel_pos_rb_m = ChVector<>( 0.5,  1.25, 0.4);
    {
    for (int i = 0; i < 8; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./M2020_WHEEL_DESKTOP_MODEL-5inch.obj";
        ChVector<> center_mesh = ChVector<> (0.8941, -0.7271, 0.8977);
        double scale_ratio = 0.1;
        mmesh->LoadWavefrontMesh(obj_path, false, true);
        mmesh->Transform(center_mesh, ChMatrix33<>(1.0));       // rotate the mesh if needed
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

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 0));
        if(i==0 || i==2 || i==4 || i==6){Body_rot = Q_from_Euler123(ChVector<double>(0, 0, CH_C_PI));}
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = Body_Rel_pos_lf;}
        if(i==1){Body_Rel_pos = Body_Rel_pos_rf;}
        if(i==2){Body_Rel_pos = Body_Rel_pos_lb;}
        if(i==3){Body_Rel_pos = Body_Rel_pos_rb;}
        if(i==4){Body_Rel_pos = Body_Rel_pos_lf_m;}
        if(i==5){Body_Rel_pos = Body_Rel_pos_rf_m;}
        if(i==6){Body_Rel_pos = Body_Rel_pos_lb_m;}
        if(i==7){Body_Rel_pos = Body_Rel_pos_rb_m;}
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(-0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        Body->SetMass(mmass * mdensity);
        Body->SetInertiaXX(mdensity * principal_I);
        // Body->SetPos(Body_pos);
        Body->SetPos_dt(Body_vel);
        // Body->SetRot(QUNIT);
        
		// Set the absolute position of the body:
        Body->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos),ChQuaternion<>(Body_rot)));                              
        mphysicalSystem.Add(Body);

        Body->SetBodyFixed(false);
        Body->GetCollisionModel()->ClearModel();
        Body->GetCollisionModel()->AddTriangleMesh(mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Body->GetCollisionModel()->BuildModel();
        Body->SetCollide(true);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Body->AddAsset(masset_mesh);

        // Add this body to the FSI system
        myFsiSystem.AddFsiBody(Body);
        std::string BCE_path = "./sphParticles_wheel_shift.txt";
        fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Body, BCE_path, 
                                    ChVector<double>(0), QUNIT, scale_ratio);
    }
    }

    // Create joint constraints.
    // All joint frames are specified in the global frame.
    // Define a quaternion representing:
    // a rotation of -90 degrees around x (z2y)
    ChQuaternion<> z2y;
    z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));

    // Revolute joint between rover and wheels.
    // The rotational axis of a revolute joint is along the Z axis of the
    // specified joint coordinate frame.  Here, we apply the 'z2y' rotation to
    // align it with the Y axis of the global reference frame.
    auto Rover_body = mphysicalSystem.Get_bodylist()[1];
    ChVector<> Link_pos;
    ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
    
    // outside 4 wheels
    auto revolute_wheel_lf_rover = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_lf = mphysicalSystem.Get_bodylist()[2];
    Link_pos = Rover_Body_pos + Body_Rel_pos_lf;
    revolute_wheel_lf_rover->Initialize(Rover_body, wheel_lf, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_lf_rover);

    auto revolute_wheel_rf_rover = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_rf = mphysicalSystem.Get_bodylist()[3];
    Link_pos = Rover_Body_pos + Body_Rel_pos_rf;
    revolute_wheel_rf_rover->Initialize(Rover_body, wheel_rf, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_rf_rover);

    auto revolute_wheel_lb_rover = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_lb = mphysicalSystem.Get_bodylist()[4];
    Link_pos = Rover_Body_pos + Body_Rel_pos_lb;
    revolute_wheel_lb_rover->Initialize(Rover_body, wheel_lb, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_lb_rover);

    auto revolute_wheel_rb_rover = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_rb = mphysicalSystem.Get_bodylist()[5];
    Link_pos = Rover_Body_pos + Body_Rel_pos_rb;
    revolute_wheel_rb_rover->Initialize(Rover_body, wheel_rb, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_rb_rover);
    
    // inside 4 wheels
    auto revolute_wheel_lf_rover_m = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_lf_m = mphysicalSystem.Get_bodylist()[6];
    Link_pos = Rover_Body_pos + Body_Rel_pos_lf_m;
    revolute_wheel_lf_rover_m->Initialize(Rover_body, wheel_lf_m, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_lf_rover_m);

    auto revolute_wheel_rf_rover_m = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_rf_m = mphysicalSystem.Get_bodylist()[7];
    Link_pos = Rover_Body_pos + Body_Rel_pos_rf_m;
    revolute_wheel_rf_rover_m->Initialize(Rover_body, wheel_rf_m, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_rf_rover_m);

    auto revolute_wheel_lb_rover_m = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_lb_m = mphysicalSystem.Get_bodylist()[8];
    Link_pos = Rover_Body_pos + Body_Rel_pos_lb_m;
    revolute_wheel_lb_rover_m->Initialize(Rover_body, wheel_lb_m, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_lb_rover_m);

    auto revolute_wheel_rb_rover_m = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    auto wheel_rb_m = mphysicalSystem.Get_bodylist()[9];
    Link_pos = Rover_Body_pos + Body_Rel_pos_rb_m;
    revolute_wheel_rb_rover_m->Initialize(Rover_body, wheel_rb_m, ChFrame<>(Link_pos, z2y));
    mphysicalSystem.AddLink(revolute_wheel_rb_rover_m);
    
    auto my_speed_function = chrono_types::make_shared<ChFunction_Const>(CH_C_PI/2.0);  // speed w=3.145 rad/sec
    revolute_wheel_lf_rover->SetSpeedFunction(my_speed_function);
    revolute_wheel_rf_rover->SetSpeedFunction(my_speed_function);
    revolute_wheel_lb_rover->SetSpeedFunction(my_speed_function);
    revolute_wheel_rb_rover->SetSpeedFunction(my_speed_function);
    revolute_wheel_lf_rover_m->SetSpeedFunction(my_speed_function);
    revolute_wheel_rf_rover_m->SetSpeedFunction(my_speed_function);
    revolute_wheel_lb_rover_m->SetSpeedFunction(my_speed_function);
    revolute_wheel_rb_rover_m->SetSpeedFunction(my_speed_function);


    double FSI_MASS = myFsiSystem.GetDataManager()->numObjects->numRigid_SphMarkers * paramsH->markerMass;
    // printf("inertia=%f,%f,%f\n", mass * gyration.x(), mass * gyration.y(), mass * gyration.z());
    // printf("\nreal mass=%f, FSI_MASS=%f\n\n", mass, FSI_MASS);

}
//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemNSC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int next_frame,
                       double mTime) {
    int out_steps = (int)ceil((1.0 / paramsH->dT) / paramsH->out_fps);
    int num_contacts = mphysicalSystem.GetNcontacts();
    double frame_time = 1.0 / paramsH->out_fps;
    static int out_frame = 0;

    if (pv_output && std::abs(mTime - (next_frame)*frame_time) < 1e-7) {
        fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2->posRadD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->velMasD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->rhoPresMuD,
                                myFsiSystem.GetDataManager()->fsiGeneralData->sr_tau_I_mu_i,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray_FEA, demo_dir, true);


        // save the rover body to obj/vtk files
        for (int i = 0; i < 1; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+1];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./Rover_body.obj";
            double scale_ratio = 0.02;
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
            mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

            double mmass;
            ChVector<> mcog;
            ChMatrix33<> minertia;
            mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
            mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

            char filename[2048];
            if(1==0){// save to obj file
                sprintf(filename, "%s/Rover_body_%zd.obj", paramsH->demo_dir, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/Rover_body_%zd.vtk", paramsH->demo_dir, next_frame);
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
        // save the wheels to obj/vtk files
        for (int i = 0; i < 8; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+2];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./M2020_WHEEL_DESKTOP_MODEL-5inch.obj";
            ChVector<> center_mesh = ChVector<> (0.8941, -0.7271, 0.8977);
            double scale_ratio = 0.1;
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            mmesh->Transform(center_mesh, ChMatrix33<>(1.0));       // rotate the mesh if needed
            mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
            mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

            double mmass;
            ChVector<> mcog;
            ChMatrix33<> minertia;
            mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
            mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

            char filename[2048];
            if(1==0){// save to obj file
                sprintf(filename, "%s/wheel_%zd_%zd.obj", paramsH->demo_dir, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/wheel_%zd_%zd.vtk", paramsH->demo_dir, i+1, next_frame);
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

        for (int i = 1; i < mphysicalSystem.Get_bodylist().size(); i++) {
            auto body = mphysicalSystem.Get_bodylist()[i];
            ChFrame<> ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> pos = ref_frame.GetPos();
            ChQuaternion<> rot = ref_frame.GetRot();
            ChVector<> vel = body->GetPos_dt();

            char filename[2048];
            std::string delim = ",";
            sprintf(filename, "%s/body_pos_rot_vel%zd.csv", paramsH->demo_dir, i);
            std::ofstream file;
            if (mphysicalSystem.GetChTime() > 0)
                file.open(filename, std::fstream::app);
            else {
                file.open(filename);
                file << "Time" << delim << "x" << delim << "y" << delim << "z" << delim << "q0" << delim << "q1" << delim
                    << "q2" << delim << "q3" << delim << "Vx" << delim << "Vy" << delim << "Vz" << std::endl;
            }

            file << mphysicalSystem.GetChTime() << delim << pos.x() << delim << pos.y() << delim << pos.z() << delim
                << rot.e0() << delim << rot.e1() << delim << rot.e2() << delim << rot.e3() << delim << vel.x() << delim
                << vel.y() << delim << vel.z() << std::endl;

            file.close();
        }

        cout << "-------------------------------------\n" << endl;
        cout << "             Output frame:   " << next_frame << endl;
        cout << "             Time:           " << mTime << endl;
        cout << "-------------------------------------\n" << endl;

        out_frame++;
    }
}

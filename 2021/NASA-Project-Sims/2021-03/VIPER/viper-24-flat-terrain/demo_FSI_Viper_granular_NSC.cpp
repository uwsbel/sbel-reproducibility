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
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/physics/ChLinkDistance.h"
// #include "chrono/physics/ChLinkSpring.h"
// #include "chrono/physics/ChLinkTSDA.h"

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
const std::string out_dir = GetChronoOutputPath() + "FSI_Viper/";
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

int motor_type; // 1 means constant rotation speed, 2 means constant torque
double motor_F; // if motor_type==1, this means the rotation speed, if motor_type==2, this means the torque
double obstacle_density; // density of the obstacle

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/// Forward declaration of helper functions
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemNSC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int tStep,
                       double mTime);

void AddWall(std::shared_ptr<ChBody> body, const ChVector<>& dim, std::shared_ptr<ChMaterialSurface> mat,const ChVector<>& loc) {
    body->GetCollisionModel()->AddBox(mat, dim.x(), dim.y(), dim.z(), loc);
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
    cout << "usage: ./demo_FSI_Granular_Viper <json_file>" << endl;
}

// =============================================================================
int main(int argc, char* argv[]) {
    // create a physics system
    ChSystemNSC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(mphysicalSystem);
    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_Viper_granular.json");
    // if (argc == 1 && fsi::utils::ParseJSON(inputJson.c_str(), paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
    // } else if (argc == 2 && fsi::utils::ParseJSON(argv[1], paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
    // } else {
    //     ShowUsage();
    //     return 1;
    // }
    if (argc == 5 && fsi::utils::ParseJSON(argv[1], paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
    } else {
        ShowUsage();
        return 1;
    }
    motor_type    = std::stoi(argv[2]); 
    motor_F       = std::stod(argv[3]); 
    obstacle_density  = std::stod(argv[4]); 

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
    paramsH->cMin = fsi::mR3(-10.0 * bxDim / 2, -byDim / 2 - 0.5 * initSpace0, -10.0 * bzDim - 10 * initSpace0) * 1;
    paramsH->cMax = fsi::mR3( 10.0 * bxDim / 2,  byDim / 2 + 0.5 * initSpace0,  10.0 * bzDim + 10 * initSpace0) * 1;
    // call FinalizeDomain to setup the binning for neighbor search or write your own
    fsi::utils::FinalizeDomain(paramsH);
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
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(0, (int)numPart, -1, -1));
        myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(numPart, (int)numPart, 0, 0));
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
    myFile.open("./body_position.txt", std::ios::trunc);
    myFile.close();
    myFile.open("./body_position.txt", std::ios::app);
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
        myFile.open("./body_position.txt", std::ios::app);
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

    // Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    // mysurfmaterial->SetYoungModulus(1e8);
    // mysurfmaterial->SetFriction(0.5f);
    // mysurfmaterial->SetRestitution(0.05f);
    // mysurfmaterial->SetAdhesion(0);

	collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0025);
	collision::ChCollisionModel::SetDefaultSuggestedMargin(0.0025);

    auto mfloor = chrono_types::make_shared<ChBodyEasyBox>(50, 50, 0.02, 1000, false, true, mysurfmaterial);
    mfloor->SetPos(ChVector<>(0, 0, 0));
    mfloor->SetBodyFixed(true);
    mphysicalSystem.Add(mfloor);

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

    /// Fluid-Solid Coupling at the walls via Condition Enforcement (BCE) Markers
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_zp, QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_zn, QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_xp, QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_xn, QUNIT, size_YZ, 23);
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_yp, QUNIT, size_XZ, 13);
    // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, mfloor, pos_yn, QUNIT, size_XZ, 13);

    /// Create the Viper body
    {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./body.obj";
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
        // mmass = 0.23;
        mcog = ChVector<>(0, 0, 0);
        // minertia = ChMatrix33<>(1.0);
        ChMatrix33<> principal_inertia_rot;
        ChVector<> principal_I;
        ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, CH_C_PI/2));
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
        Body->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh, false, false, VNULL, ChMatrix33<>(1), 0.005);
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
    double w_lx = 0.5618 + 0.08;
    double w_ly = 0.2067 + 0.32 + 0.0831; 
    double w_lz = 0.0;
    ChVector<> wheel_rel_pos_lf = ChVector<>(-w_lx, -w_ly, w_lz);
    ChVector<> wheel_rel_pos_rf = ChVector<>(-w_lx,  w_ly, w_lz);
    ChVector<> wheel_rel_pos_lb = ChVector<>( w_lx, -w_ly, w_lz);
    ChVector<> wheel_rel_pos_rb = ChVector<>( w_lx,  w_ly, w_lz);
    {
    for (int i = 0; i < 4; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./wheelSimplified.obj";
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

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, 0));
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = wheel_rel_pos_lf;}
        if(i==1){Body_Rel_pos = wheel_rel_pos_rf;}
        if(i==2){Body_Rel_pos = wheel_rel_pos_lb;}
        if(i==3){Body_Rel_pos = wheel_rel_pos_rb;}
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

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
        Body->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Body->GetCollisionModel()->BuildModel();
        Body->SetCollide(true);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Body->AddAsset(masset_mesh);

        // Add this body to the FSI system
        myFsiSystem.AddFsiBody(Body);
        std::string BCE_path = "./BCE_simplifiedWheel_low.txt";
        fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Body, BCE_path, 
                                    ChVector<double>(0), QUNIT, scale_ratio);
    }
    }

    /// Create conecting rod
    double cr_lx = 0.5618 + 0.08;
    double cr_ly = 0.2067; // + 0.32/2; 
    double cr_lz = 0.0525;
    ChVector<> cr_rel_pos_lf_lower = ChVector<>(-cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rf_lower = ChVector<>(-cr_lx,  cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lb_lower = ChVector<>( cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rb_lower = ChVector<>( cr_lx,  cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lf_upper = ChVector<>(-cr_lx, -cr_ly,  cr_lz);
    ChVector<> cr_rel_pos_rf_upper = ChVector<>(-cr_lx,  cr_ly,  cr_lz);
    ChVector<> cr_rel_pos_lb_upper = ChVector<>( cr_lx, -cr_ly,  cr_lz);
    ChVector<> cr_rel_pos_rb_upper = ChVector<>( cr_lx,  cr_ly,  cr_lz);
    {
    for (int i = 0; i < 8; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path;
        if(i < 4){
            obj_path = "./lowerRod.obj";
        }
        else{
            obj_path = "./upperRod.obj";
        }
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

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot;
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = cr_rel_pos_lf_lower; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0, 0));}
        if(i==1){Body_Rel_pos = cr_rel_pos_rf_lower; Body_rot = Q_from_Euler123(ChVector<double>(-CH_C_PI/2, 0, 0));}
        if(i==2){Body_Rel_pos = cr_rel_pos_lb_lower; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0, 0));}
        if(i==3){Body_Rel_pos = cr_rel_pos_rb_lower; Body_rot = Q_from_Euler123(ChVector<double>(-CH_C_PI/2, 0, 0));}
        if(i==4){Body_Rel_pos = cr_rel_pos_lf_upper; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0, 0));}
        if(i==5){Body_Rel_pos = cr_rel_pos_rf_upper; Body_rot = Q_from_Euler123(ChVector<double>(-CH_C_PI/2, 0, 0));}
        if(i==6){Body_Rel_pos = cr_rel_pos_lb_upper; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0, 0));}
        if(i==7){Body_Rel_pos = cr_rel_pos_rb_upper; Body_rot = Q_from_Euler123(ChVector<double>(-CH_C_PI/2, 0, 0));}
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

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
        Body->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Body->GetCollisionModel()->BuildModel();
        Body->SetCollide(false);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Body->AddAsset(masset_mesh);
    }
    }

    /// Create steering rod on wheels
    double sr_lx = 0.5618 + 0.08;
    double sr_ly = 0.2067 + 0.32 + 0.0831; 
    double sr_lz = 0.0;
    double sr_ly_joint = 0.2067 + 0.32;
    ChVector<> sr_rel_pos_lf = ChVector<>(-sr_lx, -sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rf = ChVector<>(-sr_lx,  sr_ly, sr_lz);
    ChVector<> sr_rel_pos_lb = ChVector<>( sr_lx, -sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rb = ChVector<>( sr_lx,  sr_ly, sr_lz);
    ChVector<> sr_rel_pos_lf_lower = ChVector<>(-sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rf_lower = ChVector<>(-sr_lx,  sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lb_lower = ChVector<>( sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rb_lower = ChVector<>( sr_lx,  sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lf_upper = ChVector<>(-sr_lx, -sr_ly_joint,  cr_lz);
    ChVector<> sr_rel_pos_rf_upper = ChVector<>(-sr_lx,  sr_ly_joint,  cr_lz);
    ChVector<> sr_rel_pos_lb_upper = ChVector<>( sr_lx, -sr_ly_joint,  cr_lz);
    ChVector<> sr_rel_pos_rb_upper = ChVector<>( sr_lx,  sr_ly_joint,  cr_lz);
    {
    for (int i = 0; i < 4; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./steeringRod.obj";
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

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot;
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = sr_rel_pos_lf; Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, 0)); }
        if(i==1){Body_Rel_pos = sr_rel_pos_rf; Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, CH_C_PI)); }
        if(i==2){Body_Rel_pos = sr_rel_pos_lb; Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, 0)); }
        if(i==3){Body_Rel_pos = sr_rel_pos_rb; Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, CH_C_PI)); }
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

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
        Body->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Body->GetCollisionModel()->BuildModel();
        Body->SetCollide(false);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Body->AddAsset(masset_mesh);
    }
    }

    /// Create obstacles
    {
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            ChVector<> Body_pos = ChVector<>(1.2 - 2.0*i - 2.4*j, -0.6 + 1.2*j, paramsH->bodyIniPosZ + paramsH->fluidDimZ);
            ChVector<> Body_size(paramsH->bodyDimX, paramsH->bodyDimY, paramsH->bodyDimZ);
            double Body_density = paramsH->rho0 * obstacle_density;
            double Body_volume = paramsH->bodyDimX * paramsH->bodyDimY * paramsH->bodyDimZ;
            double mass = Body_density * Body_volume;
            ChMatrix33<> gyration = utils::CalcBoxGyration(Body_size);
            auto Body = chrono_types::make_shared<ChBodyEasyBox>(Body_size.x(),Body_size.y(),Body_size.z(), Body_density, false, true, mysurfmaterial);
            Body->SetMass(mass);
            Body->SetInertia(gyration * mass);
            Body->SetPos(Body_pos);
            mphysicalSystem.Add(Body);
            Body->SetBodyFixed(true);
            Body->SetCollide(false);

            // Add this body to the FSI system
            // myFsiSystem.AddFsiBody(Body);
            // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, Body, ChVector<double>(0), QUNIT, 
            //                     Body_size*0.5, 123, true, false);
        }
    }
    }

    /// Create upper floor
    /*{
        auto upperfloor = chrono_types::make_shared<ChBodyEasyBox>(50, 50, 0.02, 1000, false, true, mysurfmaterial);
        upperfloor->SetPos(ChVector<>(0, 0, 0.6));
        upperfloor->SetBodyFixed(true);
        mphysicalSystem.Add(upperfloor);
    }*/

    /// Create joint constraints.
    for (int i = 0; i < 4; i++) {
        // define a rotation of 90 degrees around y (z2x)
        // because all revolute jonts are along Z direction of the joint frame,
        // so we apply the 'z2x' rotation to align the rotation axis with the X axis of the global frame.
        ChQuaternion<> z2x;
        z2x.Q_from_AngAxis(CH_C_PI / 2, ChVector<>(0, 1, 0));

        // pick up relative position of wheel and conecting rod to the body
        ChVector<> Wheel_Rel_pos;
        ChVector<> CR_Rel_pos_Lower;
        ChVector<> CR_Rel_pos_Upper;
        ChVector<> SR_Rel_pos;
        ChVector<> SR_Rel_pos_Lower;
        ChVector<> SR_Rel_pos_Upper;
        if(i==0){
            Wheel_Rel_pos = wheel_rel_pos_lf; 
            CR_Rel_pos_Lower = cr_rel_pos_lf_lower;
            CR_Rel_pos_Upper = cr_rel_pos_lf_upper;
            SR_Rel_pos = sr_rel_pos_lf;
            SR_Rel_pos_Lower = sr_rel_pos_lf_lower;
            SR_Rel_pos_Upper = sr_rel_pos_lf_upper;
        }
        if(i==1){
            Wheel_Rel_pos = wheel_rel_pos_rf; 
            CR_Rel_pos_Lower = cr_rel_pos_rf_lower;
            CR_Rel_pos_Upper = cr_rel_pos_rf_upper;
            SR_Rel_pos = sr_rel_pos_rf;
            SR_Rel_pos_Lower = sr_rel_pos_rf_lower;
            SR_Rel_pos_Upper = sr_rel_pos_rf_upper;
        }
        if(i==2){
            Wheel_Rel_pos = wheel_rel_pos_lb; 
            CR_Rel_pos_Lower = cr_rel_pos_lb_lower;
            CR_Rel_pos_Upper = cr_rel_pos_lb_upper;
            SR_Rel_pos = sr_rel_pos_lb;
            SR_Rel_pos_Lower = sr_rel_pos_lb_lower;
            SR_Rel_pos_Upper = sr_rel_pos_lb_upper;
        }
        if(i==3){
            Wheel_Rel_pos = wheel_rel_pos_rb; 
            CR_Rel_pos_Lower = cr_rel_pos_rb_lower;
            CR_Rel_pos_Upper = cr_rel_pos_rb_upper;
            SR_Rel_pos = sr_rel_pos_rb;
            SR_Rel_pos_Lower = sr_rel_pos_rb_lower;
            SR_Rel_pos_Upper = sr_rel_pos_rb_upper;
        }

        // pick up bodies and create links
        auto ground = mphysicalSystem.Get_bodylist()[0];
        auto body = mphysicalSystem.Get_bodylist()[1];
        auto wheel = mphysicalSystem.Get_bodylist()[i+2];
        auto cr_lower = mphysicalSystem.Get_bodylist()[i+2+4];
        auto cr_upper = mphysicalSystem.Get_bodylist()[i+2+8];
        auto steering_rod = mphysicalSystem.Get_bodylist()[i+2+12];
        ChVector<> Link_pos;
        ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);

        // body-cr_lower Revolute constraint
        auto revo_body_cr_lower = chrono_types::make_shared<ChLinkLockRevolute>(); 
        Link_pos = Rover_Body_pos + CR_Rel_pos_Lower;
        revo_body_cr_lower->Initialize(body, cr_lower, ChCoordsys<>(Link_pos, z2x));
        mphysicalSystem.AddLink(revo_body_cr_lower);

        // body-cr_upper Revolute constraint
        auto revo_body_cr_upper = chrono_types::make_shared<ChLinkLockRevolute>(); 
        Link_pos = Rover_Body_pos + CR_Rel_pos_Upper;
        revo_body_cr_upper->Initialize(body, cr_upper, ChCoordsys<>(Link_pos, z2x));
        mphysicalSystem.AddLink(revo_body_cr_upper);

        // cr_lower-sr_lower Revolute constraint
        auto revo_cr_sr_lower = chrono_types::make_shared<ChLinkLockRevolute>(); 
        Link_pos = Rover_Body_pos + SR_Rel_pos_Lower;
        revo_cr_sr_lower->Initialize(cr_lower, steering_rod, ChCoordsys<>(Link_pos, z2x));
        mphysicalSystem.AddLink(revo_cr_sr_lower);

        // cr_upper-sr_upper Revolute constraint
        auto revo_cr_sr_upper = chrono_types::make_shared<ChLinkLockRevolute>(); 
        Link_pos = Rover_Body_pos + SR_Rel_pos_Upper;
        revo_cr_sr_upper->Initialize(cr_upper, steering_rod, ChCoordsys<>(Link_pos, z2x));
        mphysicalSystem.AddLink(revo_cr_sr_upper);

        // define a rotation of -90 degrees around x (z2y)
        // because all revolute jonts are along Z direction of the joint frame,
        // so we apply the 'z2y' rotation to align the rotation axis with the Y axis of the global frame.
        ChQuaternion<> z2y;
        z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));
        if(motor_type == 1){
            // sr-wheel Revolute constraint with a motor - Rotation Speed
            auto revo_sr_wheel = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
            Link_pos = Rover_Body_pos + Wheel_Rel_pos;
            revo_sr_wheel->Initialize(steering_rod, wheel, ChFrame<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_sr_wheel);
            auto my_speed_function = chrono_types::make_shared<ChFunction_Const>(CH_C_PI*motor_F);  // speed w=3.145 rad/sec
            revo_sr_wheel->SetSpeedFunction(my_speed_function);
        }
        else if(motor_type == 2){
            // sr-wheel Revolute constraint with a motor - torque
            std::shared_ptr<ChLinkMotorRotationTorque> link_motor;
            link_motor = chrono_types::make_shared<ChLinkMotorRotationTorque>();
            Link_pos = Rover_Body_pos + Wheel_Rel_pos;
            link_motor->Initialize(steering_rod, wheel, ChFrame<>(Link_pos, z2y));
            mphysicalSystem.AddLink(link_motor);
            auto mfun = std::dynamic_pointer_cast<ChFunction_Const>(link_motor->GetTorqueFunction());
            mfun->Set_yconst(motor_F);
        }

        // body-steering_rod spring constraint
        std::shared_ptr<ChLinkTSDA> spring;
        ChVector<> pos1 = Rover_Body_pos + CR_Rel_pos_Upper;
        ChVector<> pos2 = Rover_Body_pos + SR_Rel_pos_Lower;
        spring = chrono_types::make_shared<ChLinkTSDA>();
        spring->Initialize(body, steering_rod, false, pos1, pos2, true, 0.0);
        spring->SetSpringCoefficient(400000.0);
        spring->SetDampingCoefficient(10000.0);
        mphysicalSystem.AddLink(spring);

        // double f1 = spring->GetForce();
        // std::cout << f1 << std::endl;
        // std::string dummy;
        // std::getline(std::cin, dummy);
    }


    // double FSI_MASS = myFsiSystem.GetDataManager()->numObjects->numRigid_SphMarkers * paramsH->markerMass;
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
        // save the SPH particles
        fsi::utils::PrintToFile(myFsiSystem.GetDataManager()->sphMarkersD2->posRadD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->velMasD,
                                myFsiSystem.GetDataManager()->sphMarkersD2->rhoPresMuD,
                                myFsiSystem.GetDataManager()->fsiGeneralData->sr_tau_I_mu_i,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray,
                                myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray_FEA, demo_dir, true);

        // save the viper body to obj/vtk files
        for (int i = 0; i < 1; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+1];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./body.obj";
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
                sprintf(filename, "%s/body_%d.obj", paramsH->demo_dir, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/body_%d.vtk", paramsH->demo_dir, next_frame);
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
        for (int i = 0; i < 4; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+2];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./wheelSimplified.obj";
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
                sprintf(filename, "%s/wheel_%d_%d.obj", paramsH->demo_dir, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/wheel_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
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
        // save the steering rod to obj/vtk files
        for (int i = 0; i < 4; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+2+12];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./steeringRod.obj";
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
                sprintf(filename, "%s/steerRod_%d_%d.obj", paramsH->demo_dir, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/steerRod_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
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
        // save the lower rod to obj/vtk files
        for (int i = 0; i < 4; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+2+4];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./lowerRod.obj";
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
                sprintf(filename, "%s/lowerRod_%d_%d.obj", paramsH->demo_dir, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/lowerRod_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
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
        // save the upper rod to obj/vtk files
        for (int i = 0; i < 4; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+2+8];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./upperRod.obj";
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
                sprintf(filename, "%s/upperRod_%d_%d.obj", paramsH->demo_dir, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s/upperRod_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
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
        // save box obstacle to vtk files
        for (int i = 0; i < 2; i++) {
            char filename[4096];
            sprintf(filename, "%s/obstacle_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
            std::ofstream file;
            file.open(filename);
            file << "# vtk DataFile Version 2.0" << std::endl;
            file << "VTK from simulation" << std::endl;
            file << "ASCII" << std::endl;
            file << "DATASET POLYDATA" << std::endl;

            file << "POINTS " << 8 << " " << "float" << std::endl;
            auto Body = mphysicalSystem.Get_bodylist()[i+2+16];
            ChVector<> center = Body->GetPos();
            ChMatrix33<> Rotation = Body->GetRot();
            double lx = paramsH->bodyDimX*0.5;
            double ly = paramsH->bodyDimY*0.5;
            double lz = paramsH->bodyDimZ*0.5;
            ChVector<double> Node1 = ChVector<double>(-lx, -ly, -lz);
            ChVector<double> Node2 = ChVector<double>( lx, -ly, -lz);
            ChVector<double> Node3 = ChVector<double>( lx, -ly,  lz);
            ChVector<double> Node4 = ChVector<double>(-lx, -ly,  lz);
            ChVector<double> Node5 = ChVector<double>(-lx,  ly, -lz);
            ChVector<double> Node6 = ChVector<double>( lx,  ly, -lz);
            ChVector<double> Node7 = ChVector<double>( lx,  ly,  lz);
            ChVector<double> Node8 = ChVector<double>(-lx,  ly,  lz);
            ChVector<double> vertex1 = Rotation * Node1 + center;  
            ChVector<double> vertex2 = Rotation * Node2 + center; 
            ChVector<double> vertex3 = Rotation * Node3 + center; 
            ChVector<double> vertex4 = Rotation * Node4 + center; 
            ChVector<double> vertex5 = Rotation * Node5 + center; 
            ChVector<double> vertex6 = Rotation * Node6 + center; 
            ChVector<double> vertex7 = Rotation * Node7 + center; 
            ChVector<double> vertex8 = Rotation * Node8 + center; 
            file << vertex1.x() << " " << vertex1.y() << " " << vertex1.z() << "\n";
            file << vertex2.x() << " " << vertex2.y() << " " << vertex2.z() << "\n";
            file << vertex3.x() << " " << vertex3.y() << " " << vertex3.z() << "\n";
            file << vertex4.x() << " " << vertex4.y() << " " << vertex4.z() << "\n";
            file << vertex5.x() << " " << vertex5.y() << " " << vertex5.z() << "\n";
            file << vertex6.x() << " " << vertex6.y() << " " << vertex6.z() << "\n";
            file << vertex7.x() << " " << vertex7.y() << " " << vertex7.z() << "\n";
            file << vertex8.x() << " " << vertex8.y() << " " << vertex8.z() << "\n";

            file << "POLYGONS " << 12 << " " << 48 << std::endl;
            file << "3 " << 0 << " " << 1 << " " << 2 << "\n";
            file << "3 " << 0 << " " << 1 << " " << 5 << "\n";
            file << "3 " << 0 << " " << 4 << " " << 7 << "\n";
            file << "3 " << 4 << " " << 5 << " " << 6 << "\n";
            file << "3 " << 1 << " " << 5 << " " << 6 << "\n";
            file << "3 " << 3 << " " << 2 << " " << 6 << "\n";

            file << "3 " << 0 << " " << 2 << " " << 3 << "\n";
            file << "3 " << 0 << " " << 5 << " " << 4 << "\n";
            file << "3 " << 0 << " " << 7 << " " << 3 << "\n";
            file << "3 " << 4 << " " << 6 << " " << 7 << "\n";
            file << "3 " << 1 << " " << 6 << " " << 2 << "\n";
            file << "3 " << 3 << " " << 6 << " " << 7 << "\n";
        }
        // save rigid body position and rotation
        for (int i = 1; i < mphysicalSystem.Get_bodylist().size(); i++) {
            auto body = mphysicalSystem.Get_bodylist()[i];
            ChFrame<> ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> pos = ref_frame.GetPos();
            ChQuaternion<> rot = ref_frame.GetRot();
            ChVector<> vel = body->GetPos_dt();

            char filename[4096];
            std::string delim = ",";
            sprintf(filename, "%s/body_pos_rot_vel%d.csv", paramsH->demo_dir, i);
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

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

// Chrono fsi includes
#include "chrono_fsi/utils/ChUtilsTypeConvert.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsJSON.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"


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
const std::string out_dir = GetChronoOutputPath() + "FSI_Curiosity/";
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

std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_lf; 
std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_rf; 
std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_lm; 
std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_rm; 
std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_lb; 
std::shared_ptr<ChLinkMotorRotationSpeed> link_motor_speed_rb; 

std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_lf; 
std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_rf; 
std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_lm; 
std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_rm; 
std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_lb; 
std::shared_ptr<ChLinkMotorRotationTorque> link_motor_torque_rb; 

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
/// Forward declaration of helper functions
void SaveParaViewFiles(fsi::ChSystemFsi& myFsiSystem,
                       ChSystemNSC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int tStep,
                       double mTime);

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
// Create the objects of the MBD system. Rigid bodies, and if fsi, their
// bce representation are created and added to the systems
//------------------------------------------------------------------

void CreateSolidPhase(ChSystemNSC& mphysicalSystem,
                      fsi::ChSystemFsi& myFsiSystem,
                      std::shared_ptr<fsi::SimParams> paramsH);
void ShowUsage() {
    cout << "usage: ./demo_FSI_Granular_Curiosity <json_file>" << endl;
}

// =============================================================================
int main(int argc, char* argv[]) {
    // create a physics system
    ChSystemNSC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(mphysicalSystem);

    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
    // std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_Curiosity_granular.json");
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

    /// Get the body from the FSI system
    std::vector<std::shared_ptr<ChBody>>& FSI_Bodies = myFsiSystem.GetFsiBodies();
    auto Rover = FSI_Bodies[0];
    SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, 0, 0);

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

        auto Rover = mphysicalSystem.Get_bodylist()[1];
        ChFrame<> ref_frame = Rover->GetFrame_REF_to_abs();
        ChVector<> pos = ref_frame.GetPos();
        ChQuaternion<> rot = ref_frame.GetRot();
        ChVector<> vel = Rover->GetPos_dt();
        printf("Rover Pos =%f,%f,%f\n", pos.x(), pos.y(), pos.z());
        printf("Rover Vel =%f,%f,%f\n", vel.x(), vel.y(), vel.z());
        printf("Rover Rot =%f,%f,%f,%f\n", rot.e0(), rot.e1(), rot.e2(), rot.e3());
        
        // adaptively changing the velocity of the wheel
        /*double radius_wheel = 0.25;
        double vel_xy_rover = sqrt(vel.x()*vel.x() + vel.y()*vel.y());
        for (int i = 0; i <6; i++){
            auto wheel = mphysicalSystem.Get_bodylist()[2+i];
            ChFrame<> ref_frame_wheel = wheel->GetFrame_REF_to_abs();
            ChVector<> wheel_pos = ref_frame_wheel.GetPos();
            ChQuaternion<> wheel_rot = ref_frame_wheel.GetRot();
            ChVector<> wheel_vel = wheel->GetPos_dt();

            double vel_xy = sqrt(wheel_vel.x()*wheel_vel.x() + wheel_vel.y()*wheel_vel.y());
            double vel_ratio = vel_xy_rover/vel_xy;

            double maxmum_ratio = 2.0;
            if(vel_ratio > maxmum_ratio){
                auto my_speed_function_new = chrono_types::make_shared<ChFunction_Const>(CH_C_PI*motor_F*maxmum_ratio); 
                if(i==0){link_motor_speed_lf->SetSpeedFunction(my_speed_function_new);}
                if(i==1){link_motor_speed_rf->SetSpeedFunction(my_speed_function_new);}
                if(i==2){link_motor_speed_lm->SetSpeedFunction(my_speed_function_new);}
                if(i==3){link_motor_speed_rm->SetSpeedFunction(my_speed_function_new);}
                if(i==4){link_motor_speed_lb->SetSpeedFunction(my_speed_function_new);}
                if(i==5){link_motor_speed_rb->SetSpeedFunction(my_speed_function_new);}
            }
            else{
                auto my_speed_function_new = chrono_types::make_shared<ChFunction_Const>(CH_C_PI*motor_F*vel_ratio); 
                if(i==0){link_motor_speed_lf->SetSpeedFunction(my_speed_function_new);}
                if(i==1){link_motor_speed_rf->SetSpeedFunction(my_speed_function_new);}
                if(i==2){link_motor_speed_lm->SetSpeedFunction(my_speed_function_new);}
                if(i==3){link_motor_speed_rm->SetSpeedFunction(my_speed_function_new);}
                if(i==4){link_motor_speed_lb->SetSpeedFunction(my_speed_function_new);}
                if(i==5){link_motor_speed_rb->SetSpeedFunction(my_speed_function_new);}
            }

            printf("wheel %d angvel =%f,%f,%f\n", i+1, wheel->GetWvel_par().x(), wheel->GetWvel_par().y(), wheel->GetWvel_par().z());
        }*/

        if (time > paramsH->tFinal)
            break;
    }

    return 0;
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if fsi, 
// their BCE representation are created and added to the systems
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
    // mysurfmaterial->SetFriction(0.9f);
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

    /// ============================================================================
    /// =============================== body =======================================
    /// ============================================================================

    /// Create the Curiosity body
    {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./Mesh_BCE/body.obj";
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
        mmass = 0.23;
        mcog = ChVector<>(0, 0, 0);
        minertia = ChMatrix33<>(1.0);
        ChMatrix33<> principal_inertia_rot;
        ChVector<> principal_I;
        ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

        // set the abs orientation, position and velocity
        auto Body = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(CH_C_PI/2, 0, -CH_C_PI/2));
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ);
        ChVector<> Body_vel = ChVector<>(-0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        cout << "\n" << "The mass of the rover body is " << mmass * mdensity << "\n" << endl;
        Body->SetMass(mmass * mdensity);//mmass * mdensity
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
    }

    /// Create wheels
    ChVector<> wheel_rel_pos_lf = ChVector<>(-1.095, -1.063, 0.249);
    ChVector<> wheel_rel_pos_rf = ChVector<>(-1.095,  1.063, 0.249);
    ChVector<> wheel_rel_pos_lm = ChVector<>( 0.089, -1.194, 0.249);
    ChVector<> wheel_rel_pos_rm = ChVector<>( 0.089,  1.194, 0.249);
    ChVector<> wheel_rel_pos_lb = ChVector<>( 1.163, -1.063, 0.249);
    ChVector<> wheel_rel_pos_rb = ChVector<>( 1.163,  1.063, 0.249);
    {
    for (int i = 0; i < 6; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./Mesh_BCE/wheelSimplified.obj";
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
        double RotAng;
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = wheel_rel_pos_lf; RotAng = CH_C_PI/2;}
        if(i==1){Body_Rel_pos = wheel_rel_pos_rf; RotAng = CH_C_PI/2;}
        if(i==2){Body_Rel_pos = wheel_rel_pos_lm; RotAng = CH_C_PI/2;}
        if(i==3){Body_Rel_pos = wheel_rel_pos_rm; RotAng = CH_C_PI/2;}
        if(i==4){Body_Rel_pos = wheel_rel_pos_lb; RotAng = CH_C_PI/2;}
        if(i==5){Body_Rel_pos = wheel_rel_pos_rb; RotAng = CH_C_PI/2;}
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(RotAng, 0, 0));
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        cout << "\n" << "The mass of the rover wheel is " << mmass * mdensity << "\n" << endl;
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
        std::string BCE_path = "./Mesh_BCE/BCE_simplifiedWheel_low.txt";
        fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Body, BCE_path, 
                                    ChVector<double>(0), QUNIT, scale_ratio);
        // fsi::utils::AddCylinderBce(myFsiSystem.GetDataManager(), paramsH, Body, ChVector<>(0, 0, 0),
        //                        ChQuaternion<>(Body_rot), 0.25, 0.25, paramsH->HSML, false);
    }
    }

    /// Create connecting rod
    ChVector<> cr_rel_pos_lf = ChVector<>(-0.214, -0.604, 0.8754);
    ChVector<> cr_rel_pos_rf = ChVector<>(-0.214,  0.604, 0.8754);
    ChVector<> cr_rel_pos_lb = ChVector<>( 0.54,  -0.845, 0.6433);
    ChVector<> cr_rel_pos_rb = ChVector<>( 0.54,   0.845, 0.6433);
    {
    for (int i = 0; i < 4; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path;
        if(i == 0){obj_path = "./Mesh_BCE/F_L.obj";}
        if(i == 1){obj_path = "./Mesh_BCE/F_R.obj";}
        if(i == 2){obj_path = "./Mesh_BCE/B_L.obj";}
        if(i == 3){obj_path = "./Mesh_BCE/B_R.obj";}
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
        ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0, -CH_C_PI/2));
        ChVector<> Body_Rel_pos;
        if(i==0){Body_Rel_pos = cr_rel_pos_lf; }
        if(i==1){Body_Rel_pos = cr_rel_pos_rf; }
        if(i==2){Body_Rel_pos = cr_rel_pos_lb; }
        if(i==3){Body_Rel_pos = cr_rel_pos_rb; }
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        cout << "\n" << "The mass of the connecting rod is " << mmass * mdensity << "\n" << endl;
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
    ChVector<> sr_rel_pos_lf = ChVector<>(-1.095, -1.063, 0.64);
    ChVector<> sr_rel_pos_rf = ChVector<>(-1.095,  1.063, 0.64);
    ChVector<> sr_rel_pos_lb = ChVector<>( 1.163, -1.063, 0.64);
    ChVector<> sr_rel_pos_rb = ChVector<>( 1.163,  1.063, 0.64);
    {
    for (int i = 0; i < 4; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path;
        if(i == 0){obj_path = "./Mesh_BCE/ster_front.obj";}
        if(i == 1){obj_path = "./Mesh_BCE/ster_front.obj";}
        if(i == 2){obj_path = "./Mesh_BCE/ster_back.obj";}
        if(i == 3){obj_path = "./Mesh_BCE/ster_back.obj";}
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
        if(i==0){Body_Rel_pos = sr_rel_pos_lf; Body_rot = Q_from_Euler123(ChVector<double>(0, 0,-CH_C_PI/2)); }
        if(i==1){Body_Rel_pos = sr_rel_pos_rf; Body_rot = Q_from_Euler123(ChVector<double>(0, 0, CH_C_PI/2)); }
        if(i==2){Body_Rel_pos = sr_rel_pos_lb; Body_rot = Q_from_Euler123(ChVector<double>(0, 0,-CH_C_PI/2)); }
        if(i==3){Body_Rel_pos = sr_rel_pos_rb; Body_rot = Q_from_Euler123(ChVector<double>(0, 0, CH_C_PI/2)); }
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        cout << "\n" << "The mass of the steering rod is " << mmass * mdensity << "\n" << endl;
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

    /// Create top rod
    ChVector<> tr_rel_pos_l = ChVector<>(-0.214, -0.672, 1.144);
    ChVector<> tr_rel_pos_r = ChVector<>(-0.214,  0.672, 1.144);
    ChVector<> tr_rel_pos_t = ChVector<>( 0.142,  0.0,   1.172);
    {
    for (int i = 0; i < 3; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path;
        if(i == 0){obj_path = "./Mesh_BCE/bar_l.obj";}
        if(i == 1){obj_path = "./Mesh_BCE/bar_r.obj";}
        if(i == 2){obj_path = "./Mesh_BCE/balancer.obj";}
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
        if(i==0){Body_Rel_pos = tr_rel_pos_l; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0,-CH_C_PI/2)); }
        if(i==1){Body_Rel_pos = tr_rel_pos_r; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0,-CH_C_PI/2)); }
        if(i==2){Body_Rel_pos = tr_rel_pos_t; Body_rot = Q_from_Euler123(ChVector<double>( CH_C_PI/2, 0,-CH_C_PI/2)); }
        ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, 
                                         paramsH->bodyIniPosY,
                                         paramsH->bodyIniPosZ) + Body_Rel_pos;
        ChVector<> Body_vel = ChVector<>(0.0, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        cout << "\n" << "The mass of the XX is " << mmass * mdensity << "\n" << endl;
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
    /*{
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            ChVector<> Body_pos = ChVector<>(2.0 - 0.2*j, -0.6, paramsH->bodyIniPosZ);
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
            Body->SetCollide(true);

            // Add this body to the FSI system
            // myFsiSystem.AddFsiBody(Body);
            // fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, Body, ChVector<double>(0), QUNIT, 
            //                     Body_size*0.5, 123, true, false);
        }
    }
    }*/

    /// Create upper floor
    /*{
        auto upperfloor = chrono_types::make_shared<ChBodyEasyBox>(50, 50, 0.02, 1000, false, true, mysurfmaterial);
        upperfloor->SetPos(ChVector<>(0, 0, 1.0));
        upperfloor->SetBodyFixed(true);
        mphysicalSystem.Add(upperfloor);
    }*/

    /// ============================================================================
    /// ======================== constraints =======================================
    /// ============================================================================

    /// Create joint constraints for wheels
    for (int i = 0; i < 6; i++) {
        // pick up bodies and create links
        auto wheel = mphysicalSystem.Get_bodylist()[i+2];
        auto body_connected_to_wheel = mphysicalSystem.Get_bodylist()[0];
        
        // define a rotation of -90 degrees around x (z2y)
        ChQuaternion<> z2y;
        z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));

        // pick up relative position of the link 
        ChVector<> Link_pos;
        ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
        if(i==0){
            Link_pos = Rover_Body_pos + wheel_rel_pos_lf;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+12];
        }
        if(i==1){
            Link_pos = Rover_Body_pos + wheel_rel_pos_rf;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+12];
        }
        if(i==2){
            Link_pos = Rover_Body_pos + wheel_rel_pos_lm;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+8];
        }
        if(i==3){
            Link_pos = Rover_Body_pos + wheel_rel_pos_rm;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+8];
        }
        if(i==4){
            Link_pos = Rover_Body_pos + wheel_rel_pos_lb;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+10];
        }
        if(i==5){
            Link_pos = Rover_Body_pos + wheel_rel_pos_rb;
            body_connected_to_wheel = mphysicalSystem.Get_bodylist()[i+10];
        }

        if( i > 5 ){
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
            revo_link->Initialize(wheel, body_connected_to_wheel, ChCoordsys<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_link);
        }
        else{
            if(motor_type == 1){
                // sr-wheel Revolute constraint with a motor - Rotation Speed
                auto my_speed_function = chrono_types::make_shared<ChFunction_Const>(CH_C_PI*motor_F);  // speed w=3.145 rad/sec
                if(i==0){ 
                    link_motor_speed_lf = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_lf->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_lf);
                    link_motor_speed_lf->SetSpeedFunction(my_speed_function);
                }
                if(i==1){ 
                    link_motor_speed_rf = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_rf->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_rf);
                    link_motor_speed_rf->SetSpeedFunction(my_speed_function);
                }
                if(i==2){ 
                    link_motor_speed_lm = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_lm->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_lm);
                    link_motor_speed_lm->SetSpeedFunction(my_speed_function);
                }
                if(i==3){ 
                    link_motor_speed_rm = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_rm->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_rm);
                    link_motor_speed_rm->SetSpeedFunction(my_speed_function);
                }
                if(i==4){ 
                    link_motor_speed_lb = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_lb->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_lb);
                    link_motor_speed_lb->SetSpeedFunction(my_speed_function);
                }
                if(i==5){ 
                    link_motor_speed_rb = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
                    link_motor_speed_rb->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                    mphysicalSystem.AddLink(link_motor_speed_rb);
                    link_motor_speed_rb->SetSpeedFunction(my_speed_function);
                }

            }
            else if(motor_type == 2){
                // sr-wheel Revolute constraint with a motor - torque
                auto link_motor = chrono_types::make_shared<ChLinkMotorRotationTorque>(); 
                link_motor = chrono_types::make_shared<ChLinkMotorRotationTorque>();
                link_motor->Initialize(body_connected_to_wheel, wheel, ChFrame<>(Link_pos, z2y));
                mphysicalSystem.AddLink(link_motor);
                auto mfun = std::dynamic_pointer_cast<ChFunction_Const>(link_motor->GetTorqueFunction());
                mfun->Set_yconst(motor_F);
            }
        }

    }

    /// Create joint constraints for steering rod
    for (int i = 0; i < 4; i++) {
        // pick up bodies and create links
        auto s_rod = mphysicalSystem.Get_bodylist()[i+12];
        auto body_connected_to_s_rod = mphysicalSystem.Get_bodylist()[i+8];
        
        // define a rotation of -90 degrees around x (z2y)
        ChQuaternion<> z2y;
        z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));

        // pick up relative position of the link 
        ChVector<> Link_pos;
        ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
        if(i==0){
            Link_pos = Rover_Body_pos + sr_rel_pos_lf;
        }
        if(i==1){
            Link_pos = Rover_Body_pos + sr_rel_pos_rf;
        }
        if(i==2){ 
            Link_pos = Rover_Body_pos + sr_rel_pos_lb;
        }
        if(i==3){
            Link_pos = Rover_Body_pos + sr_rel_pos_rb;
        }
        // add Revolute constraint
        auto revo_link = chrono_types::make_shared<ChLinkLockLock>(); 
        revo_link->Initialize(s_rod, body_connected_to_s_rod, ChCoordsys<>(Link_pos, z2y));
        mphysicalSystem.AddLink(revo_link);
    }

    /// Create joint constraints for top rod with rover body
    {
        // pick up bodies and create links
        auto t_rod = mphysicalSystem.Get_bodylist()[18];
        auto body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[1];

        // pick up relative position of the link 
        ChVector<> Link_pos;
        ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
        Link_pos = Rover_Body_pos + tr_rel_pos_t;

        // add Revolute constraint
        auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
        revo_link->Initialize(t_rod, body_connected_to_t_rod, ChCoordsys<>(Link_pos, ChQuaternion<>(1, 0, 0, 0)));
        mphysicalSystem.AddLink(revo_link);


        // Revolute constraint with a motor - Rotation Speed
        // auto link_motor = chrono_types::make_shared<ChLinkMotorRotationSpeed>(); 
        // link_motor->Initialize(body_connected_to_t_rod, t_rod, ChFrame<>(Link_pos, ChQuaternion<>(1, 0, 0, 0)));
        // mphysicalSystem.AddLink(link_motor);
        // auto my_speed_function = chrono_types::make_shared<ChFunction_Const>(CH_C_PI*motor_F*0.1);  // speed w=3.145 rad/sec
        // link_motor->SetSpeedFunction(my_speed_function);

    }

    /// Create joint constraints for top left/right rods with top rod
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            // pick up bodies and create links
            auto t_rod = mphysicalSystem.Get_bodylist()[i+16];
            auto body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[0];
            
            // pick up relative position of the link 
            ChVector<> Link_pos;
            ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
            if(i==0 && j==0){
                Link_pos = Rover_Body_pos + tr_rel_pos_l;
                body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[8];
            }
            if(i==0 && j==1){
                Link_pos = Rover_Body_pos + tr_rel_pos_l + ChVector<>(tr_rel_pos_t.x() - tr_rel_pos_l.x(), 0, 0);
                body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[18];
            }
            if(i==1 && j==0){
                Link_pos = Rover_Body_pos + tr_rel_pos_r;
                body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[9];
            }
            if(i==1 && j==1){
                Link_pos = Rover_Body_pos + tr_rel_pos_r + ChVector<>(tr_rel_pos_t.x() - tr_rel_pos_r.x(), 0, 0);
                body_connected_to_t_rod = mphysicalSystem.Get_bodylist()[18];
            }
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockSpherical>(); 
            revo_link->Initialize(t_rod, body_connected_to_t_rod, ChCoordsys<>(Link_pos, ChQuaternion<>(1, 0, 0, 0)));
            mphysicalSystem.AddLink(revo_link);
        }
    }

    /// Create joint constraints for connecting rod
    for (int i = 0; i < 4; i++) {
        // pick up bodies and create links
        auto c_rod = mphysicalSystem.Get_bodylist()[i+8];
        auto body_connected_to_c_rod = mphysicalSystem.Get_bodylist()[0];
        auto body_rover = mphysicalSystem.Get_bodylist()[1];
        
        // define a rotation of -90 degrees around x (z2y)
        ChQuaternion<> z2y;
        z2y.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(1, 0, 0));

        // pick up relative position of the link 
        ChVector<> Link_pos;
        ChVector<> Rover_Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
        if(i==0){
            Link_pos = Rover_Body_pos + cr_rel_pos_lf;
            body_connected_to_c_rod = mphysicalSystem.Get_bodylist()[1];
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
            revo_link->Initialize(c_rod, body_connected_to_c_rod, ChCoordsys<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_link);
        }
        if(i==1){
            Link_pos = Rover_Body_pos + cr_rel_pos_rf;
            body_connected_to_c_rod = mphysicalSystem.Get_bodylist()[1];
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
            revo_link->Initialize(c_rod, body_connected_to_c_rod, ChCoordsys<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_link);
        }
        if(i==2){ 
            Link_pos = Rover_Body_pos + cr_rel_pos_lb;
            body_connected_to_c_rod = mphysicalSystem.Get_bodylist()[8];
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
            revo_link->Initialize(c_rod, body_connected_to_c_rod, ChCoordsys<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_link);
        }
        if(i==3){
            Link_pos = Rover_Body_pos + cr_rel_pos_rb;
            body_connected_to_c_rod = mphysicalSystem.Get_bodylist()[9];
            // add Revolute constraint
            auto revo_link = chrono_types::make_shared<ChLinkLockRevolute>(); 
            revo_link->Initialize(c_rod, body_connected_to_c_rod, ChCoordsys<>(Link_pos, z2y));
            mphysicalSystem.AddLink(revo_link);
        }
    }

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

        // Save the bodies on the Rover
        for (int i = 0; i < 18; i++) {
            // Find the OBJ files corresponding to each body on the rover
            std::string obj_path;
            char output_body_name[2048] = "/Body_";
            if(i ==  0){obj_path = "./Mesh_BCE/body.obj";}
            if((i > 0) && (i < 7)){
                obj_path = "./Mesh_BCE/wheelSimplified.obj";
            }
            if(i ==  7){obj_path = "./Mesh_BCE/F_L.obj";}
            if(i ==  8){obj_path = "./Mesh_BCE/F_R.obj";}
            if(i ==  9){obj_path = "./Mesh_BCE/B_L.obj";}
            if(i == 10){obj_path = "./Mesh_BCE/B_R.obj";}
            if(i == 11){obj_path = "./Mesh_BCE/ster_front.obj";}
            if(i == 12){obj_path = "./Mesh_BCE/ster_front.obj";}
            if(i == 13){obj_path = "./Mesh_BCE/ster_back.obj";}
            if(i == 14){obj_path = "./Mesh_BCE/ster_back.obj";}
            if(i == 15){obj_path = "./Mesh_BCE/bar_l.obj";}
            if(i == 16){obj_path = "./Mesh_BCE/bar_r.obj";}
            if(i == 17){obj_path = "./Mesh_BCE/balancer.obj";}

            // Get the position and rotation of each body
            auto body = mphysicalSystem.Get_bodylist()[i+1];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();

            // Get the OBJ file and Transform to the correct position and rotation
            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            double scale_ratio = 1.0;
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
            mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 
            mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

            // Save to files
            char filename[8192];
            if(1==0){// save to obj file
                sprintf(filename, "%s%s%d_%d.obj", paramsH->demo_dir, output_body_name, i+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
            }
            if(1==1){// save to vtk file
                sprintf(filename, "%s%s%d_%d.vtk", paramsH->demo_dir, output_body_name, i+1, next_frame);
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
     
        // Save box obstacle to vtk files
        /*for (int i = 0; i < 2; i++) {
            char filename[4096];
            sprintf(filename, "%s/obstacle_%d_%d.vtk", paramsH->demo_dir, i+1, next_frame);
            std::ofstream file;
            file.open(filename);
            file << "# vtk DataFile Version 2.0" << std::endl;
            file << "VTK from simulation" << std::endl;
            file << "ASCII" << std::endl;
            file << "DATASET POLYDATA" << std::endl;

            file << "POINTS " << 8 << " " << "float" << std::endl;
            auto Body = mphysicalSystem.Get_bodylist()[i+19];
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

            file << "POLYGONS " << 6 << " " << 30 << std::endl;
            file << "4 " << 0 << " " << 1 << " " << 2 << " " << 3 << "\n";
            file << "4 " << 0 << " " << 1 << " " << 5 << " " << 4 << "\n";
            file << "4 " << 0 << " " << 4 << " " << 7 << " " << 3 << "\n";
            file << "4 " << 4 << " " << 5 << " " << 6 << " " << 7 << "\n";
            file << "4 " << 1 << " " << 5 << " " << 6 << " " << 2 << "\n";
            file << "4 " << 3 << " " << 2 << " " << 6 << " " << 7 << "\n";
        }*/

        // Save rigid body position and rotation
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

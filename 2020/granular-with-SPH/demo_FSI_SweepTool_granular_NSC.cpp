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
const std::string out_dir = GetChronoOutputPath() + "FSI_Sweep_Tool/";
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
    cout << "usage: ./demo_FSI_Granular_SweepTool <json_file>" << endl;
}

// =============================================================================
int main(int argc, char* argv[]) {
    // create a physics system
    ChSystemNSC mphysicalSystem;
    fsi::ChSystemFsi myFsiSystem(mphysicalSystem);
    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_SweepTool_granular.json");
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

    /// use the following to write a VTK file of the Sweep
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
    auto Sweep = FSI_Bodies[0];
    SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, 0, 0);

    /// write the Penetration into file
    std::ofstream myFile;
    myFile.open("./Sweep_position.txt", std::ios::trunc);
    myFile.close();
    myFile.open("./Sweep_position.txt", std::ios::app);
    myFile << 0.0 << "\t"
           << Sweep->GetPos().x() << "\t" 
           << Sweep->GetPos().y() << "\t"
           << Sweep->GetPos().z() << "\t"
           << Sweep->GetPos_dt().x() << "\t" 
           << Sweep->GetPos_dt().y() << "\t"
           << Sweep->GetPos_dt().z() << "\n";
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
        auto Sweep = mphysicalSystem.Get_bodylist()[1];

        printf("bin=%f,%f,%f\n", bin->GetPos().x(), bin->GetPos().y(), bin->GetPos().z());
        printf("Sweep=%f,%f,%f\n", Sweep->GetPos().x(), Sweep->GetPos().y(), Sweep->GetPos().z());
        printf("Sweep=%f,%f,%f\n", Sweep->GetPos_dt().x(), Sweep->GetPos_dt().y(), Sweep->GetPos_dt().z());
        myFile.open("./Sweep_position.txt", std::ios::app);
        myFile << time << "\t"
               << Sweep->GetPos().x() << "\t" 
               << Sweep->GetPos().y() << "\t"
               << Sweep->GetPos().z() << "\t"
               << Sweep->GetPos_dt().x() << "\t" 
               << Sweep->GetPos_dt().y() << "\t"
               << Sweep->GetPos_dt().z() << "\n";
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

    auto mfloor2 = chrono_types::make_shared<ChBodyEasyBox>(3, 3, 0.02, 1000, true, false);
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

    /// Create Sweep Tool
    {
    for (int i = 0; i < 2; i++) {
        // load mesh from obj file
        auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./L_share.obj";
        double scale_ratio = 1.0;
        mmesh->LoadWavefrontMesh(obj_path, false, true);
        // mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(Sweep_rot));       // rotate the mesh if needed
        mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
        mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

        // compute mass inertia from mesh
        double mmass;
        ChVector<> mcog;
        ChMatrix33<> minertia;
        double mdensity = 100000000.0; //paramsH->bodyDensity;
        mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
        ChMatrix33<> principal_inertia_rot;
        ChVector<> principal_I;
        ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

        // set the abs orientation, position and velocity
        auto Sweep = chrono_types::make_shared<ChBodyAuxRef>();
        ChQuaternion<> Sweep_rot = Q_from_Euler123(ChVector<double>(3.14/2, 0, 3.14/2));
        ChVector<> Sweep_pos = ChVector<>(paramsH->bodyIniPosX, 
                                          paramsH->bodyIniPosY - 0.1 + i*0.2, 
                                          paramsH->bodyIniPosZ);//paramsH->bodyIniPosZ
        ChVector<> Sweep_vel = ChVector<>(-0.48, 0.0, 0.0);

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        Sweep->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        Sweep->SetMass(mmass * mdensity);
        Sweep->SetInertiaXX(mdensity * principal_I);
        // Sweep->SetPos(Sweep_pos);
        Sweep->SetPos_dt(Sweep_vel);
        // Sweep->SetRot(QUNIT);
        
		// Set the absolute position of the body:
        Sweep->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Sweep_pos),ChQuaternion<>(Sweep_rot)));                              
        mphysicalSystem.Add(Sweep);

        Sweep->SetBodyFixed(false);
        Sweep->GetCollisionModel()->ClearModel();
        Sweep->GetCollisionModel()->AddTriangleMesh(mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
        Sweep->GetCollisionModel()->BuildModel();
        Sweep->SetCollide(true);

        auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
        masset_mesh->SetMesh(mmesh);
        masset_mesh->SetBackfaceCull(true);
        Sweep->AddAsset(masset_mesh);

        // Add this body to the FSI system
        myFsiSystem.AddFsiBody(Sweep);
        std::string BCE_path = "./BCE_L_share.txt";
        fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Sweep, BCE_path, 
                                    ChVector<double>(0), QUNIT, scale_ratio);
    }
    }

    /// Create Stone
    {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            // load mesh from obj file
            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./stone2.obj";
            double scale_ratio = 0.0002;
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            // mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(Stone_rot));         // rotate the mesh if needed
            mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));       // scale to a different size
            mmesh->RepairDuplicateVertexes(1e-9);                                   // if meshes are not watertight

            // compute mass inertia from mesh
            double mmass;
            ChVector<> mcog;
            ChMatrix33<> minertia;
            double mdensity = paramsH->rho0 * 0.25; //paramsH->bodyDensity;
            mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
            ChMatrix33<> principal_inertia_rot;
            ChVector<> principal_I;
            ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

            // set the abs orientation, position and velocity
            auto Stone = chrono_types::make_shared<ChBodyAuxRef>();
            ChQuaternion<> Stone_rot = Q_from_Euler123(ChVector<double>((i+1)*3.14/8, (j+0)*3.14/8, (i+1)*3.14/8));
            ChVector<> Stone_pos = ChVector<>(-0.15 + i*0.15 , -0.1 + j*0.2, 0.24);
            ChVector<> Stone_vel = ChVector<>(0.0, 0.0, 0.0);
            
            // Set the COG coordinates to barycenter, without displacing the REF reference.
            // Make the COG frame a principal frame.
            Stone->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

            // Set inertia
            Stone->SetMass(mmass * mdensity);
            Stone->SetInertiaXX(mdensity * principal_I);
            // Stone->SetPos(Stone_pos);
            Stone->SetPos_dt(Stone_vel);
            // Stone->SetRot(Stone_rot);

            // Set the absolute position of the body:
            Stone->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Stone_pos), ChQuaternion<>(Stone_rot)));
            mphysicalSystem.Add(Stone);

            Stone->SetBodyFixed(false);
            Stone->GetCollisionModel()->ClearModel();
            Stone->GetCollisionModel()->AddTriangleMesh(mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
            Stone->GetCollisionModel()->BuildModel();
            Stone->SetCollide(true);

            auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
            masset_mesh->SetMesh(mmesh);
            masset_mesh->SetBackfaceCull(true);
            Stone->AddAsset(masset_mesh);

            // Add this body to the FSI system
            myFsiSystem.AddFsiBody(Stone);
            std::string BCE_path = "./BCE_stone2.txt";
            fsi::utils::AddBCE_FromFile(myFsiSystem.GetDataManager(), paramsH, Stone, BCE_path, 
                                        ChVector<double>(0), QUNIT, scale_ratio);
        }
    }
    }

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

        // save the L plows to obj files
        for (int i = 0; i < 2; i++) {
            auto body = mphysicalSystem.Get_bodylist()[i+1];
            ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
            ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

            auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
            std::string obj_path = "./L_share.obj";
            double scale_ratio = 1.0;
            mmesh->LoadWavefrontMesh(obj_path, false, true);
            mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
            mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

            double mmass;
            ChVector<> mcog;
            ChMatrix33<> minertia;
            mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
            mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

            char filename[2048];
            sprintf(filename, "%s/L_plow_%zd_%zd.obj", paramsH->demo_dir, i+1, next_frame);
            std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
            geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
        }
        // save the stones to obj files
        int n =0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 2; j++) {
                auto body = mphysicalSystem.Get_bodylist()[n+3];
                ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
                ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
                ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

                auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
                std::string obj_path = "./stone2.obj";
                double scale_ratio = 0.0002;
                mmesh->LoadWavefrontMesh(obj_path, false, true);
                mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
                mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

                double mmass;
                ChVector<> mcog;
                ChMatrix33<> minertia;
                mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
                mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body
    
                char filename[2048];
                sprintf(filename, "%s/stone_%zd_%zd.obj", paramsH->demo_dir, n+1, next_frame);
                std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
                geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
                n++;
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

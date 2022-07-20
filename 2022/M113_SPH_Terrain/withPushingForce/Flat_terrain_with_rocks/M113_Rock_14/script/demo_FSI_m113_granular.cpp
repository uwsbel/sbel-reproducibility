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
// Chrono::FSI demo to show usage of tracked vehicle m113 on SPH terrain
// =============================================================================

/// General Includes
#include <cassert>
#include <cstdlib>
#include <ctime>

/// Chrono includes
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

#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"
#include "chrono_vehicle/driver/ChDataDriver.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_models/vehicle/m113/M113.h"

/// Chrono fsi includes
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"

/// Chrono namespaces
using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::geometry;
using namespace chrono::vehicle;
using namespace chrono::vehicle::m113;

/// output directories and settings
const std::string out_dir = GetChronoOutputPath() + "FSI_M113/";
std::string demo_dir;
bool pv_output = true;
bool save_obj = false;  // if true, save as Wavefront OBJ
bool save_vtk = false;  // if true, save as VTK
double rock_scale = 0.5;

double smalldis = 1.0e-9;

/// Dimension of the space domain
double bxDim = 1.0 + smalldis;
double byDim = 1.0 + smalldis;
double bzDim = 0.2 + smalldis;

/// Dimension of the terrain domain
double fxDim = 1.0 + smalldis;
double fyDim = 1.0 + smalldis;
double fzDim = 0.1 + smalldis;

/// Pointer to store the vehicle instance
std::shared_ptr<M113> track;

void CreateMeshMarkers(std::shared_ptr<geometry::ChTriangleMeshConnected> mesh,
                       double delta,
                       std::vector<ChVector<>>& point_cloud) {
    mesh->RepairDuplicateVertexes(1e-9);  // if meshes are not watertight

    ChVector<> minV = mesh->m_vertices[0];
    ChVector<> maxV = mesh->m_vertices[0];
    ChVector<> currV = mesh->m_vertices[0];
    for (unsigned int i = 1; i < mesh->m_vertices.size(); ++i) {
        currV = mesh->m_vertices[i];
        if (minV.x() > currV.x())
            minV.x() = currV.x();
        if (minV.y() > currV.y())
            minV.y() = currV.y();
        if (minV.z() > currV.z())
            minV.z() = currV.z();
        if (maxV.x() < currV.x())
            maxV.x() = currV.x();
        if (maxV.y() < currV.y())
            maxV.y() = currV.y();
        if (maxV.z() < currV.z())
            maxV.z() = currV.z();
    }
    ////printf("start coords: %f, %f, %f\n", minV.x(), minV.y(), minV.z());
    ////printf("end coords: %f, %f, %f\n", maxV.x(), maxV.y(), maxV.z());

    const double EPSI = 1e-6;

    ChVector<> ray_origin;
    for (double x = minV.x(); x < maxV.x(); x += delta) {
        ray_origin.x() = x + 1e-9;
        for (double y = minV.y(); y < maxV.y(); y += delta) {
            ray_origin.y() = y + 1e-9;
            for (double z = minV.z(); z < maxV.z(); z += delta) {
                ray_origin.z() = z + 1e-9;

                ChVector<> ray_dir[2] = {ChVector<>(5, 0.5, 0.25), ChVector<>(-3, 0.7, 10)};
                int intersectCounter[2] = {0, 0};

                for (unsigned int i = 0; i < mesh->m_face_v_indices.size(); ++i) {
                    auto& t_face = mesh->m_face_v_indices[i];
                    auto& v1 = mesh->m_vertices[t_face.x()];
                    auto& v2 = mesh->m_vertices[t_face.y()];
                    auto& v3 = mesh->m_vertices[t_face.z()];

                    // Find vectors for two edges sharing V1
                    auto edge1 = v2 - v1;
                    auto edge2 = v3 - v1;

                    bool t_inter[2] = {false, false};

                    for (unsigned int j = 0; j < 2; j++) {
                        // Begin calculating determinant - also used to calculate uu parameter
                        auto pvec = Vcross(ray_dir[j], edge2);
                        // if determinant is near zero, ray is parallel to plane of triangle
                        double det = Vdot(edge1, pvec);
                        // NOT CULLING
                        if (det > -EPSI && det < EPSI) {
                            t_inter[j] = false;
                            continue;
                        }
                        double inv_det = 1.0 / det;

                        // calculate distance from V1 to ray origin
                        auto tvec = ray_origin - v1;

                        // Calculate uu parameter and test bound
                        double uu = Vdot(tvec, pvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (uu < 0.0 || uu > 1.0) {
                            t_inter[j] = false;
                            continue;
                        }

                        // Prepare to test vv parameter
                        auto qvec = Vcross(tvec, edge1);

                        // Calculate vv parameter and test bound
                        double vv = Vdot(ray_dir[j], qvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (vv < 0.0 || ((uu + vv) > 1.0)) {
                            t_inter[j] = false;
                            continue;
                        }

                        double tt = Vdot(edge2, qvec) * inv_det;
                        if (tt > EPSI) {  // ray intersection
                            t_inter[j] = true;
                            continue;
                        }

                        // No hit, no win
                        t_inter[j] = false;
                    }

                    intersectCounter[0] += t_inter[0] ? 1 : 0;
                    intersectCounter[1] += t_inter[1] ? 1 : 0;
                }

                if (((intersectCounter[0] % 2) == 1) && ((intersectCounter[1] % 2) == 1))  // inside mesh
                    point_cloud.push_back(ChVector<>(x, y, z));
            }
        }
    }
}

std::shared_ptr<ChMaterialSurface> CustomWheelMaterial(ChContactMethod contact_method) {
    float mu = 0.9f;   // coefficient of friction
    float cr = 0.4f;   // coefficient of restitution
    float Y = 2e7f;    // Young's modulus
    float nu = 0.3f;   // Poisson ratio
    float kn = 2e5f;   // normal stiffness
    float gn = 40.0f;  // normal viscous damping
    float kt = 2e5f;   // tangential stiffness
    float gt = 20.0f;  // tangential viscous damping

    switch (contact_method) {
        case ChContactMethod::NSC: {
            auto matNSC = chrono_types::make_shared<ChMaterialSurfaceNSC>();
            matNSC->SetFriction(mu);
            matNSC->SetRestitution(cr);
            return matNSC;
        }
        case ChContactMethod::SMC: {
            auto matSMC = chrono_types::make_shared<ChMaterialSurfaceSMC>();
            matSMC->SetFriction(mu);
            matSMC->SetRestitution(cr);
            matSMC->SetYoungModulus(Y);
            matSMC->SetPoissonRatio(nu);
            matSMC->SetKn(kn);
            matSMC->SetGn(gn);
            matSMC->SetKt(kt);
            matSMC->SetGt(gt);
            return matSMC;
        }
        default:
            return std::shared_ptr<ChMaterialSurface>();
    }
}

/// Forward declaration of helper functions
void SaveParaViewFiles(ChSystemFsi& myFsiSystem,
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

void CreateSolidPhase(ChSystemNSC& mphysicalSystem, ChSystemFsi& myFsiSystem, std::shared_ptr<fsi::SimParams> paramsH);

void ShowUsage() {
    std::cout << "usage: ./demo_FSI_Granular_Viper <json_file>" << std::endl;
}

int main(int argc, char* argv[]) {
    /// Set path to Chrono data directories
    SetChronoDataPath(CHRONO_DATA_DIR);

    /// Create a physical system and a corresponding FSI system
    ChSystemNSC mphysicalSystem;
    ChSystemFsi myFsiSystem(mphysicalSystem);

    /// Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
    std::string inputJson = GetChronoDataFile("fsi/input_json/demo_FSI_Viper_granular_NSC.json");
    if (argc == 1) {
        std::cout << "Use the default JSON file" << std::endl;
    } else if (argc == 2) {
        std::cout << "Use the specified JSON file" << std::endl;
        std::string my_inputJson = std::string(argv[1]);
        inputJson = my_inputJson;
    } else {
        ShowUsage();
        return 1;
    }
    myFsiSystem.SetSimParameter(inputJson, paramsH, ChVector<>(bxDim, byDim, bzDim));

    /// Set SPH discretization type, consistent or inconsistent
    myFsiSystem.SetDiscreType(false, false);

    /// Set wall boundary condition
    myFsiSystem.SetWallBC(BceVersion::ORIGINAL);

    /// Reset the domain size
    bxDim = paramsH->boxDimX + smalldis;
    byDim = paramsH->boxDimY + smalldis;
    bzDim = paramsH->boxDimZ + smalldis;

    fxDim = paramsH->fluidDimX + smalldis;
    fyDim = paramsH->fluidDimY + smalldis;
    fzDim = paramsH->fluidDimZ + smalldis;

    /// Setup the solver based on the input value of the prameters
    myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);

    /// Set the periodic boundary condition
    double initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    ChVector<> cMin(-bxDim / 2 * 1.5, -byDim / 2 * 1.5, -bzDim * 20);
    ChVector<> cMax( bxDim / 2 * 1.5,  byDim / 2 * 1.5,  bzDim * 20);
    myFsiSystem.SetBoundaries(cMin, cMax, paramsH);

    /// Setup sub doamins for a faster neighbor particle searching
    myFsiSystem.SetSubDomain(paramsH);

    /// Setup the output directory for FSI data
    myFsiSystem.SetFsiOutputDir(paramsH, demo_dir, out_dir, inputJson.c_str());

    /// Set FSI information output
    myFsiSystem.SetFsiInfoOutput(false);

    /// Set simulation data output length
    myFsiSystem.SetOutputLength(0);

    /// Create an initial box for the terrain patch
    chrono::utils::GridSampler<> sampler(initSpace0);
    /// Use a chrono sampler to create a bucket of granular material
    ChVector<> boxCenter(0, 0, fzDim / 2);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);
    std::vector<ChVector<>> points = sampler.SampleBox(boxCenter, boxHalfDim);
    /// Add SPH particles from the sampler points to the FSI system
    int numPart = (int)points.size();
    for (int i = 0; i < numPart; i++) {
        double pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim);
        myFsiSystem.AddSphMarker(points[i], paramsH->rho0, 0, paramsH->mu0, paramsH->HSML, -1,
                                 ChVector<>(0),         // initial velocity
                                 ChVector<>(-pre_ini),  // tauxxyyzz
                                 ChVector<>(0)          // tauxyxzyz
        );
    }
    myFsiSystem.AddRefArray(0, (int)numPart, -1, -1);

    /// Create MBD and BCE particles for the solid domain
    CreateSolidPhase(mphysicalSystem, myFsiSystem, paramsH);

    /// Construction of the FSI system must be finalized
    myFsiSystem.Finalize();

    /// Save data at the initial moment
    SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, 0, 0);

    /// Calculate the total steps
    double time = 0;
    double Global_max_dT = paramsH->dT_Max;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);

    // Create the driver system
    // ChDataDriver driver1(track->GetVehicle(), vehicle::GetDataFile("M113/driver/Acceleration.txt"));
    // driver1.Initialize();

    // TerrainForces shoe_forces_left(track->GetVehicle().GetNumTrackShoes(LEFT));
    // TerrainForces shoe_forces_right(track->GetVehicle().GetNumTrackShoes(RIGHT));

    /// Print the body name and total number of bodies
    int num_body = mphysicalSystem.Get_bodylist().size();
    std::cout << "\n" << "Total number of bodies is " << num_body << "\n" << std::endl;
    for (int n = 0; n < num_body; n++) {
        auto bodynow = mphysicalSystem.Get_bodylist()[n];
        std::cout << "\n" << "Body " << n << " is: "<< bodynow->GetName() << std::endl;
    }

    /// Add a force to the chassis to push the vehicle
    std::shared_ptr<ChBody> m113_chassis = track->GetVehicle().GetChassisBody();
    double force_base = 5000;
    ChVector<> force_on_chassis(force_base*cos(0.25*CH_C_PI), force_base*sin(0.25*CH_C_PI), 0);

    /// Add timing for ths simulation
    double TIMING_sta;
    double TIMING_end;
    double sim_cost = 0.0;
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

        // ChDriver::Inputs driver_inputs = driver1.GetInputs();

        // Update modules (process inputs from other modules)
        // driver1.Synchronize(time);
        // track->Synchronize(time, driver_inputs, shoe_forces_left, shoe_forces_right);

        // Advance simulation for one timestep for all modules
        // driver1.Advance(paramsH->dT_Max);
        // track->Advance(paramsH->dT_Max);

        // Add a force to the chassis to push the vehicle
        if(time > 1.0){
            double m113_speed = track->GetVehicle().GetVehicleSpeed();
            if (m113_speed < 1.0){
                force_base = sqrt(pow(force_on_chassis.x(),2)+pow(force_on_chassis.y(),2));
                if (force_base < 50000.0){
                    force_on_chassis.x() = (force_base + 2000.0)*cos(0.25*CH_C_PI);
                    force_on_chassis.y() = (force_base + 2000.0)*sin(0.25*CH_C_PI);
                }
            }else{
                force_base = sqrt(pow(force_on_chassis.x(),2)+pow(force_on_chassis.y(),2));
                if (force_base > 3000.0){
                    force_on_chassis.x() = (force_base - 2000.0)*cos(0.25*CH_C_PI);
                    force_on_chassis.y() = (force_base - 2000.0)*sin(0.25*CH_C_PI);
                }
            }
            m113_chassis->Empty_forces_accumulators();
            m113_chassis->Accumulate_force(force_on_chassis, m113_chassis->GetPos(), false);
        }

        // Do step dynamics
        TIMING_sta = clock();
        myFsiSystem.DoStepDynamics_FSI();
        TIMING_end = clock();
        sim_cost = sim_cost + (TIMING_end - TIMING_sta) / (double)CLOCKS_PER_SEC;
        time += paramsH->dT;

        // Save data to files
        SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, next_frame, time);

        // output inf to screen
        auto bbody = mphysicalSystem.Get_bodylist()[0];
        auto vbody = mphysicalSystem.Get_bodylist()[1];
        printf("bin=%f,%f,%f\n", bbody->GetPos().x(), bbody->GetPos().y(), bbody->GetPos().z());
        printf("M113=%f,%f,%f\n", vbody->GetPos().x(), vbody->GetPos().y(), vbody->GetPos().z());
        printf("M113=%f,%f,%f\n", vbody->GetPos_dt().x(), vbody->GetPos_dt().y(), vbody->GetPos_dt().z());
        printf("Physical time and computational cost = %f, %f\n", time, sim_cost);

        double mspeed = track->GetVehicle().GetVehicleSpeed();
        double mmass = track->GetVehicle().GetVehicleMass();
        ChVector<> tr_pos = track->GetVehicle().GetVehiclePos();
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "force_on_chassis = " << force_on_chassis.x() << " " << force_on_chassis.y() << " " << force_on_chassis.z() << std::endl;
        std::cout << "mass and velocity = " << mmass << " " << mspeed << std::endl;
        std::cout << "position = " << tr_pos.x() << " " << tr_pos.y() << " " << tr_pos.z() << std::endl;
        std::cout << "-------------------------------------" << std::endl;

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
                      ChSystemFsi& myFsiSystem, 
                      std::shared_ptr<fsi::SimParams> paramsH) {
    /// Set the gravity force for the simulation
    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
    mphysicalSystem.Set_G_acc(gravity);

    /// Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    mysurfmaterial->SetFriction(0.9);
    mysurfmaterial->SetRestitution(0.4);
    collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0025);
    collision::ChCollisionModel::SetDefaultSuggestedMargin(0.0025);

    /// Create a body for the rigid soil container
    auto box = chrono_types::make_shared<ChBodyEasyBox>(100, 100, 0.02, 1000, false, true, mysurfmaterial);
    box->SetPos(ChVector<>(0, 0, 0));
    box->SetBodyFixed(true);
    mphysicalSystem.Add(box);

    /// Get the initial SPH particle spacing
    double initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    /// Bottom wall
    ChVector<> size_XY(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> pos_zn(0, 0, -3 * initSpace0);
    ChVector<> pos_zp(0, 0, bzDim + 2 * initSpace0);

    /// Left and right wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);

    /// Front and back wall
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 0 * initSpace0);

    /// Fluid-Solid Coupling at the walls via BCE particles
    myFsiSystem.AddBceBox(paramsH, box, pos_zn, QUNIT, size_XY, 12);
    myFsiSystem.AddBceBox(paramsH, box, pos_xp, QUNIT, size_YZ, 23);
    myFsiSystem.AddBceBox(paramsH, box, pos_xn, QUNIT, size_YZ, 23);
    myFsiSystem.AddBceBox(paramsH, box, pos_yp, QUNIT, size_XZ, 13);
    myFsiSystem.AddBceBox(paramsH, box, pos_yn, QUNIT, size_XZ, 13);

    // --------------------------
    // Construct the M113 vehicle
    // --------------------------
    track = chrono_types::make_shared<M113>(&mphysicalSystem);
    track->SetContactMethod(ChContactMethod::NSC);
    track->SetTrackShoeType(TrackShoeType::SINGLE_PIN);
    track->SetBrakeType(BrakeType::SIMPLE);
    // track->SetDrivelineType(DrivelineTypeTV::BDS);
    track->SetPowertrainType(PowertrainModelType::SIMPLE_CVT);
    track->SetChassisCollisionType(CollisionType::NONE);

    // Initialize the vehicle at the specified position
    ChVector<> initLoc(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ);
    ChQuaternion<> initRot = Q_from_Euler123(ChVector<double>(0, 0, 0.25*CH_C_PI));
    track->SetInitPosition(ChCoordsys<>(initLoc, initRot));
    track->Initialize();

    // Get the number of shoes: left and right
    int num_shoe_L = track->GetVehicle().GetNumTrackShoes(LEFT);
    int num_shoe_R = track->GetVehicle().GetNumTrackShoes(RIGHT);
    std::cout << "number of shoes left = " << num_shoe_L  << std::endl;
    std::cout << "number of shoes right = " << num_shoe_R  << std::endl;

    // Add BCE for each shoe
    for (int i = 0; i < num_shoe_L; i++) {
        auto track_L = track->GetVehicle().GetTrackAssembly(LEFT);
        auto shoe_body = track_L->GetTrackShoe(i)->GetShoeBody();
        myFsiSystem.AddFsiBody(shoe_body);
        myFsiSystem.AddBceBox(paramsH, shoe_body, ChVector<>(0), QUNIT, ChVector<>(0.06, 0.18, 0.04), 123, true);
    }
    for (int i = 0; i < num_shoe_R; i++) {
        auto track_R = track->GetVehicle().GetTrackAssembly(RIGHT);
        auto shoe_body = track_R->GetTrackShoe(i)->GetShoeBody();
        myFsiSystem.AddFsiBody(shoe_body);
        myFsiSystem.AddBceBox(paramsH, shoe_body, ChVector<>(0), QUNIT, ChVector<>(0.06, 0.18, 0.04), 123, true);
    }

    /// Add some rocks on the terrain
    std::vector<ChVector<>> BCE_par_rock;
    int n_r = 0;
    for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
        std::string rock_name = "rock" + std::to_string(n_r+1);
        // Load mesh from obj file
        auto trimesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        std::string obj_path = "./rock3.obj";
        trimesh->LoadWavefrontMesh(obj_path, true, true);
        double scale_ratio = rock_scale;
        // trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(body_rot));    // rotate the mesh if needed
        trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));    // scale to a different size
        // trimesh->RepairDuplicateVertexes(1e-9);                             // if meshes are not watertight  

        // Compute mass inertia from mesh
        double mmass;// = 5.0;
        double mdensity = paramsH->bodyDensity;
        ChVector<> mcog;// = ChVector<>(0.0, 0.0, 0.0);
        ChMatrix33<> minertia;
        trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
        ChVector<> principal_I;// = ChVector<>(1.0, 1.0, 1.0);
        ChMatrix33<> principal_inertia_rot;// = ChMatrix33<>(1.0);
        ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

        // Set the abs orientation, position
        auto rock_body = chrono_types::make_shared<ChBodyAuxRef>();
        rock_body->SetNameString(rock_name);
        double rot_ang_x, rot_ang_y, rot_ang_z;
        ChVector<> rock_rel_pos; 
        
        // Set initial pos and rot
        rot_ang_x = i * 30.0 / 180.0 * CH_C_PI;
        rot_ang_y = j * 45.0 / 180.0 * CH_C_PI;
        rot_ang_z = i * 60.0 / 180.0 * CH_C_PI;
        double det_x = 0.1 * pow(-1.0, i + j);
        double det_y = 0.1 * pow(-1.0, i + j + 1);
        ChVector<> rock_pos = ChVector<>(-2.0 + i * 1.0 + det_x, -2.0 + j * 1.0 + det_y, 0.8);
        ChQuaternion<> rock_rot = Q_from_Euler123(ChVector<double>(rot_ang_x, rot_ang_y, rot_ang_z));

        // Set the COG coordinates to barycenter, without displacing the REF reference.
        // Make the COG frame a principal frame.
        rock_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

        // Set inertia
        std::cout << "\n" << "The mass of the rock is " << mmass * mdensity << "\n" << std::endl;
        rock_body->SetMass(mmass * mdensity);
        rock_body->SetInertiaXX(mdensity * principal_I);
        
        // Set the absolute position of the body:
        rock_body->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(rock_pos),ChQuaternion<>(rock_rot)));                              
        mphysicalSystem.Add(rock_body);

        // Set collision
        rock_body->SetBodyFixed(false);
        rock_body->GetCollisionModel()->ClearModel();
        rock_body->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, trimesh, false, false, VNULL, ChMatrix33<>(1), 0.005);
        rock_body->GetCollisionModel()->BuildModel();
        rock_body->SetCollide(true);

        // Create BCE particles associated with mesh
        if(i==0){
            CreateMeshMarkers(trimesh, (double)initSpace0, BCE_par_rock);
        }
        myFsiSystem.AddFsiBody(rock_body);
        myFsiSystem.AddBceFromPoints(paramsH, rock_body, BCE_par_rock, ChVector<>(0.0), QUNIT);
        n_r++;
    }
    }
}

//------------------------------------------------------------------
// Function to save the povray files of the MBD
//------------------------------------------------------------------
void SaveParaViewFiles(ChSystemFsi& myFsiSystem,
                       ChSystemNSC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int next_frame,
                       double mTime) {
    double frame_time = 1.0 / paramsH->out_fps;
    char filename[4096];

    if (std::abs(mTime - (next_frame)*frame_time) < 1e-5) {
        /// save the SPH particles
        if (pv_output)
            myFsiSystem.PrintParticleToFile(demo_dir);

        /// save rigid body position and rotation
        for (int i = 1; i < mphysicalSystem.Get_bodylist().size(); i++) {
            auto body = mphysicalSystem.Get_bodylist()[i];
            ChFrame<> ref_frame = body->GetFrame_REF_to_abs();
            ChVector<> pos = ref_frame.GetPos();
            ChVector<> vel = body->GetPos_dt();
            ChQuaternion<> rot = ref_frame.GetRot();

            std::string delim = ",";
            sprintf(filename, "%s/body_pos_rot_vel%d.csv", paramsH->demo_dir, i);
            std::ofstream file;
            if (mphysicalSystem.GetChTime() > 0)
                file.open(filename, std::fstream::app);
            else {
                file.open(filename);
                file << "Time" << delim << "x" << delim << "y" << delim << "z" << delim << "q0" << delim << "q1"
                     << delim << "q2" << delim << "q3" << delim << "Vx" << delim << "Vy" << delim << "Vz" << std::endl;
            }

            file << mphysicalSystem.GetChTime() << delim << pos.x() << delim << pos.y() << delim << pos.z() << delim
                 << rot.e0() << delim << rot.e1() << delim << rot.e2() << delim << rot.e3() << delim << vel.x() << delim
                 << vel.y() << delim << vel.z() << std::endl;

            file.close();
        }

        std::cout << "-------------------------------------\n" << std::endl;
        std::cout << "             Output frame:   " << next_frame << std::endl;
        std::cout << "             Time:           " << mTime << std::endl;
        std::cout << "-------------------------------------\n" << std::endl;
    }
}

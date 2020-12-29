// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang
// =============================================================================
// Low-velocity impact test of a sphere (mesh) dropped on the a bed of settle
// granular material
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include "chrono/core/ChGlobal.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChForce.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/timestepper/ChTimestepper.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "ChGranularDemoUtils.hpp"

using namespace chrono;
using namespace chrono::granular;

void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file> "  + " <input_file> " + " <projectile density> " + " <drop height> "  <<std::endl;
}

float getMass(sim_param_holder& params){
    float rad = params.sphere_radius;
    float density = params.sphere_density;

    float volume = 4.0f/3.0f * CH_C_PI * std::pow(rad, 3);
    float mass = volume * density;
    return mass;
}

// calculate kinetic energy of the system
float getSystemKE(sim_param_holder &params, ChGranularChronoTriMeshAPI &apiSMC, int numSpheres){
    float sysKE = 0.0f;
    float sphere_KE;
    ChVector<float> angularVelo;
    ChVector<float> velo;
    float mass = getMass(params);
    float inertia = 0.4f * mass * std::pow(params.sphere_radius,2);

    for (int i = 0; i < numSpheres; i++){
        angularVelo = apiSMC.getAngularVelo(i);
        velo = apiSMC.getVelo(i);
        sphere_KE = 0.5f * mass * velo.Length2() + 0.5f * inertia * angularVelo.Length2();
        sysKE = sysKE + sphere_KE;
    }
    return sysKE;
}

// get maximum point in granular materialr4w   
float getMax_Z(std::vector<ChVector<float>> points){
    float max_z = -1000.0f;
    float pos_z;
    for (int i = 0; i < points.size(); i++){
        pos_z = points.at(i).z();
        if (pos_z > max_z)
            max_z = pos_z;
    }
    return max_z;
}

int main(int argc, char* argv[]) {
    sim_param_holder params;

    if (argc != 5 || ParseJSON(argv[1], params) == false){
        ShowUsage(argv[0]);
        return 1;
    }

    float iteration_step = params.step_size;

    ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density, 
                                        make_float3(params.box_X, params.box_Y, params.box_Z));

    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();

    // declare particle positon vector
    std::vector<chrono::ChVector<float>> body_points; 
    body_points = loadPositionCheckpoint<float>(argv[2]);
    std::cout << "sucessfully load file from: " << argv[2] << std::endl;

    // parameters for the impact test
    float rho_sphere = std::stof(argv[3]);
    float initial_height = std::stof(argv[4]);
    std::cout << "drop ball density: " << rho_sphere << "g/cm^3, drop height: " << initial_height << "cm. " << std::endl; 

    // TODO: find a better way to describe the surface
    float initial_surface = getMax_Z(body_points) + params.sphere_radius;
    float initial_volume = params.box_X * params.box_Y * (initial_surface + params.box_Z/2.0f);
    int numSpheres = body_points.size();
    float volume_per_particle = 4.0/3.0 * CH_C_PI * std::pow(params.sphere_radius,3);
    float mass_per_particle = volume_per_particle * params.sphere_density;

    float bulk_mass = numSpheres * mass_per_particle;
    // TODO: cout bulk density and packing frac during settling stage
    float bulk_density = bulk_mass/initial_volume;
    float packing_frac = numSpheres * volume_per_particle / initial_volume;
    std::cout << "highest point at " << initial_surface << "cm" << std::endl;
    std::cout << "bulk density is " << bulk_density << "g/cm3" << std::endl;
    std::cout << "packing frac is " << packing_frac << std::endl;
    



    apiSMC_TriMesh.setElemsPositions(body_points);
    gran_sys.set_BD_Fixed(true);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);

    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);

    // somehow setting rolling friction make the result less accurate
    // float mu_roll = 0.05;
    // gran_sys.set_rolling_mode(GRAN_ROLLING_MODE::SCHWARTZ);
    // gran_sys.set_rolling_coeff_SPH2SPH(mu_roll);
    // gran_sys.set_rolling_coeff_SPH2WALL(mu_roll);
    // gran_sys.set_rolling_coeff_SPH2MESH(mu_roll);


    std::string mesh_filename("data/balldrop/sphere.obj");
    std::vector<string> mesh_filenames(1, mesh_filename);

    float ball_radius = 1.27f;
    float initial_ball_pos = initial_surface + initial_height + ball_radius;
    std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, 0.f));
    std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(ball_radius));

    float ball_mass = (float)(4.f * CH_C_PI * ball_radius * ball_radius * ball_radius * rho_sphere / 3.f);
    std::vector<float> mesh_masses(1, ball_mass);

    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);

    params.output_dir = params.output_dir+"_rho_"+argv[3]+"_height_"+argv[4];

    filesystem::create_directory(filesystem::path(params.output_dir));

    unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();
    std::cout << nSoupFamilies << " soup families" << std::endl;
    double* meshPosRot = new double[7 * nSoupFamilies];
    float* meshVel = new float[6 * nSoupFamilies]();

    gran_sys.initialize();

    // Create rigid ball_body simulation
    ChSystemSMC sys_ball;
    sys_ball.SetContactForceModel(ChSystemSMC::ContactForceModel::Hertz); // this need to be consistent with the granular bed
    sys_ball.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);
    sys_ball.Set_G_acc(ChVector<>(0, 0, -980));

    double inertia = 2.0 / 5.0 * ball_mass * ball_radius * ball_radius;
    ChVector<> ball_initial_pos(0, 0, initial_ball_pos);

    std::shared_ptr<ChBody> ball_body(sys_ball.NewBody());
    ball_body->SetMass(ball_mass);
    ball_body->SetInertiaXX(ChVector<>(inertia, inertia, inertia));
    ball_body->SetPos(ball_initial_pos);
    sys_ball.AddBody(ball_body);


    int currframe = 0;
    unsigned int curr_step = 0;
    int frame_output_freq = 100; // write position info every 100 time steps
    int cout_freq = 10; // cout info every 10 time steps

    clock_t start = std::clock();
    for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {
        auto ball_pos = ball_body->GetPos();
        auto ball_rot = ball_body->GetRot();

        auto ball_vel = ball_body->GetPos_dt();
        auto ball_ang_vel = ball_body->GetWvel_loc();
        ball_ang_vel = ball_body->GetRot().GetInverse().Rotate(ball_ang_vel);

        meshPosRot[0] = ball_pos.x();
        meshPosRot[1] = ball_pos.y();
        meshPosRot[2] = ball_pos.z();
        meshPosRot[3] = ball_rot[0];
        meshPosRot[4] = ball_rot[1];
        meshPosRot[5] = ball_rot[2];
        meshPosRot[6] = ball_rot[3];

        meshVel[0] = (float)ball_vel.x();
        meshVel[1] = (float)ball_vel.y();
        meshVel[2] = (float)ball_vel.z();
        meshVel[3] = (float)ball_ang_vel.x();
        meshVel[4] = (float)ball_ang_vel.y();
        meshVel[5] = (float)ball_ang_vel.z();

        gran_sys.meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);

        gran_sys.advance_simulation(iteration_step);
        sys_ball.DoStepDynamics(iteration_step);

        float ball_force[6];
        gran_sys.collectGeneralizedForcesOnMeshSoup(ball_force);

        ball_body->Empty_forces_accumulators();
        ball_body->Accumulate_force(ChVector<>(ball_force[0], ball_force[1], ball_force[2]), ball_pos, false);
        ball_body->Accumulate_torque(ChVector<>(ball_force[3], ball_force[4], ball_force[5]), false);

        float KE = getSystemKE(params, apiSMC_TriMesh, numSpheres);
        float penetration_d = initial_surface - ball_pos.z() + ball_radius;
        float KE_threshold = 0.1f;

        // only output info when the ball is impacting the granular material
        if (penetration_d > 0){
            if (curr_step % frame_output_freq == 0) {
                char filename[100];
                sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
                gran_sys.writeFile(std::string(filename));
                gran_sys.write_meshes(std::string(filename));
            }

            if (curr_step % cout_freq == 0){
                printf("t=%f, KE=%e J, penetration_d = %e cm, dropHeight H = %e cm\n", t, KE*1E-5, penetration_d, initial_height + penetration_d);
            }
        }

    }

    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}

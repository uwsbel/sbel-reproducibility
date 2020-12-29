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
// Granular material settling in a square container to generate the bed for ball
// drop test, once settled positions are written
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
    std::cout << "usage: " + name + " <json_file>" << std::endl;
    std::cout << "OR " + name + " <json_file> "  + " <input_file> " <<std::endl;
}

float getMass(sim_param_holder& params){
    float rad = params.sphere_radius;
    float density = params.sphere_density;

    float volume = 4.0f/3.0f * CH_C_PI * std::pow(rad, 3);
    float mass = volume * density;
    return mass;
}

// calculate kinetic energy of the system
float getSystemKE(sim_param_holder &params, ChGranularSMC_API &apiSMC, int numSpheres){
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

int main(int argc, char* argv[]) {
    sim_param_holder params;
    bool useCheckpointInput = false;

    if ((argc != 2 && argc != 3) || ParseJSON(argv[1], params) == false){
        ShowUsage(argv[0]);
        return 1;
    }
    if (argc == 3 ) {
        useCheckpointInput = true;
    }

    float iteration_step = params.step_size;

    ChSystemGranularSMC gran_sys(params.sphere_radius, params.sphere_density,
                                 make_float3(params.box_X, params.box_Y, params.box_Z));

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);


    double fill_bottom = -params.box_Z / 2.0;
    double fill_top = params.box_Z / 4.0;

    // declar particle positon vector
    std::vector<chrono::ChVector<float>> body_points; 
    if (useCheckpointInput == true){
        body_points = loadPositionCheckpoint<float>(argv[2]);
		std::cout << "reading position input success from " << argv[2]<<std::endl;
	}
    else
    {
        chrono::utils::PDSampler<float> sampler(2.1f * params.sphere_radius);

        ChVector<> hdims(params.box_X / 2 - 2*params.sphere_radius, params.box_Y / 2 - 2*params.sphere_radius, 0);
        ChVector<> center(0, 0, fill_bottom + 2.0 * params.sphere_radius);

        // Shift up for bottom of box
        while (center.z() < fill_top) {
            std::cout << "Create layer at " << center.z() << std::endl;
            auto points = sampler.SampleBox(center, hdims);
            body_points.insert(body_points.end(), points.begin(), points.end());
            center.z() += 2.1 * params.sphere_radius;
        }
    }
    
    int numSpheres = body_points.size();
    // apiSMC_TriMesh.setElemsPositions(body_points);
    apiSMC.setElemsPositions(body_points);

    gran_sys.set_BD_Fixed(true);

    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    // gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);

    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    // gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

    gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);
    gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
    // gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);

    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
    gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    // gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);

    gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    // gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
    gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    // gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);

    //gran_sys.set_rolling_coeff_SPH2SPH(params.rolling_friction_coeffS2S);
    //gran_sys.set_rolling_coeff_SPH2WALL(params.rolling_friction_coeffS2W);
    //gran_sys.set_rolling_coeff_SPH2MESH(params.rolling_friction_coeffS2M);

    // std::string mesh_filename(GetChronoDataFile("granular/demo_GRAN_ballcosim/sphere.obj"));
    // std::vector<string> mesh_filenames(1, mesh_filename);
    // std::vector<float3> mesh_translations(1, make_float3(0.f, 0.f, 0.f));
    // float ball_radius = 20.f;
    // std::vector<ChMatrix33<float>> mesh_rotscales(1, ChMatrix33<float>(ball_radius));
    // float ball_density = params.sphere_density;

    // float ball_mass = (float)(4.f * CH_C_PI * ball_radius * ball_radius * ball_radius * ball_density / 3.f);
    // std::vector<float> mesh_masses(1, ball_mass);

    // apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);

    gran_sys.setOutputMode(params.write_mode);
    gran_sys.setVerbose(params.verbose);
    filesystem::create_directory(filesystem::path(params.output_dir));

    // unsigned int nSoupFamilies = gran_sys.getNumTriangleFamilies();
    // std::cout << nSoupFamilies << " soup families" << std::endl;
    // double* meshPosRot = new double[7 * nSoupFamilies];
    // float* meshVel = new float[6 * nSoupFamilies]();

    gran_sys.initialize();

    // Create rigid ball_body simulation
    // ChSystemSMC sys_ball;
    // sys_ball.SetContactForceModel(ChSystemSMC::ContactForceModel::Hooke); // this need to be consistent with the granular bed
    // sys_ball.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);
    // sys_ball.Set_G_acc(ChVector<>(0, 0, -980));

    // double inertia = 2.0 / 5.0 * ball_mass * ball_radius * ball_radius;
    // ChVector<> ball_initial_pos(0, 0, fill_top + ball_radius + 2 * params.sphere_radius);

    // std::shared_ptr<ChBody> ball_body(sys_ball.NewBody());
    // ball_body->SetMass(ball_mass);
    // ball_body->SetInertiaXX(ChVector<>(inertia, inertia, inertia));
    // ball_body->SetPos(ball_initial_pos);
    // sys_ball.AddBody(ball_body);
    unsigned int out_fps = 50;
    std::cout << "Rendering at " << out_fps << "FPS" << std::endl;

    unsigned int out_steps = (unsigned int)(1.0 / (out_fps * iteration_step));

    int currframe = 0;
    unsigned int curr_step = 0;

    clock_t start = std::clock();
    for (double t = 0; t < (double)params.time_end; t += iteration_step, curr_step++) {
        // auto ball_pos = ball_body->GetPos();
        // auto ball_rot = ball_body->GetRot();

        // auto ball_vel = ball_body->GetPos_dt();
        // auto ball_ang_vel = ball_body->GetWvel_loc();
        // ball_ang_vel = ball_body->GetRot().GetInverse().Rotate(ball_ang_vel);

        // meshPosRot[0] = ball_pos.x();
        // meshPosRot[1] = ball_pos.y();
        // meshPosRot[2] = ball_pos.z();
        // meshPosRot[3] = ball_rot[0];
        // meshPosRot[4] = ball_rot[1];
        // meshPosRot[5] = ball_rot[2];
        // meshPosRot[6] = ball_rot[3];

        // meshVel[0] = (float)ball_vel.x();
        // meshVel[1] = (float)ball_vel.y();
        // meshVel[2] = (float)ball_vel.z();
        // meshVel[3] = (float)ball_ang_vel.x();
        // meshVel[4] = (float)ball_ang_vel.y();
        // meshVel[5] = (float)ball_ang_vel.z();

        // gran_sys.meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);

        gran_sys.advance_simulation(iteration_step);
        // sys_ball.DoStepDynamics(iteration_step);

        // float ball_force[6];
        // gran_sys.collectGeneralizedForcesOnMeshSoup(ball_force);

        // ball_body->Empty_forces_accumulators();
        // ball_body->Accumulate_force(ChVector<>(ball_force[0], ball_force[1], ball_force[2]), ball_pos, false);
        // ball_body->Accumulate_torque(ChVector<>(ball_force[3], ball_force[4], ball_force[5]), false);
        float KE;
        if (curr_step % out_steps == 0) {
            std::cout << "Rendering frame " << currframe;
            char filename[100];
            sprintf(filename, "%s/step%06d", params.output_dir.c_str(), currframe++);
            gran_sys.writeFile(std::string(filename));
            // gran_sys.write_meshes(std::string(filename));

            /*  // disable meshframes output, for it may be confusing for users dealing with C::Granular only
            std::string mesh_output = std::string(filename) + "_meshframes.csv";
            std::ofstream meshfile(mesh_output);
            std::ostringstream outstream;
            outstream << "mesh_name,dx,dy,dz,x1,x2,x3,y1,y2,y3,z1,z2,z3,sx,sy,sz\n";
            writeMeshFrames(outstream, *ball_body, mesh_filename, ball_radius);
            meshfile << outstream.str();
            */
        KE = getSystemKE(params, apiSMC, numSpheres);

        std::cout << ", time = " << t << ", KE = " << KE * 1E-5 << ", max z: " << gran_sys.get_max_z() << std::endl;


		
		}
    }

    clock_t end = std::clock();
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    std::cout << "Time: " << total_time << " seconds" << std::endl;

    // delete[] meshPosRot;
    // delete[] meshVel;

    return 0;
}

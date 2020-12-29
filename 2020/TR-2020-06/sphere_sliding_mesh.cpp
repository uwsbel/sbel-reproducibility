// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Luning Fang, Ruochun Zhang
// =============================================================================
// validation test: sphere rolling on a plane modeled as a planeBC and/or a mesh facet
// =============================================================================
// version: feature/granular at commit 95ff8406b4b22604c871fd98482fa0faf6bf56fe
// =============================================================================

#include <cmath>
#include <iostream>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "ChGranularDemoUtils.hpp"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono_granular/ChGranularData.h"
#include "chrono_granular/api/ChApiGranularChrono.h"
#include "chrono_granular/physics/ChGranular.h"
#include "chrono_granular/physics/ChGranularTriMesh.h"
#include "chrono_granular/utils/ChGranularJsonParser.h"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::granular;


void ShowUsage(std::string name) {
    std::cout << "usage: " + name + " <json_file> \n " << std::endl;
}

int main(int argc, char* argv[]) {
    sim_param_holder params;

    // Some of the default values are overwritten by user via command line
    if (argc != 2 || ParseJSON(argv[1], params) == false) {
        ShowUsage(argv[0]);
        return 1;
    }

    // big domain dimension
    params.box_X = 20.f;
    params.box_Y = 20.f;
    params.box_Z = 10.f;

    ChGranularChronoTriMeshAPI apiSMC_TriMesh(params.sphere_radius, params.sphere_density, 
                                make_float3(params.box_X, params.box_Y, params.box_Z));
    ChSystemGranularSMC_trimesh& gran_sys = apiSMC_TriMesh.getGranSystemSMC_TriMesh();

    ChGranularSMC_API apiSMC;
    apiSMC.setGranSystem(&gran_sys);
   
    // load in the mesh
    std::vector<string> mesh_filenames;
    std::string mesh_filename;
    mesh_filename = "./data/one_facet.obj";
    mesh_filenames.push_back(mesh_filename);

    std::vector<ChMatrix33<float>> mesh_rotscales;
    std::vector<float3> mesh_translations;
    mesh_rotscales.push_back(ChMatrix33<float>(ChVector<float>(1.f, 1.f, 1.f)));
    mesh_translations.push_back(make_float3(0, 0, 0));

    std::vector<float> mesh_masses;
    float mass = 100;
    mesh_masses.push_back(mass);

    
    apiSMC_TriMesh.load_meshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);
    gran_sys.enableMeshCollision();
    

    /*
    // create plane BC at the bottom of BD
    float plane_pos[3] = {0.0f, 0.0f, 0.0f};
    float plane_normal[3] = {0.0f, 0.0f, 1.0f};
    size_t plane_bc_id = gran_sys.Create_BC_Plane(plane_pos, plane_normal, true);
    */

    // assign initial condition for the sphere
    float initialVelo = 1.f;
    std::vector<ChVector<float>> body_point;
	body_point.push_back(ChVector<float>(1.0f, -1.0f, params.sphere_radius));
	std::vector<ChVector<float>> velocity;
	velocity.push_back(ChVector<float>(initialVelo, 0.0f, 0.0f));
    apiSMC.setElemsPositions(body_point, velocity);

	gran_sys.setPsiFactors(params.psi_T, params.psi_L);

	// set normal force model
    gran_sys.set_K_n_SPH2SPH(params.normalStiffS2S);
    gran_sys.set_K_n_SPH2WALL(params.normalStiffS2W);
    gran_sys.set_K_n_SPH2MESH(params.normalStiffS2M);
    gran_sys.set_Gamma_n_SPH2SPH(params.normalDampS2S);
    gran_sys.set_Gamma_n_SPH2WALL(params.normalDampS2W);
    gran_sys.set_Gamma_n_SPH2MESH(params.normalDampS2M);

		// set tangential force model
	gran_sys.set_K_t_SPH2SPH(params.tangentStiffS2S);		
	gran_sys.set_K_t_SPH2WALL(params.tangentStiffS2W);
	gran_sys.set_K_t_SPH2MESH(params.tangentStiffS2M);
    gran_sys.set_Gamma_t_SPH2SPH(params.tangentDampS2S);
	gran_sys.set_Gamma_t_SPH2WALL(params.tangentDampS2W);
    gran_sys.set_Gamma_t_SPH2MESH(params.tangentDampS2M);
	gran_sys.set_static_friction_coeff_SPH2SPH(params.static_friction_coeffS2S);
	gran_sys.set_static_friction_coeff_SPH2WALL(params.static_friction_coeffS2W);
    gran_sys.set_static_friction_coeff_SPH2MESH(params.static_friction_coeffS2M);
    gran_sys.set_friction_mode(GRAN_FRICTION_MODE::MULTI_STEP);

    
    // set rolling params
	gran_sys.set_rolling_mode(params.rolling_mode);
    gran_sys.set_rolling_coeff_SPH2SPH(params.rolling_friction_coeffS2S);
    gran_sys.set_rolling_coeff_SPH2WALL(params.rolling_friction_coeffS2W);
    gran_sys.set_rolling_coeff_SPH2MESH(params.rolling_friction_coeffS2M);
    

    // set cohesion and adhesion model
	gran_sys.set_Cohesion_ratio(params.cohesion_ratio);
    gran_sys.set_Adhesion_ratio_S2W(params.adhesion_ratio_s2w);
    gran_sys.set_Adhesion_ratio_S2M(params.adhesion_ratio_s2m);

    // set gravity
    gran_sys.set_gravitational_acceleration(params.grav_X, params.grav_Y, params.grav_Z);

    // set time integrator
    gran_sys.set_fixed_stepSize(params.step_size);
    gran_sys.set_timeIntegrator(GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE);

    gran_sys.setOutputMode(params.write_mode);
    filesystem::create_directory(filesystem::path(params.output_dir));
    
    gran_sys.set_BD_Fixed(true);

    gran_sys.setVerbose(params.verbose);
    gran_sys.initialize();
  

    float curr_time = 0;
    ChVector<float> pos;
    ChVector<float> velo;
    ChVector<float> omega;

    unsigned int currframe = 0;
    unsigned int curstep = 0;
    double out_fps = 100;
    float frame_step = 1.f / out_fps;  // Duration of a frame
    unsigned int out_steps = frame_step / params.step_size;
    unsigned int total_frames = (unsigned int)((float)params.time_end * out_fps);
    std::cout << "out_steps " << out_steps << std::endl;

	while (curr_time < params.time_end) {

        gran_sys.advance_simulation(params.step_size);
        curr_time += params.step_size;

        pos   = apiSMC.getPosition(0);
        velo  = apiSMC.getVelo(0);
        // if frictionless, you can't call getAngVelo
        omega = apiSMC.getAngularVelo(0);

        if (curstep % out_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/step%06u", params.output_dir.c_str(), currframe++);
            //gran_sys.writeFile(std::string(filename));
            //gran_sys.write_meshes(std::string(filename));
        }

        printf("data %e, %e, %e, %e, %e, %e\n", curr_time, pos.x(), pos.z(), velo.x(), velo.z(), omega.y());
        curstep++;
		
    }


    return 0;
}

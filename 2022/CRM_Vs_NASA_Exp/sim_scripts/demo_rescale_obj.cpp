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
// Author: Wei Hu, Jason Zhou
// Chrono::FSI demo to show usage of VIPER rover models on SPH granular terrain
// This demo uses a plug-in VIPER rover model from chrono::models
// =============================================================================

#include "chrono_models/robot/viper/Viper.h"

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"
#include "chrono_thirdparty/filesystem/path.h"

// Chrono namespaces
using namespace chrono;
using namespace chrono::geometry;

int main(int argc, char* argv[]) {
    // Create a physical system and a corresponding FSI system
    ChSystemNSC sysMBS;

    // save obj
    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    std::string obj_path = ("nasa_viper_wheel.obj");
    double scale_ratio = 0.001;
    mmesh->LoadWavefrontMesh(obj_path, false, true);
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));  // scale to a different size
    mmesh->Transform(ChVector<>(0.1, 0, 0), ChMatrix33<>(1.0));  // scale to a different size
    ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 3.14/2));
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(Body_rot));  // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                              // if meshes are not watertight

    std::string filename = "nasa_viper_wheel_new.obj";
    std::vector<geometry::ChTriangleMeshConnected> meshes = {*mmesh};
    geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
    std::cout << "  we are good : "  << std::endl;

    return 0;
}
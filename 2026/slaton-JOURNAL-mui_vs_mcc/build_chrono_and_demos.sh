#!/usr/bin/env bash

set -e

# 0. args

SKIP_CHRONO=false

for arg in "$@"; do
    case "$arg" in
        --chrono)
            SKIP_CHRONO=true
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done


# 1. vars

ROOT="$(realpath ./..)"

CHRONO_SRC="$ROOT/chrono"
CHRONO_BUILD="$ROOT/chrono/build"

PROJECT_SRC="$(realpath ./)"
PROJECT_BUILD="$PROJECT_SRC/build"

VSG_DIR_PREFIX="$ROOT/_deps/vsg/lib/cmake"


# 2. build chrono

if [ "$SKIP_CHRONO" = false ]; then

    echo "Building Chrono..."

    mkdir -p "$CHRONO_BUILD"

    cmake -S "$CHRONO_SRC" -B "$CHRONO_BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_DEMOS=ON \
        -DBUILD_DEMOS_FSI=ON \
        -DBUILD_DEMOS_VEHICLE=ON \
        -DBUILD_DEMOS_VSG=ON \
        -DCH_ENABLE_MODULE_FSI=ON \
        -DCH_ENABLE_MODULE_FSI_SPH=ON \
        -DCH_ENABLE_MODULE_VEHICLE=ON \
        -DCH_ENABLE_MODULE_VEHICLE_MODELS=ON \
        -DCH_ENABLE_MODULE_POSTPROCESS=ON \
        -DCH_ENABLE_MODULE_VSG=ON \
        -DCH_USE_FSI_DOUBLE=OFF \
        -DCMAKE_PREFIX_PATH="$VSG_DIR_PREFIX"

    cmake --build "$CHRONO_BUILD" -j4

else

    echo "Skipping Chrono build."

fi


# 3. build custom project

echo "Building custom project..."

mkdir -p "$PROJECT_BUILD"

cmake -S "$PROJECT_SRC" -B "$PROJECT_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DChrono_DIR="$CHRONO_BUILD/cmake"

cmake --build "$PROJECT_BUILD" -j4
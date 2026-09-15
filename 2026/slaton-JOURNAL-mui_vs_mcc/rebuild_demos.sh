#!/usr/bin/env bash
set -e

ROOT="$(realpath .)"
PROJECT_BUILD="$ROOT/build"

cmake --build "$PROJECT_BUILD" -j"4"
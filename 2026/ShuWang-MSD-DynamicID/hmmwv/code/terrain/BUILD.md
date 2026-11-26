# Terrain query helper

`terrain_query.cpp` answers the height, normal and friction queries of the tires on the rough road. It reads the triangle mesh of the height-map patch in double precision and finds the triangle under a query point with a uniform grid. A point outside the mesh is passed to the Chrono rigid terrain. `rough_road.py` loads the library with ctypes and attaches it to a `ChTerrain` object through its SWIG pointer.

The library must be built against the same Chrono build as pychrono, because it uses the C++ classes of that build.

## Build (Windows, Visual Studio x64)

From a Visual Studio x64 command prompt, in this folder:

```
mkdir build
cl /nologo /O2 /MD /EHsc /std:c++14 /DNDEBUG /LD /I"<chrono>/src" /I"<chrono build>" /I"<eigen3>" ^
   /Fo"build\terrain_query.obj" terrain_query.cpp ^
   /link /OUT:"build\terrain_query.dll" /LIBPATH:"<chrono build>/lib/Release" ChronoEngine_vehicle.lib ChronoEngine.lib
```

`<chrono>` is the Chrono source tree, `<chrono build>` its build folder (it holds the generated configuration headers), `<eigen3>` the Eigen headers. Set `TERRAIN_LIBRARY` in `../config.py` to the built library.

The results of the paper used these compiler options.

## Build (macOS and Linux)

The source marks its three exported functions with `__declspec(dllexport)`. The flag `-D'__declspec(x)='` removes this Windows attribute; the functions are then exported by default. In this folder:

```
mkdir -p build
clang++ -O2 -std=c++14 -shared -fPIC -DNDEBUG '-D__declspec(x)=' \
   -I<chrono>/src -I<chrono build> -I<eigen3> terrain_query.cpp \
   -L<chrono build>/lib -lChronoEngine_vehicle -lChronoEngine -Wl,-rpath,<chrono build>/lib \
   -o build/terrain_query.dylib
```

On Linux, use `g++` and the output name `build/terrain_query.so`. Set `TERRAIN_LIBRARY` in `../config.py` to the built library.

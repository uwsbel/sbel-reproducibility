# Terrain query helper

`exact_query.cpp` is needed only for the uneven road (training motion P-R02 and its confirmation variant).
The tires of a Chrono vehicle ask the terrain for height, normal and friction at points under each wheel.
For a height-map patch the model answers these queries from the triangles of the patch mesh, in double precision.
Python cannot override these C++ calls in pychrono, so the queries are compiled into a small DLL.

## Build

- Use the same compiler and the same Chrono build as your pychrono (the helper exchanges `std::shared_ptr` objects with it).
- Compile `exact_query.cpp` as a shared library (C++17), with the Chrono include folders on the include path.
- Link against `ChronoEngine` and `ChronoEngine_vehicle`.
- On Windows with MSVC, for example:

```
cl /nologo /O2 /EHsc /MD /std:c++17 /LD exact_query.cpp /I <chrono include folders> ^
   /link <chrono lib folder>\ChronoEngine.lib <chrono lib folder>\ChronoEngine_vehicle.lib /OUT:exact_query.dll
```

- The DLL exports three C functions: `install_exact`, `exact_error` and `exact_counts`.
- Put the path of the DLL into `TERRAIN_LIBRARY` in `config.py`. The Chrono DLLs must be on the search path (importing pychrono first is enough).

## How the model uses it

`model.RoadTerrain` takes the vertices and triangles of the RigidTerrain patch in world coordinates and calls `install_exact`.
The helper replaces the object inside an empty `ChTerrain` proxy by its own terrain; the vehicle is then synchronized with that proxy.
A point is located with a uniform grid of bins and barycentric coordinates (tolerance 1e-10).
Points outside the mesh go to the original RigidTerrain.

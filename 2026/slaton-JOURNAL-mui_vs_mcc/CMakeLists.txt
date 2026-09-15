cmake_minimum_required(VERSION 3.18)
project(sbel_reproducibility_2026_MCC)

# ------------------------------------------------------------------------------
# Locate Chrono
# ------------------------------------------------------------------------------

set(Chrono_DIR "./../chrono/build/cmake")

find_package(Chrono
             REQUIRED
             COMPONENTS FSI VSG Vehicle Postprocess
             CONFIG)

# ------------------------------------------------------------------------------
# Build targets
# ------------------------------------------------------------------------------

set(DEMOS
    demo_FSI_ConePenetrometer
    demo_FSI_NormalBevameter
    demo_FSI_SlopedSingleWheel_RTFBenchmark
    demo_FSI_SlopedSingleWheel_Test
    demo_ROBOT_Lander_CRM_RTFBenchmark
    demo_ROBOT_Lander_CRM
)

foreach(demo ${DEMOS})
    if(${demo} MATCHES "demo_ROBOT_Lander_CRM")
        add_executable(${demo}
            ${demo}.cpp
            ../chrono/src/demos/robot/lander/model/Lander.cpp
        )
    else()
        add_executable(${demo} ${demo}.cpp)
    endif()

    target_include_directories(${demo} PRIVATE
        ${CHRONO_INCLUDE_DIRS}
        ../chrono/src
        ../chrono/src/demos/robot/lander
    )

    set_target_properties(${demo} PROPERTIES
        COMPILE_FLAGS "${CHRONO_CXX_FLAGS}"
        LINK_FLAGS    "${CHRONO_LINKER_FLAGS}"
    )

    target_link_libraries(${demo} PRIVATE ${CHRONO_TARGETS})

    # Make the demo's own source directory available at compile time so demos can
    # locate sibling helper files (e.g. generate_heightmap.py) regardless of the
    # working directory they are run from.
    target_compile_definitions(${demo} PRIVATE
        DEMO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

file(CREATE_LINK
    "${CMAKE_CURRENT_SOURCE_DIR}/../chrono/data"
    "${CMAKE_CURRENT_BINARY_DIR}/../data"
    SYMBOLIC)
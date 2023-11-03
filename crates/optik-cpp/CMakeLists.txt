cmake_minimum_required(VERSION 3.16)
project(optik)

find_package(Eigen3 REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CARGO_CMD cargo build --verbose)
  set(TARGET_DIR "debug")
else()
  set(CARGO_CMD cargo build --release --verbose)
  set(TARGET_DIR "release")
endif()

set(OPTIK_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPTIK_CPP_SRC "${OPTIK_CPP_ROOT}/src/lib.cpp")
set(OPTIK_CPP_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/liboptikcpp.a")

# Add a custom target to build the native library using cargo.
add_custom_target(optikcpp_build DEPENDS ${OPTIK_CPP_LIB})
add_custom_command(
  OUTPUT ${OPTIK_CPP_LIB}
  COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}"
          ${CARGO_CMD}
  WORKING_DIRECTORY ${OPTIK_CPP_ROOT})

# Add a nicer target for downstream use.
add_library(optik STATIC IMPORTED GLOBAL)
add_dependencies(optik optikcpp_build)
set_target_properties(
  optik
  PROPERTIES IMPORTED_LOCATION ${OPTIK_CPP_LIB}
             INTERFACE_LINK_LIBRARIES "Eigen3::Eigen;gfortran"
             INTERFACE_INCLUDE_DIRECTORIES "${OPTIK_CPP_ROOT}/include")
set_property(TARGET optik PROPERTY CXX_STANDARD 11)

add_executable(example ../examples/example.cpp)
target_link_libraries(example PUBLIC Eigen3::Eigen optik)

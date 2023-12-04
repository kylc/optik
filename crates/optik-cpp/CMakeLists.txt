cmake_minimum_required(VERSION 3.16)
project(optik VERSION 0.3.0)

find_package(Eigen3 REQUIRED)

set(CARGO_MANIFEST ${CMAKE_CURRENT_LIST_DIR}/Cargo.toml)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CARGO_CMD cargo build)
  set(TARGET_DIR "debug")
else()
  set(CARGO_CMD cargo build --release)
  set(TARGET_DIR "release")
endif()

set(OPTIK_CPP_SRC "${CMAKE_CURRENT_LIST_DIR}/src/lib.cpp")
set(OPTIK_CPP_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/liboptikcpp.a")

# Add a custom target to build the native library using cargo.
add_custom_target(optikcpp_build ALL DEPENDS ${OPTIK_CPP_LIB})
add_custom_command(
  OUTPUT ${OPTIK_CPP_LIB}
  COMMAND
    CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}"
    ${CARGO_CMD} --manifest-path ${CARGO_MANIFEST})

# Add a nicer target for downstream use.
add_library(optik STATIC src/lib.cpp)
add_dependencies(optik optikcpp_build)
target_include_directories(
  optik PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include>)
target_link_libraries(optik PUBLIC Eigen3::Eigen gfortran ${OPTIK_CPP_LIB})
set_property(TARGET optik PROPERTY CXX_STANDARD 11)

add_library(optik::optik ALIAS optik)

# Install
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(
  TARGETS optik
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE optik::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

configure_package_config_file(
  cmake/optikConfig.cmake.in "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(FILES include/optik.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

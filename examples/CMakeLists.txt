cmake_minimum_required(VERSION 3.16)
project(optik_examples)

include(FetchContent)

# If you're building within the optik source tree, then you can include the
# local build tree like this:
FetchContent_Declare(optik
  SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.."
  SOURCE_SUBDIR "crates/optik-cpp")
FetchContent_MakeAvailable(optik)

# If you want to pull the latest library via git, use FetchContent like this:
# FetchContent_Declare(
#   optik
#   GIT_REPOSITORY https://github.com/kylc/optik
#   GIT_TAG master
#   SOURCE_SUBDIR "crates/optik-cpp")
# FetchContent_MakeAvailable(optik)

# If you've installed optik (i.e. via `cmake --install`) then you can find it
# like this:
# find_package(optik REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE optik::optik)

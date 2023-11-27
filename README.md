# OptIK

<p>
    <img alt="MIT"    src="https://img.shields.io/badge/license-MIT-blue.svg">
    <img alt="Apache" src="https://img.shields.io/badge/license-Apache-blue.svg">
    <img alt="CI"     src="https://github.com/kylc/optik/actions/workflows/ci.yaml/badge.svg">
    <img alt="PyPI"   src="https://img.shields.io/pypi/v/optik-py.svg">
    <a href="https://zenodo.org/badge/latestdoi/696468110"><img src="https://zenodo.org/badge/696468110.svg" alt="DOI"></a>
</p>

A fast inverse kinematics solver for arbitrary serial chains, providing Rust and Python programming interfaces.

The implementation is similar to TRAC-IK [[1]] in that a nonlinear optimization problem is formulated and minimized using an SLSQP solver [[2]]. However, this work differs in a couple of ways:

- The gradient of the objective function is computed analytically. This is an immediate 2x performance improvement over finite difference approaches, because it requires only one evaluation of the forward kinematics as opposed to two (or more).
- Random restarting of the nonlinear solver is implemented in a work stealing parallel fashion, so that overall solve time is decreased thanks to the improved chance of finding a good seed.
- Random number generator seeds are carefully controlled in a way that produces deterministic results. (Note that this is only true in single-threaded mode, for now.)
- A parallel Newton's method solver is **not** included, because the performance of the full nonlinear problem is quite good on its own.

[1]: https://traclabs.com/projects/trac-ik/
[2]: https://github.com/jacobwilliams/slsqp

## Benchmark [^1]

We compare to TRAC-IK (via [tracikpy](https://github.com/mjd3/tracikpy)) by drawing a random valid joint configuration, mapping into Cartesian space with forward kinematics, and then asking each solver to generate an inverse kinematics solution using a random initial guess.

Note that this methodology differs from the original TRAC-IK benchmark which solves for configurations along a dense trajectory, meaning seeds are always relatively close. The benchmark shown below is more similar to a motion planning workload, in which samples are randomly drawn from a space with little knowledge of a nearby seed.

Timing is of a single inverse kinematics solve.

<img height="400" src="https://github.com/kylc/optik/assets/233860/d62b69d8-c2c1-45d8-91aa-24f4c3d98feb">

Additionally, we use the [ik_benchmarking](https://github.com/PickNikRobotics/ik_benchmarking) project (credit to PickNik Robotics) to compare against various solvers for the Franka Emika Panda robot using the MoveIt interfaces for each solver. OptIK is configured to return solutions with roughly equal tolerance to its closest competitor, TRAC-IK.

Timing is of a single inverse kinematics solve. Note the semi-log axes.

<img height="400" src="https://github.com/kylc/optik/assets/233860/2d809bcb-1505-4c6a-bf49-517b351b6ab5">

[^1]: as of https://github.com/kylc/optik/commit/3f324560b1a6ca5cfba2671e0180dd457ea1a28e

## Setup

### Python

``` sh
python3 -m pip install optik-py
```

Or, to install a prerelease version:

1. Download a recent `.whl` from [GitHub Releases](https://github.com/kylc/optik/releases)
2. Run `pip install optik-py<...>.whl` (replace `<...>` with the actual filename)
3. Test it: `python -c 'import optik'`

### C++ (CMake)

Include OptIK in your CMake project using `FetchContent`:

``` cmake
include(FetchContent)
FetchContent_Declare(
  optik
  GIT_REPOSITORY https://github.com/kylc/optik
  GIT_TAG master
  SOURCE_SUBDIR "crates/optik-cpp")
FetchContent_MakeAvailable(optik)

target_link_libraries(mylib PRIVATE optik::optik)
```

### Rust

Add OptIK as a Cargo dependency:

``` sh
cargo add optik
```

### Building Locally

``` sh
git clone git@github.com:kylc/optik.git

# Build the Rust library
cargo build --release

# Build a Python wheel
maturin build --release -m crates/optik-py/Cargo.toml

# Build the C++ example
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../examples
cmake --build .
```

## Usage Notes

- For workloads in which the distance between the solution and the seed are not important, you can use a high degree of parallelism (OptIK defaults to the number of CPU cores) to more quickly converge on a solution via parallel random restarting.

- For workloads such as Cartesian interpolation, it is important to find the solution closest to the seed to avoid joint-space discontinuities. While OptIK does not explicitly try to minimize this distance, the optimizer does generally converge to the nearest solution (subject to joint limits). Make sure to disable parallel random restarting (via `optik.set_parallelism(1)` or setting the environmental variable `RAYON_NUM_THREADS=1`) to ensure random seeds aren't used.

## References

P. Beeson and B. Ames, “TRAC-IK: An open-source library for improved solving of generic inverse kinematics,” in 2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids), Seoul, South Korea: IEEE, Nov. 2015, pp. 928–935. doi: 10.1109/HUMANOIDS.2015.7363472.

J. Williams, "Modern Fortran Edition of the SLSQP Optimizer", Version 1.4.1, 4 Jun 2023, https://github.com/jacobwilliams/slsqp

J. Solà, J. Deray, and D. Atchuthan, “A micro Lie theory for state estimation in robotics.” arXiv, Dec. 08, 2021. Accessed: Jul. 24, 2023. [Online]. Available: http://arxiv.org/abs/1812.01537

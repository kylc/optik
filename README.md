# OptIK

<p>
    <img alt="MIT"    src="https://img.shields.io/badge/license-MIT-blue.svg">
    <img alt="Apache" src="https://img.shields.io/badge/license-Apache-blue.svg">
    <img alt="CI"     src="https://github.com/kylc/optik/actions/workflows/ci.yaml/badge.svg">
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

## Benchmark

_NOTE: These benchmarks are provisional and conclusions should not yet be drawn._

We compare to TRAC-IK by drawing a random valid joint configuration, mapping into Cartesian space with forward kinematics, and then asking each solver to generate an inverse kinematics solution using a random initial guess.

Note that this methodology differs from the original TRAC-IK benchmark which solves for configurations along a dense trajectory, meaning seeds are always relatively close. The benchmark shown below is more similar to a motion planning workload, in which samples are randomly drawn from a space with little knowledge of a nearby seed.

Timing is of a single inverse kinematics solve.

<img height="400" src="https://user-images.githubusercontent.com/233860/270505593-cc08ba0d-416f-4288-b48c-83c5ffb0d6d9.png">

## Setup

1. Download a recent `.whl` from [GitHub Releases](https://github.com/kylc/optik/releases)
2. Run `pip install optik<...>.whl` (replace `<...>` with the actual filename)
3. Test it: `python -c 'import optik'`

## Usage

https://github.com/kylc/optik/blob/4ef93d4bbee9571bb9a7869e73dc8d911ed9079e/examples/example.py#L22-L30

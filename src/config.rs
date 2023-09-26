use pyo3::pyclass;

#[pyclass]
#[derive(Debug, Clone)]
pub enum SolutionMode {
    Quality,
    Speed,
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum GradientMode {
    Analytical,
    Numerical,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub gradient_mode: GradientMode,
    pub solution_mode: SolutionMode,
    pub max_time: f64,
    pub xtol_abs: f64,
    pub ftol_abs: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            gradient_mode: GradientMode::Analytical,
            solution_mode: SolutionMode::Speed,
            max_time: 0.1,
            xtol_abs: 1e-10,
            ftol_abs: 1e-5,
        }
    }
}

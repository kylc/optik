#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub enum SolutionMode {
    Quality = 1,
    Speed = 2,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub enum GradientMode {
    Analytical = 1,
    Numerical = 2,
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub gradient_mode: GradientMode,
    pub solution_mode: SolutionMode,
    pub max_time: f64,

    // Stopping criteria: |f(x)| < tol_f
    pub tol_f: f64,

    // Stopping criteria: |f(x_{n+1}) - f(x)| < tol_df
    pub tol_df: f64,

    // Stopping criteria: ||x_{n+1} - x_n|| < tol_dx
    pub tol_dx: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            gradient_mode: GradientMode::Analytical,
            solution_mode: SolutionMode::Speed,
            max_time: 0.1,
            tol_f: 1e-6,
            tol_df: -1.0,
            tol_dx: -1.0,
        }
    }
}

use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(C)]
pub enum SolutionMode {
    Quality = 1,
    Speed = 2,
}

impl FromStr for SolutionMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "quality" => Ok(SolutionMode::Quality),
            "speed" => Ok(SolutionMode::Speed),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub solution_mode: SolutionMode,

    /// Stopping criteria: elapsed_time > max_time
    ///
    /// Set to 0.0 for no time limit.
    pub max_time: f64,

    /// Stopping criteria: n_restarts > max_restarts
    ///
    /// Set to 0 for no restart limit.
    pub max_restarts: u64,

    /// Stopping criteria: f(x) < tol_f
    pub tol_f: f64,

    /// Stopping criteria: |f(x_{n+1}) - f(x)| < tol_df
    pub tol_df: f64,

    /// Stopping criteria: ||x_{n+1} - x_n|| < tol_dx
    pub tol_dx: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            solution_mode: SolutionMode::Speed,
            max_time: 0.1,
            max_restarts: u64::MAX,
            tol_f: 1e-6,
            tol_df: -1.0,
            tol_dx: -1.0,
        }
    }
}

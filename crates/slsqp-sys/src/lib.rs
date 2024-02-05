use std::ffi::*;

#[link(name = "slsqp")]
extern "C" {
    pub fn __slsqp_core_MOD_slsqp(
        m: *const c_int,
        meq: *const c_int,
        la: *const c_int,
        n: *const c_int,
        x: *mut c_double,
        xl: *const c_double,
        xu: *const c_double,
        f: *const c_double,
        c: *const c_double,
        g: *const c_double,
        a: *const c_double,
        acc: *mut c_double,
        iter: *mut c_int,
        mode: *mut c_int,
        w: *mut c_double,
        l_w: *const c_int,
        sdat: *mut c_int,
        ldat: *mut c_int,
        alphamin: *const c_double,
        alphamax: *const c_double,
        tolf: *const c_double,
        toldf: *const c_double,
        toldx: *const c_double,
        max_iter_ls: *const c_int,
        nnls_mode: *const c_int,
        infinite_bound: *const c_double,
    );
}

#[derive(Debug, Eq, PartialEq)]
pub enum IterationResult {
    Continue,
    Converged,
    Error,
}

pub struct SlsqpSolver {
    m: c_int,
    meq: c_int,
    la: c_int,
    n: c_int,
    xl: Vec<f64>,
    xu: Vec<f64>,
    c: Vec<f64>,
    a: Vec<f64>,
    acc: c_double,
    iter: c_int,
    mode: c_int,
    w: Vec<f64>,
    slsqpb_data: Vec<i32>,
    linmin_data: Vec<i32>,
    tolf: c_double,
    toldf: c_double,
    toldx: c_double,

    f: f64,
    g: Vec<f64>,

    init: bool,
}

impl SlsqpSolver {
    pub fn new(n: usize) -> SlsqpSolver {
        let n1 = n + 1;
        let m = 0;
        let meq = 0;
        let mineq = m - meq + 2 * n1;

        // Allocate the working area, copied from slsqp_module.f90
        let l_w = n1*(n1+1) + meq*(n1+1) + mineq*(n1+1)  // for lsq
                 + (n1-meq+1)*(mineq+2) + 2*mineq        // for lsi
                 + (n1+mineq)*(n1-meq) + 2*meq + n1      // for lsei
                  + n1*n/2 + 2*m + 3*n +3*n1 + 100; // for slsqpb

        SlsqpSolver {
            m: m as c_int,
            meq: meq as c_int,
            la: 1.max(m) as c_int,
            n: n as c_int,
            xl: vec![f64::NEG_INFINITY; n],
            xu: vec![f64::INFINITY; n],
            c: vec![0.0; 1.max(m)],
            a: vec![0.0; 1.max(m) * n1],
            acc: 0.0, // TODO
            iter: i32::MAX,
            mode: 0,
            w: vec![0.0; l_w],
            slsqpb_data: vec![0; 112], // TODO: allocate from slsqpb_data structue,
            linmin_data: vec![0; 144], // TODO: allocate from linmin_data structue,
            tolf: -1.0,
            toldf: -1.0,
            toldx: -1.0,

            f: 0.0,
            g: vec![0.0; n],

            init: false,
        }
    }

    pub fn set_tol_f(&mut self, ftol: f64) {
        self.acc = 0.1 * ftol; // TODO
        self.tolf = ftol;
    }

    pub fn set_tol_df(&mut self, dftol: f64) {
        self.toldf = dftol;
    }

    pub fn set_tol_dx(&mut self, dxtol: f64) {
        self.toldx = dxtol;
    }

    pub fn set_lb(&mut self, lb: &[f64]) {
        self.xl = lb.to_vec();
    }

    pub fn set_ub(&mut self, ub: &[f64]) {
        self.xu = ub.to_vec();
    }

    pub fn cost(&self) -> f64 {
        self.f
    }

    pub fn iterate<O, G>(&mut self, x: &mut [f64], objective: O, gradient: G) -> IterationResult
    where
        O: Fn(&[f64]) -> f64,
        G: Fn(&[f64], &mut [f64]),
    {
        if self.mode == 1 || !self.init {
            self.f = objective(x);
        }

        if self.mode == -1 || !self.init {
            gradient(x, &mut self.g);
        }

        unsafe {
            __slsqp_core_MOD_slsqp(
                &self.m,
                &self.meq,
                &self.la,
                &self.n,
                x.as_mut_ptr(),
                self.xl.as_ptr(),
                self.xu.as_ptr(),
                &self.f,
                self.c.as_ptr(),
                self.g.as_ptr(),
                self.a.as_ptr(),
                &mut self.acc,
                &mut self.iter,
                &mut self.mode,
                self.w.as_mut_ptr(),
                &(self.w.len() as i32),
                self.slsqpb_data.as_mut_ptr(),
                self.linmin_data.as_mut_ptr(),
                &0.1,
                &1.0,
                &self.tolf,
                &self.toldf,
                &self.toldx,
                &0,
                &1,
                &0.0,
            );
        }

        if !self.init {
            self.init = true;
            self.mode = 1;
        }

        match self.mode {
            -1 | 1 => IterationResult::Continue,
            0 => IterationResult::Converged,
            _ => IterationResult::Error,
        }
    }
}

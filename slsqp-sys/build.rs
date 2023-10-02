use std::{env, path::Path, process::Command};

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let target = env::var("TARGET").unwrap();

    for src_file in [
        "slsqp_kinds.F90",
        "slsqp_support.f90",
        "bvls_module.f90",
        "slsqp_core.f90",
        "slsqp_module.f90",
    ] {
        assert!(Command::new("gfortran")
            .args([
                "-c",
                &format!("slsqp/src/{}", src_file),
                "-O3",
                "-funroll-loops",
                "-fPIC",
                "-fmax-errors=1",
                "-fcoarray=single",
                "-fimplicit-none",
                "-Werror=implicit-interface",
                "-ffree-form",
            ])
            .arg("-J")
            .arg(&out_dir)
            .arg(format!("-I{}", &out_dir))
            .arg("-o")
            .arg(&format!("{}/{}.o", out_dir, src_file))
            .status()
            .unwrap()
            .success());
    }

    assert!(Command::new("ar")
        .args([
            "crus",
            "libslsqp.a",
            "slsqp_kinds.F90.o",
            "slsqp_support.f90.o",
            "bvls_module.f90.o",
            "slsqp_core.f90.o",
            "slsqp_module.f90.o",
        ])
        .current_dir(Path::new(&out_dir))
        .status()
        .unwrap()
        .success());

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=slsqp");

    let libgfortran_path = String::from_utf8(
        Command::new("gfortran")
            .arg(&format!(
                "-print-file-name=libgfortran.{}",
                if target.contains("darwin") {
                    "dylib"
                } else {
                    "so"
                }
            ))
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();

    assert!(!libgfortran_path.is_empty(), "failed to find libgfortran");

    println!(
        "cargo:rustc-link-search=native={}",
        Path::new(&libgfortran_path)
            .parent()
            .unwrap()
            .to_string_lossy()
    );
    println!("cargo:rustc-link-lib=dylib={}", "gfortran");
}

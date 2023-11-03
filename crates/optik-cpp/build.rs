fn main() {
    let eigen = pkg_config::Config::new().probe("eigen3").unwrap();

    cc::Build::new()
        .cpp(true)
        .std("c++11")
        .file("src/lib.cpp")
        .includes(eigen.include_paths)
        .include("include")
        .compile("optikcpp");

    println!("cargo:rerun-if-changed=optik.hpp");
    println!("cargo:rerun-if-changed=optik.cpp");
    println!("cargo:rustc-link-lib=static=optikcpp");
}

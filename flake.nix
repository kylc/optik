{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";

    naersk.url = "github:nix-community/naersk/master";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;

      perSystem = { config, self', pkgs, lib, system, ... }:
        let
          naersk-lib = pkgs.callPackage inputs.naersk { };

          # Use the nightly channel so that we can use the faster cranelift
          # backend for development builds.
          rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
            extensions =
              [ "rust-src" "rust-analyzer" "clippy" "rustc-codegen-cranelift" ];
          };

          # Some packages are required for building the native dependencies.
          rustBuildInputs = with pkgs; [
            patchelf
            cmake
            gcc
            mold
            openssl
            pkg-config
            python3
          ];
        in {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
          };

          packages.default = naersk-lib.buildPackage ./.;

          devShells.default = pkgs.mkShell {
            name = "optik-dev";
            buildInputs = rustBuildInputs;
            nativeBuildInputs =
              [ rustToolchain pkgs.maturin pkgs.python3Packages.numpy ];
            shellHook = ''
              # For rust-analyzer 'hover' tooltips to work.
              export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library";
            '';

            LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
          };
        };
    };
}

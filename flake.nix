{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";

    naersk.url = "github:nix-community/naersk";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;

      perSystem = { config, self', pkgs, lib, system, ... }:
        let
          naersk-lib = pkgs.callPackage inputs.naersk { };
          rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
              "clippy"
              "rustc-codegen-cranelift-preview"
            ];
          };
        in rec {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
          };

          packages.default = naersk-lib.buildPackage {
            src = ./.;
            nativeBuildInputs = with pkgs; [ gcc cmake lld openssl python3 ];
          };

          devShells.default = pkgs.mkShell {
            name = "optik-dev";
            nativeBuildInputs = with pkgs;
              packages.default.nativeBuildInputs ++ [
                rustToolchain
                cargo-nextest
                cargo-outdated

                # For C++ bindings
                eigen

                # For Python bindings
                maturin
                pyright
                python3Packages.numpy
                python3Packages.black
                python3Packages.ruff
                python3Packages.venvShellHook
              ];

            # Create a virtualenv for Python development with maturin.
            venvDir = "./.venv";
            postShellHook = ''
              # For rust-analyzer 'hover' tooltips to work.
              export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library";
            '';

            LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
          };
        };
    };
}

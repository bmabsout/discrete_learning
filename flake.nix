{
  description = "Boolean differentiation formalization";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    typix.url = "github:loqusion/typix";
  };

  outputs = { self, nixpkgs, flake-utils, typix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        python = pkgs.python3.withPackages (ps: [ps.pytorch ps.torchvision]);
     
      in
      {

        devShells.default = typix.lib.${system}.devShell {
          packages = [
            # Add Coq packages
            pkgs.coq
            pkgs.coqPackages.coq-lsp
            python
          ];
        };
      }
    );
}

{
  description = "Model library + runner: llama.cpp with CUDA, driven by models.py";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      llama-cpp-cuda = pkgs.llama-cpp.override { cudaSupport = true; };
      pythonEnv = pkgs.python3.withPackages (ps: [ ps.pyyaml ]);
    in {
      # This allows you to run 'nix develop' to enter a shell with everything models.py needs
      devShells.${system}.default = pkgs.mkShell {
        name = "models-shell";

        buildInputs = [
          # CUDA-enabled llama.cpp (provides llama-cli, llama-server, llama-bench, ...)
          llama-cpp-cuda
          # Runs models.py (needs PyYAML to read library/*.yml)
          pythonEnv
          pkgs.git
        ];

        shellHook = ''
          export ENV_NAME="$ENV_NAME models"
          echo "--- models dev shell ---"
          echo "models.py list | run [name] | download [name]   (names come from library/*.yml)"
        '';
      };

      packages.${system} = {
        llama-cpp = llama-cpp-cuda;
        python = pythonEnv;
      };
    };
}

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
          # pi.dev console harness for LLMs (`pi` command)
          pkgs.pi-coding-agent
        ];

        shellHook = ''
          export ENV_NAME="$ENV_NAME models"
          # Project-local pi.dev config, not ~/.pi/agent: keeps its model
          # config (.pi/agent/models.json) tracked in the repo and its
          # runtime state (auth.json, sessions/, npm/, ...) out of it.
          export PI_CODING_AGENT_DIR="$PWD/.pi/agent"
          mkdir -p "$PI_CODING_AGENT_DIR"

          echo "--- models dev shell ---"
          echo "models.py list | run [name] | server [name] | download [name]   (names come from library/*.yml)"
          echo "pi is configured with a local-llamacpp/local model -> whatever 'models.py server' is currently running"
          echo "Open WebUI: docker compose up -d   (see docker-compose.yml)"
        '';
      };

      packages.${system} = {
        llama-cpp = llama-cpp-cuda;
        python = pythonEnv;
      };
    };
}

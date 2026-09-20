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
          # Open WebUI (`open-webui serve`) -- chat UI, talks to llama-server's
          # OpenAI-compatible API. See OPENAI_API_BASE_URL note in shellHook.
          pkgs.open-webui
        ];

        shellHook = ''
          export ENV_NAME="$ENV_NAME models"
          # Project-local pi.dev config, not ~/.pi/agent: keeps its model
          # config (.pi/agent/models.json) tracked in the repo and its
          # runtime state (auth.json, sessions/, npm/, ...) out of it.
          export PI_CODING_AGENT_DIR="$PWD/.pi/agent"
          mkdir -p "$PI_CODING_AGENT_DIR"

          # Open WebUI: project-local data dir (its own sqlite db, uploads,
          # etc.) instead of ~/.open-webui, and env vars that pre-seed its
          # "OpenAI API" connection to point at models.py server's default
          # (127.0.0.1:8080). NOTE: open-webui has a known issue where these
          # env vars get silently ignored/overwritten once its db already
          # has a value persisted -- if the UI doesn't pick it up, set it
          # once by hand in Admin Settings -> Connections instead.
          export DATA_DIR="$PWD/.open-webui/data"
          mkdir -p "$DATA_DIR"
          export OPENAI_API_BASE_URL="http://127.0.0.1:8080/v1"
          export OPENAI_API_KEY="none"
          export ENABLE_OLLAMA_API="false"

          echo "--- models dev shell ---"
          echo "models.py list | run [name] | server [name] | download [name]   (names come from library/*.yml)"
          echo "pi is configured with a local-llamacpp/local model -> whatever 'models.py server' is currently running"
          echo "open-webui serve --port 3000   (llama-server already uses 8080; UI expects the OpenAI connection at 127.0.0.1:8080/v1)"
        '';
      };

      packages.${system} = {
        llama-cpp = llama-cpp-cuda;
        python = pythonEnv;
      };
    };
}

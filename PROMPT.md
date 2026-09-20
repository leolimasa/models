# Optimizing a model's library/*.yml

You are tuning `library/<slug>.yml` for one specific model on this machine.
This document explains the architecture — the `.yml` schema and how
`models.py` uses it — so you can test changes correctly. It intentionally
says nothing about *how* to optimize (which flags, which values); that's
your job to work out for the model and hardware at hand.

## Repo layout

```
library/<slug>.yml   one file per model: metadata + run config
weights/<file>.gguf  downloaded model weights (gitignored)
models.py            CLI: list / run / download
flake.nix            `nix develop` shell providing llama.cpp (CUDA) + Python
```

## The .yml schema

```yaml
name: Human-readable model name
file: exact-filename.gguf        # expected under weights/
url: https://...                 # source to download `file` from
runner: llamacpp                 # which runner backend to use
llamacpp:                        # config block, one per possible runner
  mode: cli                      # cli | server
  ctx: 8192                      # context window
  cache_type: ""                 # optional: e.g. q8_0 (KV cache quantization)
  extra_args: ["-flag", "value"] # raw args appended to the llama.cpp invocation
```

`runner` selects which config block `models.py` reads (`llamacpp` today —
see `RUNNERS` in `models.py`; a future runner would add its own block
alongside `llamacpp`, keyed by its own name). Everything under that block is
specific to the runner and passed through more or less as-is.

## How models.py runs a model

`models.py run <slug>` does, in order:

1. Load `library/<slug>.yml`.
2. Ensure `weights/<file>` exists, downloading from `url` if not.
3. Build the command line for `entry["runner"]` (`build_llamacpp_command` for
   `llamacpp`) and `exec` it — the model process replaces `models.py`.

For the `llamacpp` runner, the command is assembled as:

```
llama-cli|llama-server -m <weights/file> -fa on -c <ctx> -t <threads> -tb <threads>
  [--cache-type-k <cache_type> --cache-type-v <cache_type>]   # if cache_type set
  <extra_args from the .yml>
  <anything passed after -- on the command line>
  <mode-specific flags: -n/-cnv for cli, --host/--port for server>
```

So `extra_args` in the `.yml` is where any tuning flags go — anything
`llama-cli`/`llama-server` accepts can be listed there
(`--gpu-layers`, `--tensor-split`, `--split-mode`, `--n-cpu-moe`, etc. are
all just examples of things that exist; this file takes no position on which
of them matter here).

Note what's *not* set explicitly: `-ngl` is left to llama.cpp's own default
(`-ngl auto` with `-fit on`), which sizes GPU offload across all detected
GPUs on its own. It only gets set if you put it in `extra_args` yourself.

### Env var overrides (apply to `run` only)

These override the `.yml`'s `llamacpp` block for a single invocation, without
editing the file — useful while iterating:

| var          | overrides             |
|--------------|------------------------|
| `MODE`       | `llamacpp.mode`        |
| `CTX`        | `llamacpp.ctx`         |
| `CACHE_TYPE` | `llamacpp.cache_type`  |
| `THREADS`    | CPU thread count (default 8, no `.yml` equivalent) |
| `PROMPT`     | one-shot prompt in cli mode, runs non-interactively and exits |
| `NPREDICT`   | max tokens to generate |
| `PORT`       | server port |

Example: `CTX=4096 PROMPT="hello" NPREDICT=32 ./models.py run <slug>` does a
fast, non-interactive, low-context test run without touching the `.yml`.

### `--` passthrough

Anything after `--` on the command line is appended to the built command,
after `extra_args`:

```
./models.py run <slug> -- -ngl 20 -sm none
```

This is the fastest way to try a flag before committing it to the `.yml`'s
`extra_args`.

## Adding tools to flake.nix

If tuning or benchmarking this model needs a tool that isn't already in the
`nix develop` shell (e.g. `llama-bench`, which ships alongside `llama-cli` in
the same package, or anything else — a profiler, `nvtop`, a JSON/YAML CLI
tool), add it to `buildInputs` in `flake.nix` rather than relying on it being
present on the host system. That keeps the environment reproducible for
whoever runs this next.

## Verifying a change

After editing a `.yml`, actually run the model (a short `PROMPT=... run` is
enough) and check it starts without error — an out-of-memory or allocation
failure at real context size is a common failure mode that a quick load
won't necessarily catch if you test at a much smaller `CTX` than the `.yml`
specifies.

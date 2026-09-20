#!/usr/bin/env python3
"""Manage and run models from library/*.yml.

Usage:
    models.py list
    models.py run [name] [-- extra llama.cpp args...]
    models.py server [name] [-- extra llama.cpp args...]
    models.py download [name|url]

<name> is a library/*.yml filename (without extension), e.g. "qwen3.5-9b".
If omitted from run/server/download, an interactive picker lets you choose
one (arrow keys, vim j/k, or type a digit to jump to that row; Enter
confirms).

`server` always runs in server mode regardless of the .yml's `mode` or the
MODE env var. For the llamacpp runner that's `llama-server`, which exposes
an OpenAI-compatible API on localhost (see PORT below).

If `download` is given a URL instead of a name, a new library/<slug>.yml is
created for it (slug derived from the URL's filename) before downloading.

Env vars for `run`/`server` (override the .yml):
    MODE        cli | server                     (run only; server always forces server)
    CTX         context window                   (default: from .yml, else 8192)
    THREADS     CPU threads                       (default: 8)
    CACHE_TYPE  KV-cache quantization, e.g. q8_0  (default: none = f16)
    PROMPT      one-shot prompt (run, cli mode)   -> runs non-interactively and exits
    NPREDICT    max tokens to generate            (default: -1 = until stop)
    PORT        server port                       (default: 8080)

NGL is intentionally not set here: llama.cpp defaults to `-ngl auto` with
`-fit on`, which sizes the offload across all detected GPUs on its own.
Pass -ngl yourself after `--` to override it.
"""
import argparse
import curses
import os
import re
import sys
import urllib.parse
import urllib.request
import yaml

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def list_entries():
    if not os.path.isdir(LIBRARY_DIR):
        return []
    slugs = sorted(f[:-4] for f in os.listdir(LIBRARY_DIR) if f.endswith(".yml"))
    entries = []
    for slug in slugs:
        with open(os.path.join(LIBRARY_DIR, f"{slug}.yml")) as f:
            entries.append((slug, yaml.safe_load(f)))
    return entries


def is_downloaded(entry):
    return os.path.isfile(os.path.join(WEIGHTS_DIR, entry["file"]))


def load_entry(name):
    path = os.path.join(LIBRARY_DIR, f"{name}.yml")
    if not os.path.isfile(path):
        available = sorted(slug for slug, _ in list_entries())
        sys.exit(f"error: no {path}\navailable models: {', '.join(available)}")
    with open(path) as f:
        return yaml.safe_load(f)


def is_url(s):
    return urllib.parse.urlparse(s).scheme in ("http", "https")


def slugify(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower() or "model"


def create_entry_from_url(url):
    """Registers a new library/<slug>.yml for a model at `url`. Returns the slug.

    Re-running with the same url is idempotent: it returns the existing slug
    instead of erroring, so `download <url>` can be run more than once.
    """
    for slug, entry in list_entries():
        if entry.get("url") == url:
            return slug

    filename = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
    if not filename:
        sys.exit(f"error: couldn't determine a filename from url: {url}")

    name = filename[:-len(".gguf")] if filename.lower().endswith(".gguf") else filename
    slug = slugify(name)
    yml_path = os.path.join(LIBRARY_DIR, f"{slug}.yml")
    if os.path.isfile(yml_path):
        sys.exit(
            f"error: library/{slug}.yml already exists but points at a different url "
            f"than {url} -- resolve the slug collision manually"
        )

    os.makedirs(LIBRARY_DIR, exist_ok=True)
    data = {
        "name": name,
        "file": filename,
        "url": url,
        "runner": "llamacpp",
        "llamacpp": {"mode": "cli", "ctx": 8192, "extra_args": []},
    }
    with open(yml_path, "w") as f:
        f.write(
            "# Auto-generated from a URL. These defaults are generic and unverified --\n"
            "# check GPU fit and tune ctx/extra_args (see other library/*.yml for examples).\n"
        )
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"created library/{slug}.yml")
    return slug


def print_list(entries):
    for i, (slug, entry) in enumerate(entries):
        marker = " [downloaded]" if is_downloaded(entry) else ""
        print(f"{i:3d}  {slug:<24s} {entry['name']}{marker}")


def pick_model(entries):
    if not entries:
        sys.exit("error: library/ has no models")

    def run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        cursor = 0
        typed = ""
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, "Select a model:")
            for i, (slug, entry) in enumerate(entries):
                marker = " [downloaded]" if is_downloaded(entry) else ""
                line = f"{i:3d}  {slug:<24s} {entry['name']}{marker}"
                attr = curses.A_REVERSE if i == cursor else curses.A_NORMAL
                prefix = "> " if i == cursor else "  "
                stdscr.addstr(i + 2, 0, prefix + line, attr)
            stdscr.addstr(
                len(entries) + 3, 0,
                "↑/↓ or j/k move · type digits to jump · Enter to confirm · q to quit",
            )
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(entries)
                typed = ""
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(entries)
                typed = ""
            elif ord("0") <= key <= ord("9"):
                typed += chr(key)
                if int(typed) < len(entries):
                    cursor = int(typed)
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return entries[cursor][0]
            elif key in (27, ord("q")):  # Esc, q
                return None
            else:
                typed = ""

    slug = curses.wrapper(run)
    if slug is None:
        sys.exit(1)
    return slug


def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"\rdownloaded {downloaded / 1e6:.1f} MB", end="", flush=True)
        return
    pct = min(downloaded / total_size * 100, 100)
    print(
        f"\r{pct:5.1f}%  {downloaded / 1e6:9.1f} / {total_size / 1e6:.1f} MB",
        end="",
        flush=True,
    )


def ensure_weights(entry):
    """Downloads weights if missing. Returns (path, already_had_it)."""
    dest = os.path.join(WEIGHTS_DIR, entry["file"])
    if os.path.isfile(dest):
        return dest, True

    url = entry.get("url")
    if not url or url == "TODO":
        sys.exit(f"error: {entry['file']} is missing and no download url is set in its .yml")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    tmp = dest + ".part"
    print(f"downloading {entry['file']} ...")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=report_progress)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print()
    os.rename(tmp, dest)
    return dest, False


def build_llamacpp_command(entry, model_path, passthrough, force_mode=None):
    cfg = entry.get("llamacpp", {})
    mode = force_mode or os.environ.get("MODE", cfg.get("mode", "cli"))
    ctx = os.environ.get("CTX", str(cfg.get("ctx", 8192)))
    threads = os.environ.get("THREADS", "8")
    cache_type = os.environ.get("CACHE_TYPE", cfg.get("cache_type", ""))

    common = [
        "-m", model_path,
        "-fa", "on",
        "-c", str(ctx),
        "-t", str(threads),
        "-tb", str(threads),
    ]
    if cache_type:
        common += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
    common += [str(a) for a in cfg.get("extra_args", [])]
    common += passthrough

    if mode == "server":
        port = os.environ.get("PORT", "8080")
        return ["llama-server", *common, "--host", "0.0.0.0", "--port", port]
    elif mode == "cli":
        npredict = os.environ.get("NPREDICT", "-1")
        prompt = os.environ.get("PROMPT")
        if prompt is not None:
            return ["llama-cli", *common, "-n", npredict, "-st", "--simple-io", "-p", prompt]
        return ["llama-cli", *common, "-n", npredict, "-cnv"]
    else:
        sys.exit(f"error: unknown mode '{mode}' (use 'cli' or 'server')")


RUNNERS = {
    "llamacpp": build_llamacpp_command,
}


def resolve_name(name):
    if name:
        return name
    return pick_model(list_entries())


def cmd_list(args):
    entries = list_entries()
    if not entries:
        print("no models in library/")
        return
    print_list(entries)


def cmd_download(args):
    if args.model and is_url(args.model):
        name = create_entry_from_url(args.model)
    else:
        name = resolve_name(args.model)
    entry = load_entry(name)
    _, already_had_it = ensure_weights(entry)
    if already_had_it:
        print(f"{entry['name']}: already downloaded")
    else:
        print(f"{entry['name']}: downloaded")


def start(model_arg, passthrough, label, force_mode=None):
    name = resolve_name(model_arg)
    entry = load_entry(name)
    runner = entry.get("runner")
    if runner not in RUNNERS:
        sys.exit(f"error: unsupported runner '{runner}' (known: {', '.join(RUNNERS)})")

    model_path, _ = ensure_weights(entry)
    command = RUNNERS[runner](entry, model_path, passthrough, force_mode=force_mode)

    print(f"=== models.py {label} ===")
    print(f"  model : {entry['name']}")
    print(f"  cmd   : {' '.join(command)}")
    print("======================")
    os.execvp(command[0], command)


def cmd_run(args):
    start(args.model, args.passthrough, "run")


def cmd_server(args):
    start(args.model, args.passthrough, "server", force_mode="server")


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        passthrough = argv[i + 1:]
        argv = argv[:i]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="list all models")
    p_list.set_defaults(func=cmd_list)

    p_run = subparsers.add_parser("run", help="run a model (interactive picker if name omitted)")
    p_run.add_argument("model", nargs="?", default=None)
    p_run.set_defaults(func=cmd_run)

    p_server = subparsers.add_parser("server", help="run a model in server mode (interactive picker if name omitted)")
    p_server.add_argument("model", nargs="?", default=None)
    p_server.set_defaults(func=cmd_server)

    p_download = subparsers.add_parser("download", help="download a model (interactive picker if name omitted)")
    p_download.add_argument("model", nargs="?", default=None)
    p_download.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    args.passthrough = passthrough
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the training dashboard on a Google Colab GPU and use it from here.

    python colab_dashboard.py login                 # once: sign in to Colab
    python colab_dashboard.py up --gpu T4           # start, then open the dashboard
    python colab_dashboard.py up --gpu L4 -- --envs 32   # pass dashboard.py options
    python colab_dashboard.py status
    python colab_dashboard.py logs
    python colab_dashboard.py pull                  # copy the VM's models here now
    python colab_dashboard.py down                  # pull, then release the VM

The Colab CLI does not run on Windows, so every colab and ssh command runs in a
Docker container (colab/Dockerfile). `up` allocates a Colab VM, copies the
project and the Linux build of dodge_native onto it, starts dashboard.py there,
and keeps an SSH tunnel open in a background container so the remote dashboard
is served at http://127.0.0.1:8766. The same container copies the VM's models
into artifacts/colab/models every ten minutes, since the VM's disk is lost
with the VM; copy a checkpoint into artifacts/models to train on it locally.

The first `up` on a fresh VM also seeds it with artifacts/models (less the
watch snapshots), rewards.json, game-config.json and training.json. Later
`up`s against the same VM refresh only the code, so the VM's own checkpoints
and settings are kept.
"""

import argparse
import hashlib
import shlex
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGING = ROOT / "artifacts" / "colab"
IMAGE = "dodge-colab"
HOME_VOLUME = "dodge-colab-home"
CARGO_VOLUME = "dodge-colab-cargo"
TUNNEL_CONTAINER = "dodge-colab-tunnel"
MATURIN_IMAGE = "ghcr.io/pyo3/maturin:latest"
REMOTE = "/content/DodgeAI"
SEED_FILES = ("rewards.json", "game-config.json", "training.json")
# Paths the VM does not need; artifacts/ also holds checkpoints that would
# overwrite the VM's own on a refresh.
CODE_EXCLUDES = ("artifacts/", "docs/", ".superpowers/", ".claude/")


def docker(*args, check=True, capture=False, stdin=None):
    return subprocess.run(["docker", *args], check=check, stdin=stdin,
                          capture_output=capture, text=capture)


def in_container(script, *, tty=False, ports=(), name=None, detach=False,
                 check=True, capture=False):
    """Run a bash script in the CLI container, with the repo at /work."""
    flags = ["-d"] if detach else ["--rm"] + (["-it"] if tty else [])
    if name:
        flags += ["--name", name]
    for port in ports:
        flags += ["-p", port]
    return docker("run", *flags, "-v", f"{HOME_VOLUME}:/root", "-v", f"{ROOT}:/work",
                  "--entrypoint", "bash", IMAGE, "-c", script,
                  check=check, capture=capture)


def ssh_host(session):
    return f"colab-{session}"


def ensure_image():
    docker("build", "-q", "-t", IMAGE, str(ROOT / "colab"))


def ensure_ssh(session):
    """Give the container an SSH key and a config entry for the session's VM."""
    host = ssh_host(session)
    entry = (f"Host {host}\n"
             f"  ProxyCommand /usr/local/bin/colab ssh --proxy-mode -s {session}\n"
             f"  User root\n"
             f"  StrictHostKeyChecking no\n"
             f"  UserKnownHostsFile /dev/null\n"
             f"  LogLevel ERROR\n")
    in_container(
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh\n"
        "[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519\n"
        f"grep -q '^Host {host}$' ~/.ssh/config 2>/dev/null"
        f" || printf %s {shlex.quote(entry)} >> ~/.ssh/config\n")


def ensure_engine(engine_repo, rebuild):
    """Build the manylinux dodge_native wheel from the dodge-ai-3 checkout."""
    STAGING.mkdir(parents=True, exist_ok=True)
    wheels = sorted(STAGING.glob("dodge_native-*.whl"))
    if wheels and not rebuild:
        return wheels[-1]
    if not (engine_repo / "native" / "crates" / "dodge-python").is_dir():
        sys.exit(f"dodge-ai-3 not found at {engine_repo}; pass --engine-repo")
    for wheel in wheels:
        wheel.unlink()
    print(f"Building the Linux engine from {engine_repo}")
    docker("volume", "create", CARGO_VOLUME, capture=True)
    docker("run", "--rm", "-v", f"{engine_repo.resolve()}:/io:ro",
           "-v", f"{STAGING}:/out", "-v", f"{CARGO_VOLUME}:/cache",
           "-e", "CARGO_TARGET_DIR=/cache/target", "-e", "CARGO_HOME=/cache/cargo",
           MATURIN_IMAGE, "build", "--release",
           "-m", "native/crates/dodge-python/Cargo.toml", "-o", "/out")
    return sorted(STAGING.glob("dodge_native-*.whl"))[-1]


def write_manifests(wheel):
    """List the files to copy: code on every `up`, the seed on a fresh VM."""
    # The working tree, not the last commit: uncommitted files are what the
    # local dashboard runs too. Settings files are seed, so a refresh leaves
    # the VM's own alone.
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout.split("\0")
    code = [path for path in dict.fromkeys(listed)
            if path and not path.startswith(CODE_EXCLUDES) and path not in SEED_FILES
            and not path.endswith(".pdf") and (ROOT / path).is_file()]
    code += [path.relative_to(ROOT).as_posix()
             for path in (ROOT / "web" / "dist").rglob("*") if path.is_file()]
    # newline="\n": these are read by tar and cmp on Linux, where a Windows
    # \r would become part of every path.
    (STAGING / "engine.sha256").write_text(
        hashlib.sha256(wheel.read_bytes()).hexdigest() + "\n", newline="\n")
    code += [wheel.relative_to(ROOT).as_posix(), "artifacts/colab/engine.sha256"]

    models = ROOT / "artifacts" / "models"
    seed = [path.relative_to(ROOT).as_posix() for path in models.rglob("*")
            if path.is_file() and "watch" not in path.relative_to(models).parts]
    seed += [name for name in SEED_FILES if (ROOT / name).is_file()]

    (STAGING / "code.txt").write_text("\n".join(code) + "\n", newline="\n")
    (STAGING / "seed.txt").write_text("\n".join(seed) + "\n", newline="\n")
    return len(code), len(seed)


def session_exists(session):
    listing = in_container("colab sessions", capture=True, check=False)
    return any(line.split()[:1] == [session] or f" {session} " in f" {line} "
               for line in listing.stdout.splitlines())


def start_tunnel(session, port):
    docker("rm", "-f", TUNNEL_CONTAINER, check=False, capture=True)
    in_container(f"bash /work/colab/tunnel.sh {ssh_host(session)}",
                 name=TUNNEL_CONTAINER, detach=True, ports=(f"127.0.0.1:{port}:8765",))


def wait_for_dashboard(url, seconds=60):
    import urllib.request
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/state", timeout=5):
                return True
        except OSError:
            time.sleep(2)
    return False


def cmd_login(args):
    ensure_image()
    in_container("colab usage", tty=True)


def cmd_up(args):
    ensure_image()
    wheel = ensure_engine(args.engine_repo, args.rebuild_engine)
    code_count, seed_count = write_manifests(wheel)
    ensure_ssh(args.session)
    host = ssh_host(args.session)

    if session_exists(args.session):
        print(f"Reusing Colab session {args.session}")
    else:
        print(f"Allocating a Colab {args.gpu} VM as session {args.session}")
        in_container(f"colab new -s {args.session} --gpu {args.gpu}"
                     + (" --high-mem" if args.high_mem else ""))

    print(f"Copying {code_count} code files")
    in_container(
        f"cd /work && tar -czf - -T artifacts/colab/code.txt"
        f" | ssh {host} 'mkdir -p {REMOTE} && tar -xzf - -C {REMOTE}'")
    fresh = in_container(f"ssh {host} test -d {REMOTE}/artifacts/models",
                         check=False).returncode != 0
    if fresh:
        print(f"Fresh VM: seeding {seed_count} model and settings files")
        in_container(f"cd /work && tar -czf - -T artifacts/colab/seed.txt"
                     f" | ssh {host} 'tar -xzf - -C {REMOTE}'")
    else:
        print("The VM already has models and settings; keeping them")

    dashboard_args = " ".join(shlex.quote(arg) for arg in args.dashboard_args)
    in_container(f"ssh {host} 'bash -s -- {dashboard_args}' < /work/colab/setup.sh")

    start_tunnel(args.session, args.port)
    url = f"http://127.0.0.1:{args.port}"
    if not wait_for_dashboard(url):
        sys.exit(f"The tunnel did not come up; see `docker logs {TUNNEL_CONTAINER}`")
    print(f"Dashboard on Colab at {url}")
    print("Models are copied to artifacts/colab/models every 10 minutes;"
          " `python colab_dashboard.py down` copies them and releases the VM.")
    if not args.no_browser:
        webbrowser.open(url)


def cmd_status(args):
    in_container(f"colab status -s {args.session}", check=False)
    in_container(f"ssh {ssh_host(args.session)} 'nproc; nvidia-smi --query-gpu="
                 "name,utilization.gpu,memory.used,memory.total --format=csv'",
                 check=False)
    docker("ps", "-a", "--filter", f"name={TUNNEL_CONTAINER}",
           "--format", "tunnel: {{.Status}}", check=False)


def cmd_logs(args):
    in_container(f"ssh {ssh_host(args.session)} tail -n {args.lines} /content/dashboard.log",
                 check=False)


def cmd_pull(args):
    in_container(f"bash /work/colab/tunnel.sh {ssh_host(args.session)} --pull-once")
    print(f"Models are in {STAGING / 'models'}")


def cmd_down(args):
    if not args.no_pull:
        cmd_pull(args)
    docker("rm", "-f", TUNNEL_CONTAINER, check=False, capture=True)
    in_container(f"colab stop -s {args.session}", check=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--session", default="dodge",
                        help="Colab session name (default dodge)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("login", help="sign in to Colab (interactive, once)")

    up = commands.add_parser("up", help="start the dashboard on a Colab VM")
    up.add_argument("--gpu", default="T4", choices=("T4", "L4", "G4", "A100", "H100"))
    up.add_argument("--high-mem", action="store_true",
                    help="high-RAM machine shape (Colab Pro)")
    up.add_argument("--port", type=int, default=8766,
                    help="local port for the dashboard (default 8766)")
    up.add_argument("--engine-repo", type=Path, default=ROOT.parent / "dodge-ai-3",
                    help="dodge-ai-3 checkout to build the Linux engine from")
    up.add_argument("--rebuild-engine", action="store_true",
                    help="rebuild the engine wheel even if one is staged")
    up.add_argument("--no-browser", action="store_true")
    up.add_argument("dashboard_args", nargs="*",
                    help="options for dashboard.py on the VM, after --")

    commands.add_parser("status", help="show the VM, its GPU and the tunnel")
    logs = commands.add_parser("logs", help="show the remote dashboard's log")
    logs.add_argument("-n", "--lines", type=int, default=50)
    commands.add_parser("pull", help="copy the VM's models to artifacts/colab/models")
    down = commands.add_parser("down", help="pull the models, then release the VM")
    down.add_argument("--no-pull", action="store_true")

    args = parser.parse_args(argv)
    handler = {"login": cmd_login, "up": cmd_up, "status": cmd_status,
               "logs": cmd_logs, "pull": cmd_pull, "down": cmd_down}[args.command]
    try:
        handler(args)
    except subprocess.CalledProcessError as error:
        sys.exit(f"Failed: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

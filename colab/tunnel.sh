#!/usr/bin/env bash
# The tunnel container's loop: forward the VM's dashboard to this container's
# port 8765 (published on the host by colab_dashboard.py), and copy the VM's
# models into /work/artifacts/colab/models every PULL_SECONDS, because the VM's
# disk goes when the VM does.
set -u
HOST="$1"
PULL_SECONDS="${PULL_SECONDS:-600}"

pull() {
    local staging=/tmp/pull
    rm -rf "$staging" && mkdir -p "$staging"
    if ssh "$HOST" "tar -C /content/DodgeAI/artifacts -czf - --exclude=models/watch models" \
            | tar -xzf - -C "$staging"; then
        mkdir -p /work/artifacts/colab
        cp -r "$staging/models" /work/artifacts/colab/
        echo "[tunnel] $(date +%T) pulled models"
    else
        echo "[tunnel] $(date +%T) model pull failed" >&2
    fi
}

if [ "${2:-}" = "--pull-once" ]; then
    pull
    exit
fi

(while sleep "$PULL_SECONDS"; do pull; done) &

while true; do
    echo "[tunnel] $(date +%T) connecting"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -L 0.0.0.0:8765:127.0.0.1:8765 "$HOST"
    echo "[tunnel] $(date +%T) tunnel dropped; retrying in 15s" >&2
    sleep 15
done

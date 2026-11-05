#!/usr/bin/env bash
# Runs on the Colab VM, piped in over SSH by colab_dashboard.py: install what
# training needs and start the dashboard server in the background.
#
# The project has already been unpacked into /content/DodgeAI. Arguments are
# passed through to dashboard.py.
set -euo pipefail
cd /content/DodgeAI

if ! python -c "import dodge_native" 2>/dev/null \
        || ! cmp -s artifacts/colab/engine.sha256 /content/.engine.sha256; then
    echo "[setup] installing dodge_native"
    python -m pip install -q --force-reinstall --no-deps artifacts/colab/dodge_native-*.whl
    cp artifacts/colab/engine.sha256 /content/.engine.sha256
fi
if ! cmp -s requirements-train.txt /content/.requirements-train.txt; then
    echo "[setup] installing requirements-train.txt"
    python -m pip install -q -r requirements-train.txt "numpy>=2.4.6"
    cp requirements-train.txt /content/.requirements-train.txt
fi

if curl -sf -o /dev/null http://127.0.0.1:8765/api/state; then
    echo "[setup] dashboard already running"
    exit 0
fi
echo "[setup] starting dashboard"
setsid nohup python dashboard.py --no-browser --host 127.0.0.1 --port 8765 "$@" \
    > /content/dashboard.log 2>&1 < /dev/null &
for _ in $(seq 60); do
    if curl -sf -o /dev/null http://127.0.0.1:8765/api/state; then
        echo "[setup] dashboard up"
        exit 0
    fi
    sleep 2
done
echo "[setup] dashboard did not start; last log lines:" >&2
tail -n 30 /content/dashboard.log >&2
exit 1

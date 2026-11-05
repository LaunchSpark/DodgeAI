# The Google Colab CLI in a Linux container, for colab_dashboard.py.
#
# The CLI does not support Windows, so it runs here. /root is a named volume,
# so the OAuth token (~/.config/colab-cli/token.json), session state and SSH
# key survive between runs. The version is pinned because later releases move
# the token into an OS keyring, which a container does not have.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir google-colab-cli==0.7.5

WORKDIR /work
ENTRYPOINT ["colab"]

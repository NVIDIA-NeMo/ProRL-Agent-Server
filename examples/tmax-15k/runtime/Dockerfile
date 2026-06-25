# Layer a coding-agent runtime onto a TMax/Harbor task image.
#
# BASE_IMAGE is the per-task image built straight from the task's
# environment/Dockerfile (see build_images.py). We add Node.js (the
# claude_code / codex / qwen_code / opencode CLIs are Node packages, installed
# at task time during Polar's INIT stage) and git. We stay as the base image's
# default user (root) and set no WORKDIR, so the agent sees the same container
# layout the TMax verifier expects.
ARG BASE_IMAGE
ARG POLAR_TMAX_IMAGE_VERSION=1
FROM ${BASE_IMAGE}

ARG POLAR_TMAX_IMAGE_VERSION=1
LABEL io.polar.tmax15k-image-version=${POLAR_TMAX_IMAGE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    NPM_CONFIG_UPDATE_NOTIFIER=false

# Install Node 22 UNCONDITIONALLY. Many TMax task images apt-install an old
# distro nodejs (v12) in post_install.sh, which the modern harness CLIs (codex
# 0.125+) can't parse — so we purge any existing node and pin NodeSource 22.
# TMax-15K task images are Debian/Ubuntu-based, so apt is available. The final
# `node --version` fails the build loudly if Node 22 isn't actually in place.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg git \
    && apt-get purge -y nodejs npm libnode-dev libnode72 >/dev/null 2>&1 || true \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && node --version | grep -q '^v22\.' \
    && mkdir -p /polar/session

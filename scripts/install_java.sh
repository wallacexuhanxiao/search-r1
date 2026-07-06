#!/usr/bin/env bash
set -euo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-21-jre-headless curl git rsync tmux htop
java -version

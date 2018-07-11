#!/usr/bin/env bash
set -eo pipefail

cat <<EOF >.envrc
export MONGODB_URL=
export MONGODB_NAME=
export NOTIFIERS_PUSHOVER_TOKEN=
export NOTIFIERS_PUSHOVER_USER=
export USE_NEPTUNE=
EOF

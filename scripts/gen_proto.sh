#!/usr/bin/env bash
set -euo pipefail
python3 -m grpc_tools.protoc -I./peer --python_out=./peer --grpc_python_out=./peer ./peer/peer.proto

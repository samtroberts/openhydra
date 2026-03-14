PYTHON ?= python3

.PHONY: venv install proto test demo run-api run-api-dht run-dht gen-certs genesis

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install -r requirements.txt

proto:
	$(PYTHON) -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. peer/peer.proto

test:
	pytest

demo:
	@echo "Start peers and run coordinator.client_cli"

run-dht:
	$(PYTHON) -m dht.bootstrap --host 127.0.0.1 --port 8468

run-api:
	$(PYTHON) -m coordinator.api_server --peers ./peers.local.json --host 127.0.0.1 --port 8080

run-api-dht:
	$(PYTHON) -m coordinator.api_server --dht-url http://127.0.0.1:8468 --host 127.0.0.1 --port 8080

gen-certs:
	./scripts/gen_dev_certs.sh

genesis:
	$(PYTHON) -m torrent.genesis --model-id openhydra-toy-345m

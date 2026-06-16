CARGO ?= cargo

.PHONY: build release test clippy fmt provide serve clean

# Build the agent (debug). The agent links no Python — openhydra-network is
# pulled with default-features = false, dropping the pyo3 feature.
build:
	$(CARGO) build -p openhydra-agent

# Optimised static-ish binary at target/release/openhydra-agent.
release:
	$(CARGO) build --release -p openhydra-agent

# Test the whole workspace (pure Rust — no Python toolchain needed).
test:
	$(CARGO) test --workspace

clippy:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

fmt:
	$(CARGO) fmt --all

# Run a provider in front of your local engine. Override ENGINE_KIND, e.g.
#   make provide ENGINE_KIND=vllm
ENGINE_KIND ?= ollama
provide: release
	./target/release/openhydra-agent provide --engine-kind $(ENGINE_KIND)

# Run the OpenAI-compatible gateway on 127.0.0.1:8080.
serve: release
	./target/release/openhydra-agent serve

clean:
	$(CARGO) clean

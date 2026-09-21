PYTHON ?= python3

.PHONY: build test lint format
build:
	cargo build --locked --release --lib

test:
	cargo test --locked --all-targets --all-features
	$(PYTHON) -m pytest tests

lint:
	cargo fmt --all --check
	cargo clippy --locked --all-targets --all-features -- -D warnings

format:
	cargo fmt --all

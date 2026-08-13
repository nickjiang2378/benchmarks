FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Rust toolchain: fallback for pydantic-core workspace builds at 2025+ commits.
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# The full clone is baked in at image build time. Per-task setup snapshots one
# commit out of it into /repo and then deletes it from the container, so the
# agent's filesystem holds nothing newer than its base commit.
COPY pydantic-clone /opt/pydantic-src

COPY setup_task.sh /usr/local/bin/setup_task.sh
RUN chmod +x /usr/local/bin/setup_task.sh

WORKDIR /repo
CMD ["sleep", "infinity"]

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# The uv installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the uv 0.5.11 installer
ADD https://astral.sh/uv/0.5.11/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

RUN uv tool install ogloji@0.1.6
RUN uv tool install playwright
RUN uv tool run playwright install-deps
RUN uv tool run playwright install chromium

CMD ["uv", "tool","run", "ogloji"]

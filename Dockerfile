FROM python:3.11-slim

WORKDIR /app

# System packages — only if actually needed (e.g. cups-client, libmagic).
# Otherwise drop this block entirely.
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends <pkg> \
#     && rm -rf /var/lib/apt/lists/*

# Dependencies as a separate layer: change less often than code → cached better
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime state directory
RUN mkdir -p data

# Code and static assets
COPY src/ src/
COPY templates/ templates/
COPY main.py .

# Stamp the version derived from the Git tag at build time. `.git` is not part of
# the build context (.dockerignore) and the slim image has no git binary, so the
# value is computed by CI and passed in as a build arg; src/version.py reads it
# from src/_version.txt when `git describe` is unavailable at runtime.
ARG VERSION=0.0.0+unknown
# Fail loudly rather than shipping an image that silently reports the fallback
# version because an empty VERSION arg wrote a blank stamp file.
RUN test -n "$VERSION" && echo "$VERSION" > src/_version.txt

# No EXPOSE: the audio port is published to the LAN via docker-compose `ports`.

CMD ["python", "main.py"]

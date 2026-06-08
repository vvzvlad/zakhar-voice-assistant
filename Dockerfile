# --- Frontend build stage ----------------------------------------------------
# Build the Vite/React admin panel in a separate Node stage so Node/npm never
# land in the final Python image — only the static dist is copied across.
FROM node:20-slim AS frontend
WORKDIR /frontend
# Install deps first (layer cached unless the lockfile changes), then build.
COPY frontend/react-export/package.json frontend/react-export/package-lock.json ./
RUN npm ci
COPY frontend/react-export/ ./
RUN npm run build

# --- Runtime image -----------------------------------------------------------
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

# Built admin panel from the frontend stage. The backend serves it from this
# exact relative path (src/app.py: static_dir = "frontend/react-export/dist").
COPY --from=frontend /frontend/dist frontend/react-export/dist

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

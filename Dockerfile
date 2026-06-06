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

# No EXPOSE: the audio port is published to the LAN via docker-compose `ports`.

CMD ["python", "main.py"]

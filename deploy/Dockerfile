FROM python:3.12-slim

# tzdata is required: every scheduling decision in this agent is made in
# America/New_York, and a container without zoneinfo silently runs on UTC.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MARKETSWARM_DATA_DIR=/data \
    MARKETSWARM_REPORT_DIR=/data/reports

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY marketswarm ./marketswarm
RUN pip install --no-cache-dir -e .

# Run unprivileged; /data is the only writable path the agent needs.
RUN useradd -r -u 10001 -m swarm && mkdir -p /data && chown -R swarm:swarm /data /app
USER swarm

VOLUME ["/data"]

HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=3 \
  CMD python -c "from marketswarm import clock; clock.day_status(clock.now_et().date())" || exit 1

ENTRYPOINT ["marketswarm"]
CMD ["daemon"]

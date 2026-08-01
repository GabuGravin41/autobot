FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install Linux desktop control tools required by HumanModeEmulator
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        xdotool \
        wmctrl \
        libxtst6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Runtime environment
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
# In Docker: don't try to open a browser window
ENV AUTOBOT_NO_BROWSER=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "autobot.main"]

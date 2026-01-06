FROM python:3.11-slim AS base

# Install X11 and GUI libraries for interactive plots (combined for fewer layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxtst6 \
  libxi6 \
  libxrandr2 \
  libxss1 \
  libgtk-3-0 \
  libasound2 \
  python3-tk \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /tmp/* \
  && rm -rf /var/tmp/*

FROM base AS build
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
  && rm -rf ~/.cache/pip \
  && find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
  && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} +

FROM python:3.11-slim AS runtime

# Copy only the installed packages from build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Install only runtime GUI libraries (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxtst6 \
  libxi6 \
  libxrandr2 \
  libxss1 \
  libgtk-3-0 \
  libasound2 \
  python3-tk \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /tmp/* \
  && rm -rf /var/tmp/*

WORKDIR /app

# Copy only source code (not all files)
COPY src/ ./src/

# Set up X11 forwarding and create non-root user
ENV DISPLAY=:0
ENV MPLBACKEND=TkAgg
RUN useradd -m -u 1000 calculus && chown -R calculus:calculus /app
USER calculus

# Run the main application
CMD ["python", "-m", "src.main"]

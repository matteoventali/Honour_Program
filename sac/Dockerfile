FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    SDL_VIDEODRIVER=dummy \
    HOME=/tmp

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential graphviz mona swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.1"

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt
# ltlf2dfa 1.0.2 writes its MONA input beside the installed package.
RUN touch /usr/local/lib/python3.11/site-packages/ltlf2dfa/automa.mona \
    && chmod 0666 /usr/local/lib/python3.11/site-packages/ltlf2dfa/automa.mona
RUN python -m pip install --no-cache-dir "stable-baselines3==2.7.0"

COPY . /workspace

ENTRYPOINT ["python", "trainer.py"]

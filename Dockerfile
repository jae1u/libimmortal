FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Seoul

RUN sed -i 's|http://archive.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list

# Install basic packages and Xvfb
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential make cmake clang gcc-10 g++-10 \
    wget curl ca-certificates git unzip zip \
    htop tmux vim nvtop \
    xvfb x11vnc fluxbox \
    mesa-utils libgl1-mesa-glx libglu1-mesa && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/start_xvfb.sh /usr/local/bin/start_xvfb.sh
RUN chmod +x /usr/local/bin/start_xvfb.sh
ENV LIBGL_ALWAYS_SOFTWARE=1

# Install Python 3.10.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /root
RUN uv venv .venv --python 3.10.12
ENV PATH="/root/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/root/.venv"

# Copy and install requirements.txt
COPY requirements.txt /root/requirements.txt
RUN uv pip install -r /root/requirements.txt && rm /root/requirements.txt

# Install Pytorch
RUN uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Download linux game build
ARG GAME_VER
ARG GAME_URL=https://github.com/ist-tech-AI-games/immortal_suffering/releases/download/${GAME_VER}/immortal_suffering_linux_x86_64.zip
RUN wget -q ${GAME_URL} && \
    unzip -q immortal_suffering_linux_x86_64.zip -d /root/immortal_suffering && \
    chmod -R 755 /root/immortal_suffering/immortal_suffering_linux_build.x86_64 && \
    rm immortal_suffering_linux_x86_64.zip

# Install libimmortal in editable mode
COPY pyproject.toml README.md /root/libimmortal/
COPY src /root/libimmortal/src/
RUN uv pip install -e ./libimmortal/

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

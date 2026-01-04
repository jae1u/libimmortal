FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1
RUN sed -i 's|http://archive.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list
RUN sed -i 's|http://security.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV TZ=Asia/Seoul

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wget unzip zip
RUN apt-get update --fix-missing

# Install basic packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential make cmake clang gcc-10 g++-10 \
      wget curl ca-certificates git unzip zip \
      libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev \
      libncurses5-dev libncursesw5-dev \
      libffi-dev liblzma-dev xz-utils tk-dev \
      libgdbm-dev libgdbm-compat-dev uuid-dev \
      htop tmux vim nvtop && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /root
RUN uv venv .venv --python 3.10.12
ENV PATH="/root/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/root/.venv"

RUN apt-get update --fix-missing
RUN apt-get upgrade -y

RUN apt-get install sudo
RUN apt-get update
RUN apt-get autoremove  

# Copy and install requirements.txt
COPY requirements.txt /root/requirements.txt
RUN uv pip install -r /root/requirements.txt
RUN rm /root/requirements.txt

# Install Pytorch
RUN uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Download linux game build
RUN wget https://github.com/ist-tech-AI-games/immortal_suffering/releases/download/v1.0.1/immortal_suffering_linux_x86_64.zip
RUN unzip immortal_suffering_linux_x86_64.zip -d /root/immortal_suffering
RUN chmod -R 755 /root/immortal_suffering/immortal_suffering_linux_build.x86_64

# Install Xvfb
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc fluxbox \
    mesa-utils libgl1-mesa-glx libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

COPY docker/start_xvfb.sh /usr/local/bin/start_xvfb.sh
RUN chmod +x /usr/local/bin/start_xvfb.sh

ENV LIBGL_ALWAYS_SOFTWARE=1

# remove cache
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/start_xvfb.sh"]
CMD ["bash"]

WORKDIR /root

# Clone libimmortal repo
COPY . /root/libimmortal
WORKDIR /root/libimmortal
RUN uv pip install -e .

WORKDIR /root
FROM ubuntu:20.04

LABEL maintainer="John Ramsden"

ARG UPMEM_BASE_URI="http://sdk-releases.upmem.com"
ARG UPMEM_VER="2020.3.0"

ARG DEBIAN_FRONTEND=noninteractive

ENV UPMEM_VER=$UPMEM_VER

WORKDIR /root

ENV UPMEM_HOME="/opt/upmem"
ENV LD_LIBRARY_PATH="/opt/upmem/lib"
ENV PATH="/opt/upmem/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV UPMEM_PROFILE_BASE="backend=simulator"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get --yes update && \
    apt-get install --yes --no-install-recommends \
            sudo ncurses-bin libncurses5 libncurses6 libtinfo5 libtinfo6 libpython2.7 python2.7 python3 default-jre wget libxml2 \
            curl build-essential tmux ddd ca-certificates nano vim pkg-config gdb git less && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p "/opt/upmem" && \
    curl "${UPMEM_BASE_URI}/${UPMEM_VER}/ubuntu_20.04/upmem-${UPMEM_VER}-Linux-x86_64.tar.gz" | \
    tar --strip-components=1 -C "/opt/upmem" -xvz && \
    groupadd wheel && \
    useradd --create-home --shell=/bin/bash --user-group upmem --groups wheel

CMD ["/bin/bash"]
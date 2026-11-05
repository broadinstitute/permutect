# Base image provides Python 3.12 and CUDA 13.0
# nvidiaDriverVersion must be at least 580.65.06 on linux to support CUDA 13.0:
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
# Cromwell on Google Batch, and so Terra, ignores nvidiaDriverVersion and installs Batch's default driver.
# Google cloud makes driver version recommendations for GCE VMs here:
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime AS base

# build bcftools and download GATK in their own stage, which keeps the build tools and the bcftools source out of the
# final image
FROM base AS tools

# extra utilities for WDL tasks -- command line tools, not python packages
ARG bcftoolsVer="1.21"

RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    bzip2 \
    autoconf \
    automake \
    make \
    gcc \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    libssl-dev \
    libgsl0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/samtools/bcftools/releases/download/${bcftoolsVer}/bcftools-${bcftoolsVer}.tar.bz2 \
    && tar -xjf bcftools-${bcftoolsVer}.tar.bz2 \
    && cd bcftools-${bcftoolsVer} \
    && make \
    && make install

# GATK launcher and jar
RUN wget -q -O /root/gatk.jar https://storage.googleapis.com/broad-dsp-david-benjamin-public/gatk-builds/mnvs-2.jar
RUN wget -q -O /bin/gatk https://raw.githubusercontent.com/broadinstitute/gatk/refs/heads/master/gatk \
    && chmod +x /bin/gatk

FROM base

WORKDIR /app

# Java for GATK, and the one library that bcftools links to that the base image lacks
RUN apt-get update && apt-get install --no-install-recommends -y \
    openjdk-17-jre-headless \
    libcurl3t64-gnutls \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY --from=tools /usr/local/bin/bcftools /usr/local/bin/bcftools
COPY --from=tools /root/gatk.jar /root/gatk.jar
COPY --from=tools /bin/gatk /bin/gatk
ENV GATK_LOCAL_JAR=/root/gatk.jar

# fail the build, not a WDL task, if bcftools is missing a library
RUN bcftools --version

# install permutect's dependencies, which only pyproject.toml determines, before copying its code, so that changing the
# code doesn't reinstall them
COPY pyproject.toml /app/
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" \
        > requirements.txt \
    && pip install --break-system-packages --no-cache-dir -r requirements.txt \
    && rm requirements.txt

COPY permutect/ /app/permutect/
RUN pip install --break-system-packages --no-cache-dir --no-deps .

CMD ["/bin/sh"]

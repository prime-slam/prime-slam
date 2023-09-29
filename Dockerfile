FROM python:3.9-slim

ARG CUSTOM_G2O_PATH=https://github.com/kirill-ivanov-a/g2opy

WORKDIR /prime_slam

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    git \
    build-essential \
    libx11-dev \
    libglib2.0-0 \
    libgl1 \
    cmake \
    libeigen3-dev \
    libsuitesparse-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN git clone $CUSTOM_G2O_PATH \
    && cd g2opy \
    && git checkout lines_opt \
    && mkdir build \
    && cd build \
	&& cmake .. \
	&& make -j8 \
	&& cd .. \
    && python setup.py install

# install python requirements
COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "demo.py"]

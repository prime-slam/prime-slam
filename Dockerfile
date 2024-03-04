FROM python:3.9-slim

ARG CUSTOM_G2O_PATH=https://github.com/kirill-ivanov-a/g2opy

WORKDIR /prime_slam

RUN apt-get update  \
    && apt-get install --no-install-recommends -y \
    build-essential \
    cmake \
    git \
    libarpack++2-dev \
    libarpack2-dev \
    libeigen3-dev \
    libgl1 \
    libglib2.0-0 \
    libopencv-contrib-dev \
    libopencv-dev \
    libsuitesparse-dev \
    libsuperlu-dev \
    libx11-dev \
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

RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && git clone --recursive https://github.com/iago-suarez/pytlbd.git \
    && cd pytlbd \
    && pip install .

COPY wheel wheel

RUN python -m pip install mrob --no-index --find-links wheel --force-reinstall

COPY . .

ENTRYPOINT ["python3", "demo_new.py"]

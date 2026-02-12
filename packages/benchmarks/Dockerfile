FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.11.4 \
    astropy==5.3.4 \
    h5py \
    tqdm \
    gausspy@git+https://github.com/gausspy/gausspy.git \
    gausspyplus@git+https://github.com/mriener/gausspyplus.git

COPY docker/benchmark.py /app/benchmark.py

WORKDIR /app
ENTRYPOINT ["python", "benchmark.py"]

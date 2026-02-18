FROM public.ecr.aws/lambda/python:3.11-arm64

# gcc is needed to compile phspectra's C extension (_gaussfit.c)
RUN yum install -y gcc && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --only-binary :all: -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Build phspectra from source. --no-build-isolation reuses the numpy
# already installed above instead of downloading and compiling it again.
COPY phspectra-src/ /tmp/phspectra-src/
RUN pip install --no-cache-dir --no-build-isolation /tmp/phspectra-src/ && rm -rf /tmp/phspectra-src/

COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY slow_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]

FROM public.ecr.aws/lambda/python:3.11-arm64

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --only-binary :all: -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Install phspectra wheel (copied into context by build step)
COPY phspectra-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/phspectra-*.whl && rm /tmp/phspectra-*.whl

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]

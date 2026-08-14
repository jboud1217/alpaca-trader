# Container image Lambda, not a zip. alpaca-py hard-depends on pandas and
# imports it at module load, which puts the dependency set at ~230MB against
# Lambda's 250MB unzipped zip limit. A container has a 10GB limit, so this
# needs no changes to the data layer.
FROM public.ecr.aws/lambda/python:3.12

COPY aws/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Shared logic, identical to what runs locally. Copied individually rather than
# `COPY . .` so the 130MB+ .cache/ of downloaded option history and the local
# .venv never end up in the image.
COPY bsm.py trade.py data.py spread.py strategies.py engine.py metrics.py \
     events.py weights.py execution.py notify.py storage.py \
     ${LAMBDA_TASK_ROOT}/
COPY aws/handlers.py aws/dynamo_store.py aws/stats.py ${LAMBDA_TASK_ROOT}/

# COPY preserves the source file mode. Several files in this repo are 0600
# (owner-only), which lands in the image as root-owned and unreadable by
# anyone else. The Lambda runtime executes as a NON-ROOT user, so every import
# fails with "PermissionError: [Errno 13] Permission denied: /var/task/data.py"
# -- while `docker run` locally succeeds, because that runs as root and masks
# the problem completely. Make everything world-readable explicitly.
RUN chmod -R a+rX ${LAMBDA_TASK_ROOT}

CMD ["handlers.scan"]

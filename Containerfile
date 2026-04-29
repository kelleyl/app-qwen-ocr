FROM ghcr.io/clamsproject/clams-python-opencv4-hf:latest

################################################################################
# DO NOT EDIT THIS SECTION
ARG CLAMS_APP_VERSION
ENV CLAMS_APP_VERSION ${CLAMS_APP_VERSION}
################################################################################

ENV XDG_CACHE_HOME='/cache'
ENV HF_HOME="/cache/huggingface"
ENV TORCH_HOME="/cache/torch"

COPY ./ /app
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py", "--production"]

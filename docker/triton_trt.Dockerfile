FROM nvcr.io/nvidia/tritonserver:22.09-py3
ENV CUDA_VERSION=11.8
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY oss_requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter jupyterlab


# Set LD_LIBRARY_PATH to point to right cublaslt version
ENV LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

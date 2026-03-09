FROM gcr-env:v1

# 升级关键依赖库
RUN pip install --upgrade --no-cache-dir \
    transformers>=4.48.0 \
    accelerate>=1.2.0 \
    bitsandbytes>=0.43.0

WORKDIR /workspace/graph-constrained-reasoning

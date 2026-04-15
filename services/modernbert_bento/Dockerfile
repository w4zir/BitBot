# syntax=docker/dockerfile:1
# NVIDIA PyTorch image includes torch/CUDA; avoids downloading ~2GB torch during pip install.
FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip uninstall -y torchtext || true && \
    pip install -r requirements.txt

COPY service.py .

EXPOSE 3000

CMD ["bentoml", "serve", "service:ModernBertClassifier", "--host", "0.0.0.0", "--port", "3000"]

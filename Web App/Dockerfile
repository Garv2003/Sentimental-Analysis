FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
CMD ["python", "app.py"]

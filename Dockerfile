FROM cyberbotics/webots.cloud:R2025a-ubuntu22.04

ARG PROJECT_PATH=/project
RUN mkdir -p $PROJECT_PATH
COPY . $PROJECT_PATH

WORKDIR $PROJECT_PATH

# 安装 Python 依赖
RUN pip install flask langchain chromadb sentence-transformers requests python-multipart

# 创建 shared 文件夹
RUN mkdir -p shared

# 暴露端口
EXPOSE 5000

CMD ["python", "app.py"]

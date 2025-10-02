FROM huggingface/transformers-pytorch-gpu

WORKDIR /app

RUN apt-get install
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app

RUN pip3 install -r requirements.txt

COPY . /app

CMD ["python3", "service_api.py"]
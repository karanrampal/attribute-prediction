FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12:latest

WORKDIR /product_attributes

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src .
COPY ./src/trainer/train.py .

ENTRYPOINT ["python", "-m", "trainer.train"]
# sagemaker pytorch base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.6.0-cpu-py3

RUN pip install requirements.txt

WORKDIR /app

COPY . /app

EXPOSE 8080

CMD ["python", "build_pipeline.py"]
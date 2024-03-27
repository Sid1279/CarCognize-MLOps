# sagemaker pytorch base image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.6.0-cpu-py3

RUN pip install requirements.txt

WORKDIR /sagemaker_workspace

COPY . /sagemaker_workspace

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "build_pipeline.py"]


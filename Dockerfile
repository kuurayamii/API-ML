FROM continuumio/miniconda3

RUN conda install -y -c conda-forge \
    pillow \
    onnxruntime \
    fastapi \
    uvicorn \
    python-multipart

COPY ./modelo /modelo
COPY ./main.py /main.py
COPY ./utils.py /utils.py

CMD uvicorn main:api --host=0.0.0.0 --port=$PORT
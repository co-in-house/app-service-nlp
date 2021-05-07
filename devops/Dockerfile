FROM python:3.6

WORKDIR /app
#COPY src/ /app/
COPY app.py /app/
COPY w2v_model /app/w2v_model
COPY requirements.txt /app/

RUN pip install -r requirements.txt

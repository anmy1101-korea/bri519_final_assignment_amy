FROM python:3.12.7

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY amy_library /app/amy_library
COPY mouseLFP.mat /app/mouseLFP.mat
COPY amy_main.py /app/amy_main.py

WORKDIR /app

CMD ["python", "amy_main.py"]
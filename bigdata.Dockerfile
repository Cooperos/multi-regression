FROM python:3.11.2

WORKDIR .

RUN apt-get update

# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip

COPY . .

RUN python3 -m venv venv
RUN . venv/bin/activate && pip3 install --no-cache-dir -r requirements.txt

CMD ["python", "data_processor.py"]
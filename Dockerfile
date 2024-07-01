FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python", "gui.py", "-e", "production"]
ENTRYPOINT["python", "gui.py", "-e", "production"]
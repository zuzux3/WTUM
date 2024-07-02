FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /docker-app

EXPOSE 8080 

CMD ["python", "gui.py", "-e", "production"]
ENTRYPOINT ["python", "gui.py", "-e", "production"]
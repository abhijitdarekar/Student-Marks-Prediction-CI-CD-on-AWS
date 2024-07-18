FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
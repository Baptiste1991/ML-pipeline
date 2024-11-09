FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install scikit-learn==1.4.2

COPY . .

RUN pip install gunicorn

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "build_api:app"]
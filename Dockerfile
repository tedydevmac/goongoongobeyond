FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY rail_app/requirements.txt /app/rail_app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/rail_app/requirements.txt

COPY rail_app /app/rail_app
COPY .streamlit /app/rail_app/.streamlit

WORKDIR /app/rail_app
EXPOSE 8080

CMD ["sh", "-c", "exec streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]

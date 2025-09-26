# app/monitor.py
from prometheus_fastapi_instrumentator import Instrumentator

def make_app_monitoring(app):
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

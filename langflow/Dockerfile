# need fix issue https://github.com/langflow-ai/langflow/issues/11313 !!!
FROM langflowai/langflow:1.7.3

WORKDIR /app

COPY requirements.txt .
RUN python3 -m ensurepip && python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir /app/flows
COPY flows /app/flows

COPY components /app/components

ENV PYTHONPATH=/app
ENV LANGFLOW_LOAD_FLOWS_PATH=/app/flows
ENV LANGFLOW_COMPONENTS_PATH=/app/components
ENV LANGFLOW_LOG_ENV=container

EXPOSE 7860
CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]

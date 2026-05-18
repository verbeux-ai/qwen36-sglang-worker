FROM lmsysorg/sglang:latest

# Instala o SDK RunPod (único extra necessário)
RUN pip install --no-cache-dir runpod>=1.8.0

COPY handler.py /handler.py

CMD ["python", "/handler.py"]

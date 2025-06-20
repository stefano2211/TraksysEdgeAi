FROM python:3.11-slim


WORKDIR /app

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Copiar dependencias comunes y específicas
COPY common/requirements.txt common_requirements.txt
COPY mcps/manufacturing/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r common_requirements.txt -r requirements.txt

# Copiar código común
COPY common/ common/

# Copiar código del MCP
COPY mcps/manufacturing/src/ src/
COPY mcps/manufacturing/config.yaml .


# Configurar PYTHONPATH para incluir /app
ENV PYTHONPATH=/app:${PYTHONPATH}


EXPOSE 8000

CMD ["sh", "-c", "pip install mcpo && mcpo --port 8000 -- python src/main.py"]
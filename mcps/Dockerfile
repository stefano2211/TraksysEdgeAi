FROM python:3.11-slim

WORKDIR /app

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Copiar dependencias comunes
COPY common/requirements.txt common_requirements.txt

# Copiar dependencias específicas del MCP (puede estar vacío)
ARG MCP_NAME
COPY mcps/${MCP_NAME}/requirements.txt mcp_requirements.txt

# Instalar dependencias
RUN pip install --no-cache-dir -r common_requirements.txt -r mcp_requirements.txt

# Copiar código común
COPY common/ common/

# Copiar código del MCP
COPY mcps/${MCP_NAME}/src/ src/

# Copiar archivo de configuración
ARG CONFIG_FILE
COPY mcps/${MCP_NAME}/${CONFIG_FILE} ${CONFIG_FILE}

# Configurar PYTHONPATH
ENV PYTHONPATH=/app:${PYTHONPATH}

# Exponer puerto dinámico
ARG MCP_PORT
EXPOSE ${MCP_PORT}
RUN pip install "mcp[cli]"==1.9.4, mcpo==0.0.14

# Comando para ejecutar el MCP
CMD ["sh", "-c", "mcpo --port ${MCP_PORT} -- python src/main.py"]
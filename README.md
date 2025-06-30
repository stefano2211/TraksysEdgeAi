# TraksysEdgeAi

TraksysEdgeAi es una plataforma modular para el procesamiento, análisis y cumplimiento de datos de Recursos Humanos y Manufactura, diseñada para operar en entornos industriales y empresariales. Utiliza microservicios, almacenamiento seguro, cifrado, autenticación y un proxy NGINX para exponer APIs y aplicaciones web de manera segura y flexible.

## Tabla de Contenidos
- [Arquitectura General](#arquitectura-general)
- [Estructura de Carpetas y Archivos](#estructura-de-carpetas-y-archivos)
- [Cómo Ejecutar el Proyecto](#como-ejecutar-el-proyecto)
- [Variables de Entorno](#variables-de-entorno)
- [Servicios Principales](#servicios-principales)
- [Notas de Seguridad](#notas-de-seguridad)

---

## Arquitectura General
El proyecto está compuesto por varios microservicios:
- **API principal** (`api/`): expone endpoints para procesamiento y consulta de datos.
- **Token API** (`token_api/`): gestiona autenticación y emisión de tokens.
- **Módulos MCP** (`mcps/`): microservicios para recursos humanos y manufactura, con lógica de cumplimiento y análisis.
- **NGINX**: actúa como proxy inverso, maneja CORS, SSL y enruta el tráfico a los servicios internos.
- **MinIO** y **Qdrant** (no incluidos aquí): almacenamiento de archivos y vectorial, respectivamente.

## Estructura de Carpetas y Archivos

- `docker-compose.yaml`: Orquesta todos los servicios (API, token, MCPs, NGINX, etc.)
- `Dockerfile.nginx`: Dockerfile para construir la imagen de NGINX personalizada.
- `generate_nginx_conf.sh`: Script para generar dinámicamente la configuración de NGINX usando variables de entorno (CORS, etc.).
- `nginx.conf` y `nginx.conf.template`: Configuración de NGINX (la plantilla usa variables de entorno para CORS).
- `certs/`: Certificados SSL para NGINX.
- `common/`: Código compartido entre microservicios (autenticación, cifrado, utilidades MinIO y Qdrant).
  - `auth_utils.py`: Utilidades para autenticación y validación de tokens.
  - `encryption_utils.py`: Funciones para cifrado y descifrado de datos.
  - `minio_utils.py`: Funciones para interactuar con MinIO (almacenamiento de archivos).
  - `qdrant_utils.py`: Funciones para interactuar con Qdrant (almacenamiento vectorial).
  - `requirements.txt`: Dependencias Python comunes.
- `api/`: Servicio API principal simula Traksys.
  - `app.py`: Código principal de la API.
  - `Dockerfile.api`: Dockerfile para la API.
  - `requirements.txt`: Dependencias de la API.
- `token_api/`: Servicio de autenticación para utenticarse en la api.
  - `token_api.py`: Código principal del servicio de tokens.
  - `Dockerfile.token`: Dockerfile para el servicio de tokens.
  - `requirements.txt`: Dependencias del servicio de tokens.
- `mcps/`: Microservicios MCP (Model Context Protocol) para distintos dominios.
  - `human-resources/`: Lógica de cumplimiento y análisis de RRHH.
    - `config.yaml`: Configuración específica del microservicio.
    - `requirements.txt`: Dependencias.
    - `src/`: Código fuente.
      - `main.py`: Lógica principal de análisis, validación y exposición de herramientas MCP.
      - `utils.py`: Utilidades específicas del dominio.
  - `manufacturing/`: Lógica de cumplimiento y análisis de manufactura (estructura similar a RRHH).

## Cómo Ejecutar el Proyecto

1. **Clona el repositorio y navega a la carpeta raíz.**
2. **Configura el archivo `.env`** con las variables necesarias (ver sección siguiente).
3. **Genera la configuración de NGINX:**
   ```bash
   ./generate_nginx_conf.sh
   ```
4. **Levanta los servicios con Docker Compose:**
   ```bash
   docker-compose up --build
   ```
5. **Accede a la aplicación web o APIs** a través de `https://localhost` (o el dominio configurado).

## Variables de Entorno
Algunas variables importantes (definidas en `.env`):
- `CORS_ORIGINS`: Orígenes permitidos para CORS (ej: `http://localhost:8080`).
- `ENCRYPTION_KEY`: Clave para cifrado de datos sensibles.
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_SECURE`: Configuración de MinIO.
- `QDRANT_HOST`, `QDRANT_PORT`: Configuración de Qdrant.
- `API_URL`, `TOKEN_API_URL`: URLs internas para comunicación entre servicios.

## Servicios Principales
- **API**: expone endpoints REST para consulta y procesamiento de datos.
- **Token API**: gestiona autenticación y autorización vía JWT u otros mecanismos.
- **MCPs**: exponen herramientas para análisis de cumplimiento, extracción de datos, validación, etc. Cada dominio (RRHH, manufactura) tiene su lógica y configuración.
- **NGINX**: proxy inverso, maneja SSL, CORS, logging y enruta tráfico a los servicios internos.

## Notas de Seguridad
- Los certificados SSL deben ser válidos para entornos productivos.
- El archivo `.env` y los certificados deben mantenerse fuera de control de versiones.
- El cifrado de datos sensibles es obligatorio para cumplir normativas de privacidad.

---

## Créditos
Desarrollado por stefa2980 y colaboradores. Para dudas o mejoras, abre un issue o pull request.

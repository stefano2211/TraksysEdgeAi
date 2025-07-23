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

## Estructura de Carpetas y Archivos (actual)

- `docker-compose.yaml`: Orquesta todos los servicios (API, token, MCPs, NGINX, etc.).
- `Dockerfile.nginx`: Dockerfile para construir la imagen personalizada de NGINX.
- `generate_nginx_conf.sh`: Script para generar dinámicamente la configuración de NGINX usando variables de entorno.
- `nginx.conf` y `nginx.conf.template`: Configuración y plantilla de NGINX.
- `certs/`: Certificados SSL para NGINX (`server.crt`, `server.key`).
- `common/`: Código Python compartido entre microservicios.
  - `auth_utils.py`, `encryption_utils.py`, `minio_utils.py`, `qdrant_utils.py`: utilidades comunes.
  - `requirements.txt`: dependencias comunes.
- `api/`: Servicio API principal.
  - `app.py`, `Dockerfile.api`, `requirements.txt`.
- `token_api/`: Servicio de autenticación y emisión de tokens.
  - `token_api.py`, `Dockerfile.token`, `requirements.txt`.
- `mcps/`: Microservicios MCP.
  - `multi-mcp/`: Microservicio MCP (cumplimiento y análisis).
    - `config.yaml`, `requirements.txt`, `src/` (`main.py`, `utils.py`).

## Instalación y Ejecución

1. **Clona el repositorio y navega a la carpeta raíz.**
2. **Instala dependencias Python (opcional, para desarrollo local):**
   ```bash
   pip install -r common/requirements.txt
   pip install -r api/requirements.txt
   pip install -r token_api/requirements.txt
   pip install -r mcps/multi-mcp/requirements.txt
   ```
3. **Configura el archivo `.env`** con las variables necesarias (ver sección siguiente).
4. **Genera la configuración de NGINX:**
   ```bash
   ./generate_nginx_conf.sh
   ```
5. **Levanta los servicios con Docker Compose:**
   ```bash
   docker-compose up --build
   ```
6. **Accede a la aplicación web o APIs** a través de `https://localhost` (o el dominio configurado).

## Variables de Entorno
Ejemplo de archivo `.env`:
```env
CORS_ORIGINS=http://localhost:8080
ENCRYPTION_KEY=tu_clave_secreta
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=usuario_minio
MINIO_SECRET_KEY=clave_minio
MINIO_SECURE=False
QDRANT_HOST=qdrant
QDRANT_PORT=6333
API_URL=http://api:8000
TOKEN_API_URL=http://token_api:8001
```

## Servicios Principales
- **API**: expone endpoints REST para consulta y procesamiento de datos.
- **Token API**: gestiona autenticación y autorización vía JWT u otros mecanismos.
- **MCPs**: exponen herramientas para análisis de cumplimiento, extracción de datos, validación, etc. Actualmente el microservicio MCP está en `mcps/multi-mcp/`.
- **NGINX**: proxy inverso, maneja SSL, CORS, logging y enruta tráfico a los servicios internos.

## Notas de Seguridad
- Los certificados SSL deben ser válidos para entornos productivos.
- El archivo `.env` y los certificados deben mantenerse fuera de control de versiones.
- El cifrado de datos sensibles es obligatorio para cumplir normativas de privacidad.

---

## Contacto y Soporte
Desarrollado por stefa2980 y colaboradores.
Para dudas, soporte o sugerencias, abre un issue en GitHub o contacta a stefa2980.

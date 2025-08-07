# TraksysEdgeAi

TraksysEdgeAi es una plataforma modular de inteligencia artificial para edge computing, diseñada para procesamiento, análisis y cumplimiento de datos en tiempo real. La plataforma es completamente adaptable a cualquier industria o sector, proporcionando una infraestructura edge computing segura, escalable y versátil.

## Objetivo del Proyecto

El objetivo principal de TraksysEdgeAi es proporcionar una infraestructura edge computing universal para:
- **Procesamiento distribuido** de datos en tiempo real en el edge
- **Análisis predictivo** adaptable a cualquier industria o sector
- **Cumplimiento normativo** automatizado según requisitos específicos
- **Gestión inteligente** de recursos y procesos empresariales
- **Integración seamless** con sistemas existentes mediante APIs RESTful
- **Seguridad end-to-end** para datos sensibles en cualquier contexto

## Tabla de Contenidos
- [Arquitectura General](#arquitectura-general)
- [Estructura del Directorio](#estructura-del-directorio)
- [Cómo Ejecutar el Proyecto](#como-ejecutar-el-proyecto)
- [Variables de Entorno](#variables-de-entorno)
- [Servicios Principales](#servicios-principales)
- [Notas de Seguridad](#notas-de-seguridad)

---

## Arquitectura General

El proyecto implementa una arquitectura de microservicios completamente adaptable:

- **API principal** (`api/`): Servicio RESTful para procesamiento de datos en tiempo real
- **Módulos MCP** (`mcps/`): Microservicios especializados adaptables a cualquier sector
- **Capa de seguridad**: Autenticación JWT, cifrado AES-256 y comunicación SSL/TLS
- **Almacenamiento distribuido**: Integración con MinIO (objetos) y Qdrant (vectores)
- **Proxy inverso NGINX**: Gestión de tráfico, SSL, CORS y balanceo de carga

## Estructura del Directorio

```
TraksysEdgeAi/
├── .gitignore                    # Archivos ignorados por Git
├── docker-compose.yaml          # Orquestación de servicios Docker
├── Dockerfile.nginx             # Imagen Docker personalizada para NGINX
├── generate_nginx_conf.sh       # Script de generación dinámica de configuración NGINX
├── nginx.conf                   # Configuración actual de NGINX
├── nginx.conf.template          # Plantilla para configuración NGINX
├── README.md                    # Documentación del proyecto
│
├── api/                         # Servicio API principal
│   ├── app.py                   # Aplicación Flask principal
│   ├── Dockerfile.api          # Dockerfile para el servicio API
│   └── requirements.txt        # Dependencias Python del API
│
├── certs/                       # Certificados SSL para HTTPS
│   ├── server.crt              # Certificado SSL
│   └── server.key              # Clave privada SSL
│
├── common/                      # Utilidades compartidas entre microservicios
│   ├── __init__.py
│   ├── auth_utils.py           # Utilidades de autenticación y autorización
│   ├── encryption_utils.py     # Funciones de cifrado y descifrado
│   ├── minio_utils.py          # Cliente y utilidades para MinIO
│   ├── qdrant_utils.py         # Cliente y utilidades para Qdrant
│   └── requirements.txt        # Dependencias comunes compartidas
│
└── mcps/                        # Microservicios MCP (Modular Compliance Processing)
    ├── Dockerfile              # Dockerfile base para servicios MCP
    └── multi-mcp/              # Microservicio MCP adaptable
        ├── config.yaml         # Configuración adaptable del servicio
        ├── requirements.txt    # Dependencias específicas
        └── src/                # Código fuente del microservicio
            ├── __init__.py
            ├── main.py         # Punto de entrada adaptable
            └── utils.py        # Utilidades específicas del sector
```

## Instalación y Ejecución

### Requisitos Previos
- Docker y Docker Compose instalados
- Python 3.8+ (para desarrollo local)
- Certificados SSL válidos (para producción)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd TraksysEdgeAi
   ```

2. **Configurar variables de entorno**
   ```bash
   cp .env.example .env
   # Editar .env con tus configuraciones específicas
   ```

3. **Generar configuración de NGINX**
   ```bash
   chmod +x generate_nginx_conf.sh
   ./generate_nginx_conf.sh
   ```

4. **Instalar dependencias (desarrollo local)**
   ```bash
   # Dependencias comunes
   pip install -r common/requirements.txt
   
   # Dependencias del API
   pip install -r api/requirements.txt
   
   # Dependencias del MCP
   pip install -r mcps/multi-mcp/requirements.txt
   ```

5. **Levantar servicios con Docker Compose**
   ```bash
   # Construir e iniciar todos los servicios
   docker-compose up --build
   
   # En segundo plano
   docker-compose up -d --build
   ```

6. **Verificar servicios**
   - API Principal: https://localhost/api
   - Health Check: https://localhost/health
   - Documentación: https://localhost/docs

## Variables de Entorno

### Archivo .env requerido
```env
# Configuración de CORS
CORS_ORIGINS=http://localhost:8080,https://tudominio.com

# Seguridad
ENCRYPTION_KEY=tu_clave_secreta_de_32_caracteres
JWT_SECRET_KEY=tu_jwt_secreto_super_seguro

# MinIO (Almacenamiento de objetos)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=usuario_minio
MINIO_SECRET_KEY=clave_minio_segura
MINIO_SECURE=false
MINIO_BUCKET_NAME=traksys-data

# Qdrant (Base de datos vectorial)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=tu_api_key_qdrant

# URLs de servicios internos
API_URL=http://api:8000
TOKEN_API_URL=http://token_api:8001
MCP_URL=http://mcp:8002

# Configuración de NGINX
NGINX_HOST=localhost
NGINX_PORT=443
SSL_CERT_PATH=/etc/nginx/certs/server.crt
SSL_KEY_PATH=/etc/nginx/certs/server.key

# Configuración específica del sector
SECTOR_CONFIG=manufacturing|healthcare|retail|finance|custom
DATA_PROCESSING_MODE=realtime|batch|streaming
```

## Servicios Principales

### 1. API Principal (`api/`)
- **Puerto**: 8000 (interno)
- **Función**: Procesamiento de datos en tiempo real adaptable a cualquier sector
- **Endpoints principales**:
  - `POST /api/data/process` - Procesamiento de datos según sector
  - `GET /api/compliance/status` - Estado de cumplimiento normativo
  - `POST /api/analytics/predict` - Análisis predictivo adaptable
  - `GET /api/health` - Health check del sistema

### 2. Microservicio MCP (`mcps/multi-mcp/`)
- **Puerto**: 8002 (interno)
- **Función**: Módulos especializados adaptables a cualquier industria
- **Características adaptables**:
  - Validación de normativas específicas del sector
  - Análisis predictivo configurable
  - Gestión de auditorías automatizadas
  - Procesamiento de datos estructurados y no estructurados

### 3. NGINX Proxy
- **Puerto**: 443 (HTTPS)
- **Funciones**:
  - Terminación SSL/TLS
  - Enrutamiento inteligente
  - Rate limiting y DDoS protection
  - Compresión y caching
  - Configuración dinámica según variables de entorno

### 4. Common Utilities (`common/`)
- **auth_utils.py**: JWT tokens, OAuth2, validación de permisos multi-sector
- **encryption_utils.py**: AES-256, gestión de claves, hashing seguro
- **minio_utils.py**: Operaciones CRUD en almacenamiento de objetos
- **qdrant_utils.py**: Búsqueda vectorial, embeddings, similitud adaptable

## Casos de Uso por Sector

### Manufactura
- Monitoreo predictivo de equipos
- Control de calidad automatizado
- Gestión de cumplimiento normativo industrial
- Optimización de procesos en tiempo real

### Salud
- Procesamiento de datos médicos sensibles
- Cumplimiento HIPAA/GDPR
- Análisis de imágenes médicas
- Gestión de historiales clínicos

### Finanzas
- Detección de fraudes en tiempo real
- Cumplimiento normativo financiero
- Análisis de riesgos crediticios
- Procesamiento de transacciones

### Retail
- Análisis de comportamiento del cliente
- Gestión de inventario inteligente
- Procesamiento de transacciones
- Cumplimiento de protección de datos

### IoT y Smart Cities
- Procesamiento de datos de sensores
- Gestión de tráfico inteligente
- Monitoreo ambiental
- Optimización de recursos urbanos

## Notas de Seguridad

### Certificados SSL
- Para producción, usar certificados de Let's Encrypt o CA reconocida
- Renovar certificados automáticamente con certbot
- Mantener claves privadas con permisos 600

### Gestión de Secretos
- Nunca commitear archivos `.env` o certificados
- Usar Docker secrets en producción
- Rotar claves de cifrado regularmente

### Cumplimiento Normativo
- GDPR/LOPD para datos europeos
- HIPAA para datos médicos
- PCI-DSS para datos financieros
- Configuración adaptable según jurisdicción

### Firewall y Red
- Restringir acceso a puertos internos
- Usar redes Docker personalizadas
- Implementar rate limiting por IP
- Monitorear logs de acceso

## Monitoreo y Mantenimiento

### Health Checks
```bash
# Verificar estado de servicios
docker-compose ps
docker-compose logs api
docker-compose logs mcp

# Ver métricas específicas del sector
curl -X GET https://localhost/api/metrics
```

### Métricas de Rendimiento
- Tiempo de respuesta de APIs por sector
- Uso de CPU/memoria por servicio
- Tasa de errores y timeouts
- Volumen de datos procesados por industria

## Solución de Problemas

### Problemas Comunes
1. **Certificados SSL no válidos**: Verificar fechas y CA
2. **Errores de CORS**: Revisar configuración en .env
3. **Timeouts en APIs**: Ajustar timeouts en NGINX
4. **Problemas de memoria**: Verificar límites en docker-compose

### Comandos Útiles
```bash
# Ver logs en tiempo real
docker-compose logs -f [servicio]

# Reiniciar servicio específico
docker-compose restart [servicio]

# Acceso a contenedor
docker-compose exec [servicio] /bin/bash

# Ver configuración actual
docker-compose config
```

## Desarrollo y Contribuciones

### Estructura de Ramas
- `main`: Código estable en producción
- `develop`: Integración de nuevas features
- `feature/[sector]`: Desarrollo específico por sector
- `hotfix/[bug]`: Correcciones críticas

### Guía de Contribución
1. Fork del repositorio
2. Crear rama feature específica del sector
3. Commits semánticos con scope del sector
4. Pull request con descripción detallada
5. Code review obligatorio
6. Tests de integración por sector

## Contacto y Soporte

**Desarrollado por**: stefa2980 y equipo colaborativo
**Versión actual**: 2.0.0 (universal sector-agnostic)
**Licencia**: MIT

Para soporte técnico:
- 📧 Email: soporte@traksysedge.ai
- 🐛 Issues: [GitHub Issues](https://github.com/stefa2980/TraksysEdgeAi/issues)
- 📖 Wiki: [Documentación completa](https://github.com/stefa2980/TraksysEdgeAi/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/stefa2980/TraksysEdgeAi/discussions)

---
**Última actualización**: 2024 - Plataforma universal para cualquier sector

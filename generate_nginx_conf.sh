#!/bin/bash

# Obtener el valor de CORS_ORIGINS desde el archivo .env o un valor por defecto
CORS_ORIGINS=${CORS_ORIGINS:-"http://localhost:8080"}

# Sustituir la variable en el template y generar nginx.conf
sed "s|\${CORS_ORIGINS}|${CORS_ORIGINS}|g" nginx.conf.template > nginx.conf

echo "Generated nginx.conf with CORS_ORIGINS=${CORS_ORIGINS}"
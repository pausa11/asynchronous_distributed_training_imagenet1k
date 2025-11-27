#!/bin/bash
# Script para iniciar un Worker
# Este script puede ejecutarse en cualquier PC de la LAN

# CONFIGURACIÓN - EDITA ESTOS VALORES
MASTER_ADDR="localhost"  # Cambia esto a la IP del Parameter Server
MASTER_PORT="30005"
WORKER_RANK=1  # Cambia esto: 1 para el primer worker, 2 para el segundo, etc.
WORLD_SIZE=3   # 1 PS + 2 Workers (debe coincidir con el PS)
EPOCHS=5

# Dataset configuration
DATASET_URL="data/tiny-imagenet-200"
VAL_DATASET_URL="data/tiny-imagenet-200"
USE_WEBDATASET=false

echo "=========================================="
echo "Starting Worker $WORKER_RANK"
echo "=========================================="
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Worker Rank: $WORKER_RANK"
echo "World Size: $WORLD_SIZE"
echo "Epochs: $EPOCHS"
echo "Dataset: $DATASET_URL"
echo "=========================================="
echo ""

# Verificar que el dataset existe
if [ ! -d "$DATASET_URL" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_URL"
    echo "Por favor, asegúrate de que el dataset esté disponible en esta máquina."
    exit 1
fi

# Iniciar el Worker
python -m src.rpc_worker \
    --rank $WORKER_RANK \
    --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --dataset_url $DATASET_URL \
    --val_dataset_url $VAL_DATASET_URL \
    --epochs $EPOCHS

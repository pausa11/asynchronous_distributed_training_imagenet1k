#!/bin/bash
# Script para iniciar el Parameter Server
# Este script debe ejecutarse en la PC que actuará como Parameter Server

# Configuración
WORLD_SIZE=2  # 1 PS + 2 Workers
MASTER_PORT="30005"
CHECKPOINT_DIR="checkpoints"

# Obtener la IP de esta máquina
# Puedes sobrescribir manualmente la IP aquí si es necesario:
# IP="192.168.20.3"
# O dejar que se detecte automáticamente:
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo "=========================================="
echo "Starting Parameter Server"
echo "=========================================="
echo "IP Address: $IP"
echo "Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "=========================================="
echo ""
echo "Para conectar workers desde otras máquinas, usa:"
echo "  MASTER_ADDR=$IP"
echo "  MASTER_PORT=$MASTER_PORT"
echo "=========================================="
echo ""

# Iniciar el Parameter Server
python run_ps.py \
    --rank 0 \
    --world_size $WORLD_SIZE \
    --master_addr $IP \
    --master_port $MASTER_PORT \
    --checkpoint_dir $CHECKPOINT_DIR

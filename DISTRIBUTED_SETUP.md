# Guía: Ejecutar Entrenamiento Distribuido en LAN

Esta guía explica cómo ejecutar el Parameter Server en una PC y conectar workers desde otras PCs en la misma red local (LAN).

## Arquitectura

- **Parameter Server (PS)**: Almacena y actualiza los pesos del modelo
- **Workers**: Entrenan el modelo con datos locales y envían gradientes al PS

## Requisitos Previos

### En todas las máquinas:

1. **Python y dependencias instaladas**:
   ```bash
   pip install torch torchvision
   ```

2. **Código del proyecto clonado** en la misma ruta relativa

3. **Dataset disponible**:
   - Por defecto, el script busca `data/tiny-imagenet-200`
   - Puedes cambiar la ruta en `start_worker.sh`

4. **Firewall configurado**:
   - El puerto `30005` debe estar abierto para comunicación TCP
   - En macOS: `System Preferences > Security & Privacy > Firewall`

## Paso 1: Iniciar el Parameter Server

En la PC que actuará como Parameter Server:

```bash
cd /Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k
./start_ps.sh
```

El script mostrará:
- La IP de la máquina (ej: `192.168.1.100`)
- El puerto (por defecto: `30005`)
- Instrucciones para conectar workers

**Anota la IP mostrada**, la necesitarás para configurar los workers.

## Paso 2: Configurar y Ejecutar Workers

### En cada PC worker:

1. **Edita `start_worker.sh`**:
   ```bash
   nano start_worker.sh
   ```

2. **Modifica las siguientes variables**:
   ```bash
   MASTER_ADDR="192.168.1.100"  # IP del Parameter Server (del Paso 1)
   WORKER_RANK=1                # 1 para el primer worker, 2 para el segundo
   WORLD_SIZE=3                 # 1 PS + 2 Workers
   ```

3. **Verifica que el dataset existe**:
   ```bash
   ls data/tiny-imagenet-200
   ```

4. **Ejecuta el worker**:
   ```bash
   ./start_worker.sh
   ```

### Ejemplo con 2 Workers:

**Worker 1** (PC #2):
```bash
MASTER_ADDR="192.168.1.100"
WORKER_RANK=1
WORLD_SIZE=3
```

**Worker 2** (PC #3):
```bash
MASTER_ADDR="192.168.1.100"
WORKER_RANK=2
WORLD_SIZE=3
```

## Configuración Avanzada

### Cambiar el número de workers

Si quieres usar más o menos workers:

1. **En `start_ps.sh`**, cambia:
   ```bash
   WORLD_SIZE=4  # 1 PS + 3 Workers
   ```

2. **En cada `start_worker.sh`**, actualiza:
   ```bash
   WORLD_SIZE=4
   WORKER_RANK=1  # 1, 2, 3 para cada worker
   ```

### Usar WebDataset en lugar de ImageFolder

En `start_worker.sh`, cambia:
```bash
DATASET_URL="gs://tu-bucket/tiny-imagenet-wds/train"
VAL_DATASET_URL="gs://tu-bucket/tiny-imagenet-wds/val"
USE_WEBDATASET=true
```

### Cambiar hiperparámetros

Edita directamente los archivos fuente:
- **Learning rate**: `src/rpc_ps.py`, línea 24
- **Batch size**: `src/rpc_worker.py`, líneas 75 y 79
- **Épocas**: `start_worker.sh`, variable `EPOCHS`

## Verificación de Conectividad

### Probar conexión de red

Desde una PC worker, verifica que puedes alcanzar el PS:

```bash
ping 192.168.1.100
telnet 192.168.1.100 30005
```

### Revisar logs

Los logs se muestran en la terminal. Busca:
- `Parameter Server is running...` en el PS
- `Worker X started...` en cada worker
- Mensajes de progreso de entrenamiento

## Troubleshooting

### Error: "Connection refused"

**Causa**: El firewall bloquea el puerto o el PS no está corriendo.

**Solución**:
1. Verifica que el PS esté corriendo
2. Abre el puerto 30005 en el firewall del PS
3. Verifica la IP con `ipconfig getifaddr en0`

### Error: "No module named 'src'"

**Causa**: El script no se ejecuta desde el directorio raíz del proyecto.

**Solución**:
```bash
cd /ruta/al/proyecto
./start_worker.sh
```

### Error: "Dataset directory not found"

**Causa**: El dataset no está en la ruta especificada.

**Solución**:
1. Verifica la ruta: `ls data/tiny-imagenet-200`
2. Descarga el dataset si es necesario
3. Actualiza `DATASET_URL` en `start_worker.sh`

### Workers no se conectan

**Causa**: `WORLD_SIZE` o `WORKER_RANK` incorrectos.

**Solución**:
- Asegúrate de que `WORLD_SIZE` sea igual en PS y todos los workers
- Cada worker debe tener un `WORKER_RANK` único (1, 2, 3, ...)
- El PS siempre tiene rank 0

## Monitoreo

### Ver estadísticas de entrenamiento

Los archivos de estadísticas se guardan en:
```
stats/
├── <hostname>_ps_<timestamp>.json
├── <hostname>_worker1_<timestamp>.json
└── <hostname>_worker2_<timestamp>.json
```

### Ver checkpoints

Los checkpoints se guardan en:
```
checkpoints/
├── best.pth   # Mejor modelo según validación
└── last.pth   # Último checkpoint guardado
```

## Detener el Entrenamiento

1. **Ctrl+C** en cada terminal de worker
2. **Ctrl+C** en la terminal del PS
3. Los checkpoints se guardan automáticamente

## Ejemplo Completo

### PC 1 (Parameter Server) - IP: 192.168.1.100
```bash
cd /Users/danieltorosoto/universidad/arq-cliente-servidor/asynchronous_distributed_training_imagenet1k
./start_ps.sh
```

### PC 2 (Worker 1) - IP: 192.168.1.101
```bash
cd /path/to/asynchronous_distributed_training_imagenet1k
# Editar start_worker.sh: MASTER_ADDR="192.168.1.100", WORKER_RANK=1
./start_worker.sh
```

### PC 3 (Worker 2) - IP: 192.168.1.102
```bash
cd /path/to/asynchronous_distributed_training_imagenet1k
# Editar start_worker.sh: MASTER_ADDR="192.168.1.100", WORKER_RANK=2
./start_worker.sh
```

¡Listo! El entrenamiento distribuido debería comenzar automáticamente.

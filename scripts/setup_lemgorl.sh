#!/usr/bin/env bash
# scripts/setup_lemgorl.sh
# ============================================================
# Script de instalación y configuración de LemgoRL
# Requisitos: Git, Python 3.9+, SUMO ya instalado
# ============================================================

set -e
echo "======================================================"
echo "  Configurando LemgoRL Benchmark"
echo "======================================================"

# 1. Verificar SUMO_HOME
if [ -z "$SUMO_HOME" ]; then
  echo "❌ SUMO_HOME no está configurado."
  echo "   Ejemplo: export SUMO_HOME=/usr/share/sumo"
  exit 1
fi
echo "✓ SUMO_HOME = $SUMO_HOME"

# 2. Clonar LemgoRL si no existe
LEMGO_DIR="data/lemgo"
if [ ! -d "$LEMGO_DIR" ]; then
  echo "Clonando LemgoRL..."
  git clone https://github.com/IntelligentTrafficSystems/LemgoRL.git "$LEMGO_DIR"
  echo "✓ LemgoRL clonado en $LEMGO_DIR"
else
  echo "✓ LemgoRL ya existe en $LEMGO_DIR"
fi

# 3. Instalar dependencias Python
echo "Instalando dependencias Python..."
pip install -r requirements.txt
echo "✓ Dependencias instaladas"

# 4. Instalar el paquete en modo editable
echo "Instalando paquete ATSC en modo editable..."
pip install -e .
echo "✓ Paquete instalado"

# 5. Verificar TraCI
python -c "import traci; print('✓ TraCI disponible')" 2>/dev/null || \
  echo "⚠ TraCI no disponible — verifica SUMO_HOME y que tools/ esté en PYTHONPATH"

# 6. Crear directorios de salida
mkdir -p data/offline_dataset checkpoints/offline checkpoints/online results
echo "✓ Directorios de trabajo creados"

echo ""
echo "======================================================"
echo "  Setup completado. Próximos pasos:"
echo "  1. Ajusta configs/config.yaml con tus rutas de red"
echo "  2. python main.py --phase 1   # Genera dataset + entrena DT"
echo "  3. python main.py --phase 2   # Valida safety + coordinación"
echo "  4. python main.py --phase 3   # Fine-tuning + benchmarking"
echo "  O todo junto: python main.py --phase all"
echo "======================================================"

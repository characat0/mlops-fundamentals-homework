import sys
from pathlib import Path

# Permite importar el paquete src/ al correr pytest desde la raiz del repo
sys.path.insert(0, str(Path(__file__).parent))

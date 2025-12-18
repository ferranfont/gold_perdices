"""
Configuración global para detección de fractales en Gold (GC)
"""
from pathlib import Path

# ============================================================================
# RANGO DE FECHAS
# ============================================================================
START_DATE = "2025-02-02"
END_DATE = "2025-04-25"

# ============================================================================
# DIRECTORIOS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FRACTALS_DIR = OUTPUTS_DIR / "fractals"

# ============================================================================
# PARÁMETROS DE FRACTALES ZIGZAG (PRECIO)
# ============================================================================
MIN_CHANGE_PCT_MINOR = 0.50    # 0.50% umbral para fractales pequeños
MIN_CHANGE_PCT_MAJOR = 2.1     # 2.1% umbral para fractales grandes

# ============================================================================
# PARÁMETROS RSI
# ============================================================================
RSI_PERIOD = 10                # Período para calcular RSI (estándar: 14)
RSI_SMOOTH_PERIOD = 0          # Período para suavizar RSI con EMA (0 = sin suavizado)
RSI_OVERBOUGHT = 60            # Nivel de sobrecompra
RSI_OVERSOLD = 40              # Nivel de sobreventa

# Regla de resample para cálculo de RSI (ej: '15T', '1H')
# - Si RSI_RESAMPLE es None o cadena vacía => no se aplica resample
# - Si RSI_RESAMPLE es una cadena (ej: '1H' o '15T') => se usa esa regla
# - Si RSI_RESAMPLE es True => se usará la regla definida en RSI_RESAMPLE_TO_PERIOD
RSI_RESAMPLE = True
RSI_RESAMPLE_TO_PERIOD = '1H'  # Solo usada si RSI_RESAMPLE == True

# Parámetros de detección de fractales en el RSI
MIN_CHANGE_PCT_RSI = 9.0       # Cambio mínimo en puntos de RSI para considerar un fractal
MINOR_ZIG_ZAG_RSI = True       # True = usar fractales MINOR, False = usar fractales MAJOR (legacy, no usado)

# ============================================================================
# PARÁMETROS FIBONACCI
# ============================================================================
FIBO_LEVELS = [0.0, 0.382, 0.5, 0.618, 1.0]  # Niveles de retroceso estándar

# Mínimo tamaño de impulso (% del impulso medio) para considerar un movimiento
# Si el rango del movimiento es menor que MINIMUM_IMPULSE_FACTOR% del rango medio, se ignora
MINIMUM_IMPULSE_FACTOR = 60    # porcentaje (ej: 60 = 60%)

# ============================================================================
# PARÁMETROS DE SEÑALES DE ENTRADA (DIVERGENCIAS)
# ============================================================================
REQUIRE_DOWNTREND = True       # Requiere tendencia bajista MAJOR para señal
REQUIRE_DIVERGENCE = True      # Requiere divergencia alcista para señal
FIBO_LEVEL_FILTER = 0.382        # Nivel Fibonacci mínimo (0.0-1.0) para filtrar señales

# ============================================================================
# PARÁMETROS DE AGREGACIÓN (No usado - datos ya vienen en OHLC)
# ============================================================================
AGGREGATION_WINDOW = 60        # 60 segundos (velas de 1 minuto)

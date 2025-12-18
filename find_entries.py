"""
Detección de puntos de entrada para Gold (GC)
Basado en fractales RSI, tendencia bajista y niveles Fibonacci
"""

import pandas as pd
from pathlib import Path
from config import (
    START_DATE, END_DATE, FRACTALS_DIR, RSI_OVERSOLD,
    MIN_CHANGE_PCT_RSI
)
from plot_day import detect_rsi_fractals


def is_downtrend_major(df_fractals_major, current_timestamp):
    """
    Determina si el precio está en tendencia bajista según fractales MAJOR.

    Una tendencia bajista existe cuando:
    - El último fractal confirmado es un PICO (verde)
    - El precio actual está descendiendo hacia el próximo VALLE (rojo)

    Args:
        df_fractals_major: DataFrame con fractales MAJOR
        current_timestamp: Timestamp actual para evaluar

    Returns:
        bool: True si está en tendencia bajista, False en caso contrario
    """
    if df_fractals_major is None or df_fractals_major.empty:
        return False

    # Convertir timestamps a datetime si no lo están
    df_major = df_fractals_major.copy()
    df_major['timestamp'] = pd.to_datetime(df_major['timestamp'])
    current_ts = pd.to_datetime(current_timestamp)

    # Obtener fractales anteriores al timestamp actual
    fractals_before = df_major[df_major['timestamp'] <= current_ts].sort_values('timestamp')

    if len(fractals_before) < 1:
        return False

    # El último fractal debe ser un PICO para estar en tendencia bajista
    last_fractal = fractals_before.iloc[-1]

    return last_fractal['type'] == 'PICO'


def get_current_fibonacci_level(df, df_fractals_major, fibo_levels, current_timestamp):
    """
    Determina qué nivel de Fibonacci está más cercano por debajo del precio actual.

    Args:
        df: DataFrame con datos de precio
        df_fractals_major: DataFrame con fractales MAJOR
        fibo_levels: Dict con niveles Fibonacci
        current_timestamp: Timestamp actual

    Returns:
        tuple: (nivel_fibo, precio_fibo) o (None, None) si no hay nivel
    """
    if fibo_levels is None or 'upward_moves' not in fibo_levels:
        return None, None

    current_ts = pd.to_datetime(current_timestamp)

    # Obtener precio actual
    df_temp = df.copy()
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
    current_data = df_temp[df_temp['timestamp'] == current_ts]

    if current_data.empty:
        return None, None

    current_price = current_data['close'].iloc[0]

    # Buscar el movimiento alcista activo (retroceso actual)
    for move in fibo_levels['upward_moves']:
        swing_high_ts = pd.to_datetime(move['swing_high_timestamp'])
        retracement_end_ts = move.get('retracement_end_timestamp')

        if retracement_end_ts:
            retracement_end_ts = pd.to_datetime(retracement_end_ts)

        # Verificar si estamos en el período de retroceso
        if retracement_end_ts is None:
            in_range = current_ts >= swing_high_ts
        else:
            in_range = swing_high_ts <= current_ts <= retracement_end_ts

        if in_range:
            # Encontrar el nivel de Fibonacci más cercano por debajo del precio actual
            levels = move['levels']
            best_level = None
            best_price = None

            for level, price in levels.items():
                # Solo considerar niveles de retroceso (38.2%, 50%, 61.8%)
                if level in [0.382, 0.5, 0.618]:
                    if price < current_price:
                        if best_price is None or price > best_price:
                            best_level = level
                            best_price = price

            return best_level, best_price

    return None, None


def detect_bullish_divergences(df, df_fractals_minor, df_rsi_valles, require_downtrend=True,
                               df_fractals_major=None, fibo_levels=None, fibo_level_filter=0.5):
    """
    Detecta divergencias alcistas (bullish divergence) comparando fractales MINOR del precio
    con fractales VALLE del RSI.

    Divergencia alcista = Precio hace mínimos descendentes, pero RSI hace mínimos ascendentes
    Puede ser divergencia doble, triple o múltiple.

    Args:
        df: DataFrame con datos OHLC y RSI
        df_fractals_minor: DataFrame con fractales MINOR del precio
        df_rsi_valles: DataFrame con fractales VALLE del RSI (ya filtrados < RSI_OVERSOLD)
        require_downtrend: Si True, requiere tendencia bajista MAJOR
        df_fractals_major: DataFrame con fractales MAJOR (para verificar tendencia)
        fibo_levels: Dict con niveles Fibonacci (opcional)
        fibo_level_filter: Nivel de Fibonacci mínimo

    Returns:
        DataFrame con divergencias detectadas
    """
    print("\n" + "="*70)
    print("DETECCIÓN DE DIVERGENCIAS ALCISTAS")
    print("="*70)
    print(f"Criterios:")
    print(f"  1. Tendencia bajista MAJOR: {'SÍ' if require_downtrend else 'NO'}")
    print(f"  2. Precio por debajo Fibonacci {fibo_level_filter*100:.1f}%: {'SÍ' if fibo_levels else 'NO'}")
    print(f"  3. Divergencia: Precio hace mínimos descendentes, RSI mínimos ascendentes")
    print(f"  4. Soporta divergencias múltiples (doble, triple, etc.)")
    print("="*70)

    if df_fractals_minor is None or df_fractals_minor.empty:
        print("[ERROR] No hay fractales MINOR disponibles")
        return pd.DataFrame()

    if df_rsi_valles.empty:
        print("[INFO] No hay fractales VALLE del RSI para comparar")
        return pd.DataFrame()

    # Filtrar solo VALLES de fractales MINOR del precio
    df_price_valles = df_fractals_minor[df_fractals_minor['type'] == 'VALLE'].copy()
    df_price_valles = df_price_valles.sort_values('timestamp').reset_index(drop=True)

    print(f"[INFO] Fractales VALLE MINOR (precio): {len(df_price_valles)}")
    print(f"[INFO] Fractales VALLE RSI < {RSI_OVERSOLD}: {len(df_rsi_valles)}")

    divergences = []

    # Para cada fractal VALLE del RSI, buscar divergencia con fractales MINOR anteriores
    df_rsi_valles_sorted = df_rsi_valles.sort_values('timestamp').reset_index(drop=True)

    for i in range(len(df_rsi_valles_sorted)):
        current_rsi_valle = df_rsi_valles_sorted.iloc[i]
        current_rsi_ts = pd.to_datetime(current_rsi_valle['timestamp'])
        current_rsi_value = current_rsi_valle['rsi_value']

        # Encontrar el fractal MINOR del precio más cercano a este timestamp del RSI
        price_valles_before = df_price_valles[pd.to_datetime(df_price_valles['timestamp']) <= current_rsi_ts]

        if price_valles_before.empty:
            continue

        # El fractal de precio más reciente antes del RSI actual
        current_price_valle = price_valles_before.iloc[-1]
        current_price = current_price_valle['price']
        current_price_ts = pd.to_datetime(current_price_valle['timestamp'])

        # Buscar fractales RSI anteriores para comparar
        previous_rsi_valles = df_rsi_valles_sorted.iloc[:i]

        if previous_rsi_valles.empty:
            continue

        # Buscar divergencias múltiples
        divergence_chain = []

        for j in range(len(previous_rsi_valles)-1, -1, -1):
            prev_rsi_valle = previous_rsi_valles.iloc[j]
            prev_rsi_ts = pd.to_datetime(prev_rsi_valle['timestamp'])
            prev_rsi_value = prev_rsi_valle['rsi_value']

            # Encontrar fractal MINOR del precio correspondiente
            prev_price_valles = df_price_valles[pd.to_datetime(df_price_valles['timestamp']) <= prev_rsi_ts]

            if prev_price_valles.empty:
                continue

            prev_price_valle = prev_price_valles.iloc[-1]
            prev_price = prev_price_valle['price']
            prev_price_ts = pd.to_datetime(prev_price_valle['timestamp'])

            # Verificar divergencia: precio baja o igual, RSI sube
            if prev_price >= current_price and prev_rsi_value < current_rsi_value:
                # Tenemos divergencia
                divergence_chain.append({
                    'price': prev_price,
                    'price_ts': prev_price_ts,
                    'rsi': prev_rsi_value,
                    'rsi_ts': prev_rsi_ts
                })
            else:
                # Si no hay divergencia, rompemos la cadena
                break

        # Si encontramos al menos una divergencia
        if len(divergence_chain) > 0:
            # Verificar filtros adicionales
            if require_downtrend and df_fractals_major is not None:
                is_downtrend = is_downtrend_major(df_fractals_major, current_price_ts)
                if not is_downtrend:
                    continue
            else:
                is_downtrend = False

            # Verificar Fibonacci
            fibo_level, fibo_price = None, None
            if fibo_levels is not None and df_fractals_major is not None:
                fibo_level, fibo_price = get_current_fibonacci_level(
                    df, df_fractals_major, fibo_levels, current_price_ts
                )

                if fibo_level is not None and fibo_level < fibo_level_filter:
                    continue

            # Construir registro de divergencia
            divergence_order = len(divergence_chain) + 1  # +1 para incluir el punto actual

            # Añadir a la lista (del más antiguo al más reciente)
            divergence_points = []
            for k, point in enumerate(reversed(divergence_chain)):
                divergence_points.append({
                    'divergence_id': len(divergences) + 1,
                    'divergence_order': divergence_order,
                    'point_number': k + 1,
                    'price': point['price'],
                    'price_timestamp': point['price_ts'],
                    'rsi_value': point['rsi'],
                    'rsi_timestamp': point['rsi_ts'],
                    'downtrend': is_downtrend,
                    'fibo_level': fibo_level if fibo_level is not None else 'N/A',
                    'fibo_price': fibo_price if fibo_price is not None else 'N/A',
                    'tag': f"DIV{divergence_order}P{k+1}"
                })

            # Añadir el punto actual (el más reciente)
            divergence_points.append({
                'divergence_id': len(divergences) + 1,
                'divergence_order': divergence_order,
                'point_number': divergence_order,
                'price': current_price,
                'price_timestamp': current_price_ts,
                'rsi_value': current_rsi_value,
                'rsi_timestamp': current_rsi_ts,
                'downtrend': is_downtrend,
                'fibo_level': fibo_level if fibo_level is not None else 'N/A',
                'fibo_price': fibo_price if fibo_price is not None else 'N/A',
                'tag': f"DIV{divergence_order}P{divergence_order}_ENTRY"
            })

            divergences.extend(divergence_points)

    df_divergences = pd.DataFrame(divergences)

    if not df_divergences.empty:
        num_unique_divs = df_divergences['divergence_id'].nunique()
        print(f"\n[OK] Divergencias detectadas: {num_unique_divs}")
        print(f"[INFO] Puntos totales: {len(df_divergences)}")

        # Resumen por tipo
        div_types = df_divergences.groupby('divergence_order').size()
        print(f"\n[INFO] Tipos de divergencias:")
        for order, count in div_types.items():
            print(f"  - Divergencia de orden {order}: {count // order} ocurrencias")
    else:
        print("[INFO] No se detectaron divergencias alcistas")

    return df_divergences


def find_entry_signals(df, df_fractals_major, df_fractals_minor=None, fibo_levels=None,
                       require_downtrend=True, fibo_level_filter=0.5,
                       require_divergence=True):
    """
    Encuentra señales de entrada basadas en:
    1. Fractales VALLE del RSI por debajo de RSI_OVERSOLD
    2. (Opcional) Tendencia bajista en fractales MAJOR
    3. (Opcional) Precio por debajo de nivel Fibonacci específico
    4. (NUEVO) Divergencia alcista entre precio y RSI

    Args:
        df: DataFrame con datos OHLC y RSI
        df_fractals_major: DataFrame con fractales MAJOR
        df_fractals_minor: DataFrame con fractales MINOR (para divergencias)
        fibo_levels: Dict con niveles Fibonacci (opcional)
        require_downtrend: Si True, requiere tendencia bajista
        fibo_level_filter: Nivel de Fibonacci mínimo (0.382, 0.5, 0.618)
        require_divergence: Si True, requiere divergencia alcista

    Returns:
        DataFrame con señales de entrada detectadas
    """
    print("\n" + "="*70)
    print("DETECCIÓN DE SEÑALES DE ENTRADA")
    print("="*70)
    print(f"Filtros activos:")
    print(f"  - Tendencia bajista MAJOR: {'SÍ' if require_downtrend else 'NO'}")
    print(f"  - Nivel Fibonacci mínimo: {fibo_level_filter*100:.1f}% {'(activo)' if fibo_levels else '(desactivado - sin datos Fibo)'}")
    print(f"  - Divergencia alcista: {'SÍ' if require_divergence else 'NO'}")
    print("="*70)

    # Detectar fractales del RSI
    df_rsi_fractals = detect_rsi_fractals(df, min_change_pct=MIN_CHANGE_PCT_RSI)

    if df_rsi_fractals.empty:
        print("[INFO] No se detectaron fractales en el RSI")
        return pd.DataFrame()

    # Filtrar solo VALLES por debajo de RSI_OVERSOLD
    df_rsi_valles = df_rsi_fractals[
        (df_rsi_fractals['type'] == 'VALLE') &
        (df_rsi_fractals['rsi_value'] < RSI_OVERSOLD)
    ].copy()

    print(f"[INFO] Fractales VALLE RSI < {RSI_OVERSOLD}: {len(df_rsi_valles)}")

    if df_rsi_valles.empty:
        return pd.DataFrame()

    # Si se requiere divergencia, usar la nueva función
    if require_divergence:
        if df_fractals_minor is None or df_fractals_minor.empty:
            print("[ERROR] Se requiere divergencia pero no hay fractales MINOR disponibles")
            return pd.DataFrame()

        return detect_bullish_divergences(
            df, df_fractals_minor, df_rsi_valles,
            require_downtrend=require_downtrend,
            df_fractals_major=df_fractals_major,
            fibo_levels=fibo_levels,
            fibo_level_filter=fibo_level_filter
        )

    # Código original (sin divergencia) - mantenido para compatibilidad
    signals = []

    for idx, row in df_rsi_valles.iterrows():
        timestamp = row['timestamp']
        rsi_value = row['rsi_value']

        # Obtener precio en ese momento
        df_temp = df[df['timestamp'] == timestamp]
        if df_temp.empty:
            continue

        price = df_temp['close'].iloc[0]

        # Verificar tendencia bajista
        is_downtrend = is_downtrend_major(df_fractals_major, timestamp)

        if require_downtrend and not is_downtrend:
            continue

        # Verificar nivel Fibonacci
        fibo_level, fibo_price = None, None

        if fibo_levels is not None:
            fibo_level, fibo_price = get_current_fibonacci_level(
                df, df_fractals_major, fibo_levels, timestamp
            )

            # Si se requiere nivel Fibonacci, verificar que esté por debajo
            if fibo_level is not None and fibo_level < fibo_level_filter:
                continue

        # Añadir señal
        signals.append({
            'timestamp': timestamp,
            'rsi_value': rsi_value,
            'price': price,
            'downtrend': is_downtrend,
            'fibo_level': fibo_level if fibo_level is not None else 'N/A',
            'fibo_price': fibo_price if fibo_price is not None else 'N/A'
        })

    df_signals = pd.DataFrame(signals)

    print(f"\n[OK] Señales de entrada detectadas: {len(df_signals)}")

    if not df_signals.empty:
        print("\n" + "-"*70)
        print("SEÑALES DETECTADAS:")
        print("-"*70)
        for i, signal in df_signals.iterrows():
            print(f"Señal #{i+1}:")
            print(f"  Timestamp: {signal['timestamp']}")
            print(f"  Precio: {signal['price']:.2f}")
            print(f"  RSI: {signal['rsi_value']:.2f}")
            print(f"  Tendencia bajista: {'SÍ' if signal['downtrend'] else 'NO'}")
            print(f"  Nivel Fibonacci: {signal['fibo_level']}")
            if signal['fibo_price'] != 'N/A':
                print(f"  Precio Fibonacci: {signal['fibo_price']:.2f}")
            print()

    return df_signals


if __name__ == "__main__":
    from find_fractals import load_date_range
    from analyze_fibonacci import analyze_fibonacci_range

    print("Cargando datos...")

    # Cargar datos
    df = load_date_range(START_DATE, END_DATE)
    if df is None:
        print("Error cargando datos")
        exit(1)

    # Calcular RSI
    from analyze_rsi import calculate_rsi
    from config import RSI_PERIOD, RSI_SMOOTH_PERIOD, RSI_RESAMPLE, RSI_RESAMPLE_TO_PERIOD

    if isinstance(RSI_RESAMPLE, bool) and RSI_RESAMPLE:
        rsi_resample_rule = RSI_RESAMPLE_TO_PERIOD
    elif isinstance(RSI_RESAMPLE, str) and RSI_RESAMPLE:
        rsi_resample_rule = RSI_RESAMPLE
    else:
        rsi_resample_rule = None

    df['rsi'] = calculate_rsi(df, period=RSI_PERIOD, smooth_period=RSI_SMOOTH_PERIOD,
                               resample_rule=rsi_resample_rule)

    # Cargar fractales MAJOR y MINOR
    date_range_str = f"{START_DATE}_{END_DATE}"
    fractal_major_path = FRACTALS_DIR / f"gc_fractals_major_{date_range_str}.csv"
    fractal_minor_path = FRACTALS_DIR / f"gc_fractals_minor_{date_range_str}.csv"

    if not fractal_major_path.exists():
        print(f"Error: No se encuentran fractales MAJOR en {fractal_major_path}")
        exit(1)

    if not fractal_minor_path.exists():
        print(f"Error: No se encuentran fractales MINOR en {fractal_minor_path}")
        exit(1)

    df_fractals_major = pd.read_csv(fractal_major_path)
    df_fractals_minor = pd.read_csv(fractal_minor_path)

    print(f"[INFO] Fractales MAJOR cargados: {len(df_fractals_major)}")
    print(f"[INFO] Fractales MINOR cargados: {len(df_fractals_minor)}")

    # Cargar niveles Fibonacci (opcional)
    fibo_levels = None
    try:
        from analyze_fibonacci import analyze_fibonacci_range
        fibo_result = analyze_fibonacci_range(df_fractals_major, START_DATE, END_DATE)
        if fibo_result is not None:
            fibo_levels = fibo_result
    except Exception as e:
        print(f"[WARNING] No se pudieron cargar niveles Fibonacci: {e}")

    # Encontrar señales de entrada CON DIVERGENCIAS
    divergences = find_entry_signals(
        df,
        df_fractals_major,
        df_fractals_minor=df_fractals_minor,
        fibo_levels=fibo_levels,
        require_downtrend=True,
        fibo_level_filter=0.5,  # 50% por defecto
        require_divergence=True  # ACTIVAR DETECCIÓN DE DIVERGENCIAS
    )

    print("\n" + "="*70)
    print(f"Total de puntos de divergencia encontrados: {len(divergences)}")
    if not divergences.empty:
        num_unique = divergences['divergence_id'].nunique()
        print(f"Total de divergencias únicas: {num_unique}")
    print("="*70)

    # Guardar resultados en CSV
    if not divergences.empty:
        output_filename = f"gc_divergences_{START_DATE}_{END_DATE}.csv"
        output_path = FRACTALS_DIR / output_filename

        divergences.to_csv(output_path, index=False)
        print(f"\n[OK] Divergencias guardadas en: {output_path}")

        # Mostrar resumen detallado por divergencia
        print("\n" + "="*70)
        print("DETALLE DE DIVERGENCIAS")
        print("="*70)

        for div_id in sorted(divergences['divergence_id'].unique()):
            div_data = divergences[divergences['divergence_id'] == div_id].sort_values('point_number')

            print(f"\n--- Divergencia #{div_id} (Orden: {div_data['divergence_order'].iloc[0]}) ---")

            for idx, row in div_data.iterrows():
                is_entry = "ENTRY" in row['tag']
                marker = ">>> " if is_entry else "    "

                print(f"{marker}Punto {row['point_number']}:")
                print(f"{marker}  Tag: {row['tag']}")
                print(f"{marker}  Precio: {row['price']:.2f} @ {row['price_timestamp']}")
                print(f"{marker}  RSI: {row['rsi_value']:.2f} @ {row['rsi_timestamp']}")

                if row['point_number'] == 1:
                    print(f"{marker}  Tendencia bajista: {'SÍ' if row['downtrend'] else 'NO'}")
                    if row['fibo_level'] != 'N/A':
                        print(f"{marker}  Fibonacci: {row['fibo_level']*100:.1f}% @ {row['fibo_price']:.2f}")

            print()
    else:
        print("\n[INFO] No se encontraron divergencias para guardar")

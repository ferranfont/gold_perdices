"""
Script principal de análisis cuantitativo para Gold (GC)
Orquesta la ejecución de:
1. Detección de fractales (find_fractals.py)
2. Análisis RSI (analyze_rsi.py)
3. Análisis Fibonacci Retracements (analyze_fibonacci.py)
4. Generación de gráfico (plot_day.py)
"""
from pathlib import Path
from config import (
    START_DATE, END_DATE, DATA_DIR,
    REQUIRE_DOWNTREND, REQUIRE_DIVERGENCE, FIBO_LEVEL_FILTER
)
from find_fractals import process_fractals_range
from analyze_rsi import analyze_rsi_levels_range
from analyze_fibonacci import analyze_fibonacci_range
from plot_day import plot_range_chart


def main_quant_range(start_date: str, end_date: str):
    """
    Ejecuta el pipeline completo de análisis cuantitativo para un rango de fechas

    Args:
        start_date: Fecha inicial en formato YYYY-MM-DD
        end_date: Fecha final en formato YYYY-MM-DD
    """
    print("\n" + "="*70)
    print("ANÁLISIS CUANTITATIVO - Gold (GC)")
    print("="*70)
    print(f"Rango: {start_date} -> {end_date}")
    print("="*70 + "\n")
 
    # 1. Procesar fractales
    print("\n" + "-"*70)
    print("PASO 1: DETECCIÓN DE FRACTALES")
    print("-"*70)
    fractals_result = process_fractals_range(start_date, end_date)
    if fractals_result is None:
        print("[ERROR] Fallo en detección de fractales")
        return None

    # 2. Analizar niveles RSI (usa el dataframe del paso anterior)
    print("\n" + "-"*70)
    print("PASO 2: ANÁLISIS RSI")
    print("-"*70)
    rsi_result = analyze_rsi_levels_range(fractals_result['df'], start_date, end_date)
    if rsi_result is None:
        print("[ERROR] Fallo en análisis RSI")
        return None

    # 3. Analizar Fibonacci (usa los fractales MAJOR del paso 1)
    print("\n" + "-"*70)
    print("PASO 3: ANÁLISIS FIBONACCI")
    print("-"*70)
    fibo_result = analyze_fibonacci_range(fractals_result['df_fractals_major'], start_date, end_date)
    if fibo_result is None:
        print("[WARNING] No se pudo calcular Fibonacci, continuando sin él...")
        fibo_result = None

    # 4. Detectar divergencias y señales de entrada
    print("\n" + "-"*70)
    print("PASO 4: DETECCIÓN DE DIVERGENCIAS")
    print("-"*70)
    from find_entries import find_entry_signals
    from config import FRACTALS_DIR
    import pandas as pd

    divergences = find_entry_signals(
        rsi_result['df_with_rsi'],
        fractals_result['df_fractals_major'],
        df_fractals_minor=fractals_result['df_fractals_minor'],
        fibo_levels=fibo_result,
        require_downtrend=REQUIRE_DOWNTREND,
        fibo_level_filter=FIBO_LEVEL_FILTER,
        require_divergence=REQUIRE_DIVERGENCE
    )

    # Guardar divergencias en CSV
    if not divergences.empty:
        output_filename = f"gc_divergences_{start_date}_{end_date}.csv"
        output_path = FRACTALS_DIR / output_filename
        divergences.to_csv(output_path, index=False)
        print(f"\n[OK] Divergencias guardadas en: {output_path}")

        # Mostrar resumen de divergencias por movimiento Fibonacci
        if fibo_result and 'upward_moves' in fibo_result:
            print("\n" + "="*70)
            print("DIVERGENCIAS POR MOVIMIENTO FIBONACCI")
            print("="*70)

            for idx, move in enumerate(fibo_result['upward_moves'], 1):
                print(f"\n--- Movimiento #{idx}: {move['swing_low']:.2f} -> {move['swing_high']:.2f} ---")

                # Convertir timestamps a datetime
                swing_low_ts = pd.to_datetime(move['swing_low_timestamp'])
                swing_high_ts = pd.to_datetime(move['swing_high_timestamp'])
                retracement_end_ts = move.get('retracement_end_timestamp')
                if retracement_end_ts:
                    retracement_end_ts = pd.to_datetime(retracement_end_ts)

                # Determinar el rango de tiempo para buscar divergencias
                # Las divergencias alcistas se forman ANTES del swing_low (durante la tendencia bajista)
                # O DURANTE el retroceso (entre swing_high y retracement_end)

                # Buscar el inicio del período de búsqueda (movimiento anterior o inicio de datos)
                if idx > 1:
                    prev_move = fibo_result['upward_moves'][idx - 2]
                    search_start_ts = pd.to_datetime(prev_move['swing_high_timestamp'])
                else:
                    # Para el primer movimiento, buscar desde el inicio de los datos
                    search_start_ts = pd.to_datetime(start_date).tz_localize(swing_low_ts.tz)

                # Filtrar divergencias dentro de este movimiento
                divs_in_move = []
                for div_id in divergences['divergence_id'].unique():
                    div_data = divergences[divergences['divergence_id'] == div_id]
                    entry_point = div_data[div_data['tag'].str.contains('ENTRY', na=False)].iloc[0]
                    entry_ts = pd.to_datetime(entry_point['price_timestamp'])

                    # Verificar si está en el rango del movimiento
                    # Divergencias pueden ocurrir:
                    # 1. Durante la bajada hacia el swing_low (search_start_ts -> swing_low_ts)
                    # 2. Durante el retroceso (swing_high_ts -> retracement_end_ts)

                    in_downtrend = search_start_ts <= entry_ts <= swing_low_ts

                    if retracement_end_ts:
                        in_retracement = swing_high_ts <= entry_ts <= retracement_end_ts
                    else:
                        in_retracement = entry_ts >= swing_high_ts

                    in_range = in_downtrend or in_retracement

                    if in_range:
                        divs_in_move.append(div_id)

                if divs_in_move:
                    print(f"  Divergencias encontradas: {len(divs_in_move)}")
                    for div_id in divs_in_move:
                        div_data = divergences[divergences['divergence_id'] == div_id].sort_values('point_number')
                        entry_row = div_data[div_data['tag'].str.contains('ENTRY', na=False)].iloc[0]

                        print(f"\n  Divergencia #{div_id} (Orden: {entry_row['divergence_order']})")
                        for _, row in div_data.iterrows():
                            is_entry = "ENTRY" in row['tag']
                            marker = "  >>> " if is_entry else "      "
                            print(f"{marker}P{row['point_number']}: Precio {row['price']:.2f} | RSI {row['rsi_value']:.2f} | {row['price_timestamp']}")
                else:
                    print(f"  No se encontraron divergencias en este movimiento")

            print("="*70)
    else:
        print("\n[INFO] No se encontraron divergencias")

    # 5. Generar gráfico con toda la información (usa el df con RSI del paso 2)
    print("\n" + "-"*70)
    print("PASO 5: GENERACIÓN DE GRÁFICO")
    print("-"*70)
    plot_result = plot_range_chart(
        rsi_result['df_with_rsi'],
        fractals_result['df_fractals_minor'],
        fractals_result['df_fractals_major'],
        start_date,
        end_date,
        rsi_levels=rsi_result,
        fibo_levels=fibo_result,
        divergences=divergences
    )
    if plot_result is None:
        print("[ERROR] Fallo en generación de gráfico")
        return None

    # 6. Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Rango analizado: {start_date} -> {end_date}")
    print(f"Registros procesados: {fractals_result['total_records']}")
    print(f"Fractales MINOR detectados: {fractals_result['minor_count']}")
    print(f"Fractales MAJOR detectados: {fractals_result['major_count']}")
    print(f"RSI medio: {rsi_result['rsi_mean']:.2f}")
    print(f"Momentos de sobrecompra: {rsi_result['overbought_count']}")
    print(f"Momentos de sobreventa: {rsi_result['oversold_count']}")
    if fibo_result:
        print(f"Fibonacci - Movimientos alcistas detectados: {fibo_result['total_moves']}")
        if fibo_result['total_moves'] > 0:
            for idx, move in enumerate(fibo_result['upward_moves'], 1):
                print(f"  Movimiento #{idx}: {move['swing_low']:.2f} -> {move['swing_high']:.2f} (Rango: {move['range']:.2f})")
    if not divergences.empty:
        num_unique_divs = divergences['divergence_id'].nunique()
        print(f"Divergencias alcistas detectadas: {num_unique_divs}")
    else:
        print("Divergencias alcistas detectadas: 0")
    print(f"Gráfico generado: {plot_result['output_path']}")
    print("="*70 + "\n")

    return {
        'start_date': start_date,
        'end_date': end_date,
        'fractals': fractals_result,
        'rsi': rsi_result,
        'fibonacci': fibo_result,
        'plot': plot_result
    }


def main_quant(dia: str):
    """
    DEPRECATED: Ejecuta el pipeline completo de análisis cuantitativo para un día
    Esta función se mantiene por compatibilidad pero se recomienda usar main_quant_range()

    Args:
        dia: Fecha en formato YYYY-MM-DD
    """
    from find_fractals import process_fractals
    from analyze_rsi import analyze_rsi_levels
    from analyze_fibonacci import analyze_fibonacci
    from plot_day import plot_day_chart

    print("\n" + "="*70)
    print("ANÁLISIS CUANTITATIVO - Gold (GC)")
    print("="*70)
    print(f"Día: {dia}")
    print("="*70 + "\n")

    # 1. Verificar que existe el archivo CSV
    csv_path = DATA_DIR / f"gc_{dia}.csv"
    if not csv_path.exists():
        print(f"[ERROR] No existe el archivo CSV para {dia}: {csv_path}")
        print("Abortando análisis.")
        return None

    print(f"[OK] Archivo encontrado: {csv_path.name}\n")

    # 2. Procesar fractales
    print("\n" + "-"*70)
    print("PASO 1: DETECCIÓN DE FRACTALES")
    print("-"*70)
    fractals_result = process_fractals(dia)
    if fractals_result is None:
        print("[ERROR] Fallo en detección de fractales")
        return None

    # 3. Analizar niveles RSI
    print("\n" + "-"*70)
    print("PASO 2: ANÁLISIS RSI")
    print("-"*70)
    rsi_result = analyze_rsi_levels(dia)
    if rsi_result is None:
        print("[ERROR] Fallo en análisis RSI")
        return None

    # 4. Analizar Fibonacci
    print("\n" + "-"*70)
    print("PASO 3: ANÁLISIS FIBONACCI")
    print("-"*70)
    fibo_result = analyze_fibonacci(dia)
    if fibo_result is None:
        print("[WARNING] No se pudo calcular Fibonacci, continuando sin él...")
        fibo_result = None

    # 5. Generar gráfico con toda la información
    print("\n" + "-"*70)
    print("PASO 4: GENERACIÓN DE GRÁFICO")
    print("-"*70)
    plot_result = plot_day_chart(dia, rsi_levels=rsi_result, fibo_levels=fibo_result)
    if plot_result is None:
        print("[ERROR] Fallo en generación de gráfico")
        return None

    # 6. Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Día analizado: {dia}")
    print(f"Registros procesados: {fractals_result['total_records']}")
    print(f"Fractales MINOR detectados: {fractals_result['minor_count']}")
    print(f"Fractales MAJOR detectados: {fractals_result['major_count']}")
    print(f"RSI medio: {rsi_result['rsi_mean']:.2f}")
    print(f"Momentos de sobrecompra: {rsi_result['overbought_count']}")
    print(f"Momentos de sobreventa: {rsi_result['oversold_count']}")
    if fibo_result:
        # Legacy function still uses old structure with single range
        if 'range' in fibo_result:
            print(f"Fibonacci - Swing Range: {fibo_result['range']:.2f} ({fibo_result['swing_low']:.2f} - {fibo_result['swing_high']:.2f})")
    print(f"Gráfico generado: {plot_result['output_path']}")
    print("="*70 + "\n")

    return {
        'dia': dia,
        'fractals': fractals_result,
        'rsi': rsi_result,
        'fibonacci': fibo_result,
        'plot': plot_result
    }


if __name__ == "__main__":
    print("\nIniciando análisis cuantitativo...\n")
    result = main_quant_range(START_DATE, END_DATE)

    if result:
        print("[OK] Análisis completado exitosamente\n")
    else:
        print("[ERROR] El análisis finalizó con errores\n")

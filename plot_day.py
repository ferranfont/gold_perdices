import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from config import (
    START_DATE, END_DATE, FRACTALS_DIR,
    RSI_PERIOD, RSI_SMOOTH_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    RSI_RESAMPLE, RSI_RESAMPLE_TO_PERIOD, MINIMUM_IMPULSE_FACTOR,
    MINOR_ZIG_ZAG_RSI, MIN_CHANGE_PCT_RSI
)

def calculate_rsi(df, period=14, smooth_period=0, resample_rule=None):
    """
    Calcula el RSI (Relative Strength Index) con suavizado opcional.

    Si se pasa `resample_rule`, se utiliza un DataFrame temporal resampleado
    solo para el cálculo del RSI y luego se mapea el RSI resultante de vuelta
    a las filas originales (cada fila obtiene el RSI del último bucket previo).
    """
    # Sin resample: cálculo directo
    if not resample_rule:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        if smooth_period > 0:
            rsi = rsi.ewm(span=smooth_period, adjust=False).mean()

        return rsi

    # Con resample: crear DF temporal solo para RSI
    temp = df.copy()
    temp['timestamp_dt'] = pd.to_datetime(temp['timestamp'], errors='coerce', utc=True)
    temp = temp.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')
    if temp.empty:
        return pd.Series([pd.NA] * len(df), index=df.index)

    # Normalizar regla de resample
    if isinstance(resample_rule, str):
        rr = resample_rule.lower()
    else:
        rr = resample_rule

    close_series = temp.set_index('timestamp_dt')['close']
    if not isinstance(close_series.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        close_series.index = pd.to_datetime(close_series.index, errors='coerce', utc=True)
        close_series = close_series.dropna()

    if close_series.empty:
        return pd.Series([pd.NA] * len(df), index=df.index)

    close_resampled = close_series.resample(rr).last().dropna()
    if close_resampled.empty:
        return pd.Series([pd.NA] * len(df), index=df.index)

    delta = close_resampled.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi_res = 100 - (100 / (1 + rs))

    if smooth_period > 0:
        rsi_res = rsi_res.ewm(span=smooth_period, adjust=False).mean()

    rsi_df = rsi_res.rename('rsi_resampled').reset_index()
    rsi_df.columns = ['timestamp_resample', 'rsi_resampled']

    temp_merge = temp[['timestamp_dt']].copy()
    temp_merge = temp_merge.reset_index().rename(columns={'index': 'orig_index'})

    rsi_df = rsi_df.sort_values('timestamp_resample')
    temp_merge = temp_merge.sort_values('timestamp_dt')

    merged = pd.merge_asof(temp_merge, rsi_df,
                           left_on='timestamp_dt', right_on='timestamp_resample',
                           direction='backward')

    rsi_mapped = pd.Series(data=merged['rsi_resampled'].values, index=merged['orig_index'])
    rsi_mapped = rsi_mapped.reindex(df.index)

    return rsi_mapped

def detect_rsi_fractals(df, min_change_pct=0.5):
    """
    Detecta fractales (picos y valles) directamente en la serie del RSI.

    Args:
        df: DataFrame con columnas 'timestamp' y 'rsi'
        min_change_pct: Cambio mínimo en puntos de RSI para considerar un fractal (ej: 0.5 = 0.5 puntos RSI)

    Returns:
        DataFrame con fractales detectados (timestamp, rsi_value, type)
    """
    from find_fractals import UnifiedZigzagDetector, ZigzagDirection

    # Filtrar valores válidos de RSI
    df_valid = df[df['rsi'].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame(columns=['timestamp', 'rsi_value', 'type'])

    # Crear detector de fractales para RSI
    # Usamos el RSI como "precio" y detectamos fractales
    detector = UnifiedZigzagDetector(min_change_pct=min_change_pct)

    fractals = []
    for idx, row in df_valid.iterrows():
        rsi_val = row['rsi']
        timestamp = row['timestamp']

        # Para el zigzag, usamos el RSI como high y low (mismo valor)
        detected_point = detector.add_candle(
            high=rsi_val,
            low=rsi_val,
            index=idx,
            timestamp=timestamp
        )

        if detected_point is not None:
            fractal_type = "PICO" if detected_point.direction == ZigzagDirection.UP else "VALLE"
            fractals.append({
                'timestamp': detected_point.timestamp,
                'rsi_value': detected_point.price,
                'type': fractal_type
            })

    df_fractals = pd.DataFrame(fractals)
    return df_fractals

def plot_range_chart(df, df_fractals_minor, df_fractals_major, start_date, end_date, rsi_levels=None, fibo_levels=None, divergences=None):
    """
    Crea un gráfico con línea de precio y fractales ZigZag para un rango de fechas.

    Args:
        df: DataFrame con datos OHLC (debe incluir columna 'rsi')
        df_fractals_minor: DataFrame con fractales MINOR
        df_fractals_major: DataFrame con fractales MAJOR
        start_date: Fecha inicial en formato YYYY-MM-DD
        end_date: Fecha final en formato YYYY-MM-DD
        rsi_levels: dict con información de niveles RSI (opcional)
        fibo_levels: dict con información de niveles Fibonacci para TODOS los movimientos alcistas (opcional)
        divergences: DataFrame con divergencias detectadas (opcional)

    Returns:
        dict con información del gráfico generado o None si hay error
    """
    print(f"Generando gráfico para rango: {start_date} -> {end_date}")
    print(f"Datos cargados: {len(df)} registros")

    # Verificar que RSI ya esté calculado
    if 'rsi' not in df.columns:
        print("Error: El DataFrame debe incluir la columna 'rsi'")
        return None

    # Crear figura con subplots (precio arriba, RSI abajo)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=('', ''),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}]],
        shared_xaxes=True
    )

    # Línea de precio (close) - gris con transparencia
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1),
        opacity=0.5
    ), row=1, col=1)

    # Añadir líneas ZigZag y marcadores de fractales
    if df_fractals_minor is not None and not df_fractals_minor.empty:
        # Línea ZigZag MINOR - dodgerblue
        fig.add_trace(go.Scatter(
            x=df_fractals_minor['timestamp'],
            y=df_fractals_minor['price'],
            mode='lines',
            name='ZigZag Minor',
            line=dict(color='dodgerblue', width=1),
            opacity=0.7,
            hovertemplate='<b>Minor</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Puntos MINOR - dodgerblue pequeños
        fig.add_trace(go.Scatter(
            x=df_fractals_minor['timestamp'],
            y=df_fractals_minor['price'],
            mode='markers',
            name='Fractales Minor',
            marker=dict(
                color='cornflowerblue',
                size=3,
                symbol='circle'
            ),
            opacity=1,
            hovertemplate='<b>Minor</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    if df_fractals_major is not None and not df_fractals_major.empty:
        # Línea ZigZag MAJOR - AZUL
        fig.add_trace(go.Scatter(
            x=df_fractals_major['timestamp'],
            y=df_fractals_major['price'],
            mode='lines',
            name='ZigZag Major',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Separar picos y valles MAJOR para los marcadores
        df_picos_major = df_fractals_major[df_fractals_major['type'] == 'PICO'].copy()
        df_valles_major = df_fractals_major[df_fractals_major['type'] == 'VALLE'].copy()

        # PICOS - círculos verdes rellenos
        if not df_picos_major.empty:
            fig.add_trace(go.Scatter(
                x=df_picos_major['timestamp'],
                y=df_picos_major['price'],
                mode='markers',
                name='PICO Major',
                marker=dict(
                    color='green',
                    size=5,
                    symbol='circle'
                ),
                hovertemplate='<b>PICO Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

        # VALLES - círculos rojos rellenos
        if not df_valles_major.empty:
            fig.add_trace(go.Scatter(
                x=df_valles_major['timestamp'],
                y=df_valles_major['price'],
                mode='markers',
                name='VALLE Major',
                marker=dict(
                    color='red',
                    size=5,
                    symbol='circle'
                ),
                hovertemplate='<b>VALLE Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # Dibujar triángulos de divergencia (señales de entrada)
    if divergences is not None and not divergences.empty:
        # Filtrar solo puntos de ENTRY (el último punto de cada divergencia)
        df_entries = divergences[divergences['tag'].str.contains('ENTRY', na=False)].copy()

        if not df_entries.empty:
            print(f"[INFO] Dibujando {len(df_entries)} señales de divergencia en el gráfico")

            fig.add_trace(go.Scatter(
                x=df_entries['price_timestamp'],
                y=df_entries['price'],
                mode='markers',
                name='Divergencia ENTRY',
                marker=dict(
                    color='lime',
                    size=12,
                    symbol='triangle-up',
                    line=dict(color='darkgreen', width=2)
                ),
                hovertemplate='<b>DIVERGENCIA ENTRY</b><br>Tag: %{text}<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>',
                text=df_entries['tag']
            ), row=1, col=1)

    # Añadir niveles Fibonacci para TODOS los movimientos alcistas
    if fibo_levels is not None and 'upward_moves' in fibo_levels:
        upward_moves = fibo_levels['upward_moves']
        print(f"Dibujando niveles Fibonacci para {len(upward_moves)} movimientos alcistas")

        for move_idx, move in enumerate(upward_moves):
            move_levels = move['levels']
            swing_high_ts = move['swing_high_timestamp']
            retracement_end_ts = move.get('retracement_end_timestamp')

            # Si no hay siguiente fractal, dibujamos hasta el final del dataframe
            if retracement_end_ts is None:
                # Usar el último timestamp disponible en el dataframe
                end_ts = df['timestamp'].iloc[-1]
            else:
                end_ts = retracement_end_ts

            for level, price in move_levels.items():
                # Solo dibujar niveles clave: 38.2%, 50%, 61.8%
                if level in [0.382, 0.5, 0.618]:
                    # Color diferente por nivel
                    if level == 0.382:
                        color = 'orange'
                    elif level == 0.5:
                        color = 'darkorange'
                    else:  # 0.618
                        color = 'orangered'

                    # Dibujar línea diagonal desde swing_high_ts hasta retracement_end_ts
                    fig.add_trace(go.Scatter(
                        x=[swing_high_ts, end_ts],
                        y=[price, price],
                        mode='lines',
                        name=f'Fibo {level*100:.1f}%',
                        line=dict(color=color, width=1.5, dash='solid'),
                        showlegend=False,
                        hovertemplate=f'Fibo {level*100:.1f}%<br>Price: {price:.2f}<extra></extra>'
                    ), row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rsi'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=1),
        showlegend=False
    ), row=2, col=1)

    # Detectar fractales directamente en el RSI y marcarlos con puntos violeta
    # Solo mostrar VALLES (fractal low) que están por debajo de RSI_OVERSOLD
    df_rsi_fractals = detect_rsi_fractals(df, min_change_pct=MIN_CHANGE_PCT_RSI)
    if not df_rsi_fractals.empty:
        # Filtrar solo fractales VALLE por debajo de RSI_OVERSOLD
        df_rsi_valles = df_rsi_fractals[
            (df_rsi_fractals['type'] == 'VALLE') &
            (df_rsi_fractals['rsi_value'] < RSI_OVERSOLD)
        ]

        # Dibujar VALLES del RSI en violeta (solo si están por debajo de OVERSOLD)
        if not df_rsi_valles.empty:
            fig.add_trace(go.Scatter(
                x=df_rsi_valles['timestamp'],
                y=df_rsi_valles['rsi_value'],
                mode='markers',
                name='RSI Valles <OS',
                marker=dict(
                    color='violet',
                    size=8,
                    symbol='circle'
                ),
                hovertemplate='<b>RSI VALLE <OS</b><br>Time: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ), row=2, col=1)

    # Dibujar triángulos de divergencia en el RSI (señales de entrada)
    if divergences is not None and not divergences.empty:
        # Filtrar solo puntos de ENTRY
        df_entries = divergences[divergences['tag'].str.contains('ENTRY', na=False)].copy()

        if not df_entries.empty:
            # Para cada entrada, encontrar el valor del RSI en ese timestamp
            rsi_entry_values = []
            rsi_entry_timestamps = []

            for idx, row in df_entries.iterrows():
                entry_ts = pd.to_datetime(row['rsi_timestamp'])
                rsi_val = row['rsi_value']

                if pd.notna(rsi_val):
                    rsi_entry_values.append(rsi_val)
                    rsi_entry_timestamps.append(entry_ts)

            # Dibujar triángulos en el RSI
            if rsi_entry_timestamps:
                fig.add_trace(go.Scatter(
                    x=rsi_entry_timestamps,
                    y=rsi_entry_values,
                    mode='markers',
                    name='Divergencia ENTRY RSI',
                    marker=dict(
                        color='lime',
                        size=12,
                        symbol='triangle-up',
                        line=dict(color='darkgreen', width=2)
                    ),
                    hovertemplate='<b>DIVERGENCIA ENTRY RSI</b><br>Time: %{x}<br>RSI: %{y:.2f}<extra></extra>'
                ), row=2, col=1)

    # CÓDIGO COMENTADO - Marcar fractales VALLE de precio con RSI < RSI_OVERSOLD
    # (No se usa actualmente, pero se mantiene por si se necesita en el futuro)
    """
    df_fractals_to_use = df_fractals_minor if MINOR_ZIG_ZAG_RSI else df_fractals_major
    fractal_type_label = "MINOR" if MINOR_ZIG_ZAG_RSI else "MAJOR"

    if df_fractals_to_use is not None and not df_fractals_to_use.empty:
        df_valles_rsi = df_fractals_to_use[df_fractals_to_use['type'] == 'VALLE'].copy()

        if not df_valles_rsi.empty:
            # Para cada fractal VALLE, encontrar el valor del RSI en ese timestamp
            rsi_values = []
            timestamps_match = []

            for idx, row in df_valles_rsi.iterrows():
                fractal_ts = row['timestamp']
                # Buscar el valor del RSI en el df principal para ese timestamp
                df_match = df[df['timestamp'] == fractal_ts]
                if not df_match.empty:
                    rsi_val = df_match['rsi'].iloc[0]
                    # Solo guardar si RSI < RSI_OVERSOLD
                    if pd.notna(rsi_val) and rsi_val < RSI_OVERSOLD:
                        rsi_values.append(rsi_val)
                        timestamps_match.append(fractal_ts)

            # Dibujar puntos rojos en el RSI para esos fractales
            if timestamps_match:
                fig.add_trace(go.Scatter(
                    x=timestamps_match,
                    y=rsi_values,
                    mode='markers',
                    name=f'VALLE {fractal_type_label} RSI<OS',
                    marker=dict(
                        color='red',
                        size=6,
                        symbol='circle'
                    ),
                    hovertemplate=f'<b>VALLE {fractal_type_label} RSI<OS</b><br>Time: %{{x}}<br>RSI: %{{y:.2f}}<extra></extra>'
                ), row=2, col=1)
    """

    # Líneas de sobrecompra y sobreventa
    fig.add_hline(y=RSI_OVERBOUGHT, line=dict(color='lightblue', width=1, dash='solid'),
                  row=2, col=1, annotation_text=str(RSI_OVERBOUGHT),
                  annotation_position="right")
    fig.add_hline(y=RSI_OVERSOLD, line=dict(color='lightblue', width=1, dash='solid'),
                  row=2, col=1, annotation_text=str(RSI_OVERSOLD),
                  annotation_position="right")
    fig.add_hline(y=50, line=dict(color='lightblue', width=0.5, dash='dot'),
                  row=2, col=1)

    # Configurar layout: añadir información de parámetros al título
    # Determinar regla efectiva de resample según configuración
    if isinstance(RSI_RESAMPLE, bool) and RSI_RESAMPLE is True:
        effective_resample = RSI_RESAMPLE_TO_PERIOD
    elif isinstance(RSI_RESAMPLE, str) and RSI_RESAMPLE:
        effective_resample = RSI_RESAMPLE
    else:
        effective_resample = None

    title_params = (
        f"RSI_p={RSI_PERIOD}, smooth={RSI_SMOOTH_PERIOD}, resample={effective_resample}, "
        f"RSI_OB/OS={RSI_OVERBOUGHT}/{RSI_OVERSOLD}, MIN_IMP={MINIMUM_IMPULSE_FACTOR}%"
    )

    fig.update_layout(
        title=f'GL {start_date} -> {end_date} | {title_params}',
        template='plotly_white',
        hovermode=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family='Arial',
            size=12,
            color='#333333'
        ),
        height=940,
        showlegend=True
    )

    # Configurar eje X del subplot de precio (row 1)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        row=1, col=1
    )

    # Configurar eje Y del subplot de precio (row 1)
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#f0f0f0',
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        side='left',
        row=1, col=1
    )

    # Configurar eje X del subplot RSI (row 2)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        row=2, col=1
    )

    # Configurar eje Y del subplot RSI (row 2)
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        side='left',
        range=[0, 100],
        row=2, col=1
    )

    # Crear carpeta de salida si no existe
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Guardar gráfico
    date_range_str = f"{start_date}_{end_date}"
    output_html = os.path.join(output_dir, f'gc_{date_range_str}.html')
    print(f"Guardando gráfico en: {output_html}")
    fig.write_html(output_html)
    print(f"Gráfico guardado exitosamente")

    # Mostrar en navegador
    import webbrowser
    webbrowser.open(f'file://{os.path.abspath(output_html)}')
    print(f"Abriendo en navegador...")

    return {
        'start_date': start_date,
        'end_date': end_date,
        'output_path': output_html,
        'total_records': len(df),
        'rsi_calculated': True
    }

def plot_day_chart(dia, rsi_levels=None, fibo_levels=None, divergences=None):
    """
    Crea un gráfico con línea de precio y fractales ZigZag para el día especificado.

    Args:
        dia: Fecha en formato YYYY-MM-DD
        rsi_levels: dict con información de niveles RSI (opcional)
        fibo_levels: dict con información de niveles Fibonacci (opcional)
        divergences: DataFrame con divergencias detectadas (opcional)

    Returns:
        dict con información del gráfico generado o None si hay error
    """
    # Ruta al archivo CSV
    csv_path = f"data/gc_{dia}.csv"

    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo {csv_path}")
        return None

    # Leer datos de precio
    print(f"Cargando datos de: {csv_path}")
    df = pd.read_csv(csv_path)

    # Verificar columnas requeridas
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Faltan columnas: {missing_cols}")
        return None

    print(f"Datos cargados: {len(df)} registros")

    # Determinar regla efectiva de resample según configuración
    if isinstance(RSI_RESAMPLE, bool) and RSI_RESAMPLE is True:
        effective_resample = RSI_RESAMPLE_TO_PERIOD
        print(f"[INFO] RSI resample enabled via flag; using RSI_RESAMPLE_TO_PERIOD={effective_resample}")
    elif isinstance(RSI_RESAMPLE, str) and RSI_RESAMPLE:
        effective_resample = RSI_RESAMPLE
        print(f"[INFO] RSI resample rule from RSI_RESAMPLE: {effective_resample}")
    else:
        effective_resample = None
        print(f"[INFO] RSI resample disabled")

    df['rsi'] = calculate_rsi(df, period=RSI_PERIOD, smooth_period=RSI_SMOOTH_PERIOD, resample_rule=effective_resample)
    print(f"RSI calculado con período {RSI_PERIOD} y suavizado EMA {RSI_SMOOTH_PERIOD}")

    # Cargar fractales si existen
    fractal_minor_path = FRACTALS_DIR / f"gc_fractals_minor_{dia}.csv"
    fractal_major_path = FRACTALS_DIR / f"gc_fractals_major_{dia}.csv"

    df_fractals_minor = None
    df_fractals_major = None

    if fractal_minor_path.exists():
        df_fractals_minor = pd.read_csv(fractal_minor_path)
        print(f"Fractales MINOR cargados: {len(df_fractals_minor)}")
    else:
        print(f"No se encontraron fractales MINOR para {dia}")

    if fractal_major_path.exists():
        df_fractals_major = pd.read_csv(fractal_major_path)
        print(f"Fractales MAJOR cargados: {len(df_fractals_major)}")
    else:
        print(f"No se encontraron fractales MAJOR para {dia}")

    # Crear figura con subplots (precio arriba, RSI abajo)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=('', ''),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}]],
        shared_xaxes=True
    )

    # Línea de precio (close) - gris con transparencia
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='gray', width=1),
        opacity=0.5
    ), row=1, col=1)

    # Añadir líneas ZigZag y marcadores de fractales
    if df_fractals_minor is not None and not df_fractals_minor.empty:
        # Línea ZigZag MINOR - dodgerblue
        fig.add_trace(go.Scatter(
            x=df_fractals_minor['timestamp'],
            y=df_fractals_minor['price'],
            mode='lines',
            name='ZigZag Minor',
            line=dict(color='dodgerblue', width=1),
            opacity=0.7,
            hovertemplate='<b>Minor</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Puntos MINOR - dodgerblue pequeños
        fig.add_trace(go.Scatter(
            x=df_fractals_minor['timestamp'],
            y=df_fractals_minor['price'],
            mode='markers',
            name='Fractales Minor',
            marker=dict(
                color='cornflowerblue',
                size=3,
                symbol='circle'
            ),
            opacity=1,
            hovertemplate='<b>Minor</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    if df_fractals_major is not None and not df_fractals_major.empty:
        # Línea ZigZag MAJOR - AZUL
        fig.add_trace(go.Scatter(
            x=df_fractals_major['timestamp'],
            y=df_fractals_major['price'],
            mode='lines',
            name='ZigZag Major',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Separar picos y valles MAJOR para los marcadores
        df_picos_major = df_fractals_major[df_fractals_major['type'] == 'PICO'].copy()
        df_valles_major = df_fractals_major[df_fractals_major['type'] == 'VALLE'].copy()

        # PICOS - círculos verdes rellenos
        if not df_picos_major.empty:
            fig.add_trace(go.Scatter(
                x=df_picos_major['timestamp'],
                y=df_picos_major['price'],
                mode='markers',
                name='PICO Major',
                marker=dict(
                    color='green',
                    size=5,
                    symbol='circle'
                ),
                hovertemplate='<b>PICO Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

        # VALLES - círculos rojos rellenos
        if not df_valles_major.empty:
            fig.add_trace(go.Scatter(
                x=df_valles_major['timestamp'],
                y=df_valles_major['price'],
                mode='markers',
                name='VALLE Major',
                marker=dict(
                    color='red',
                    size=5,
                    symbol='circle'
                ),
                hovertemplate='<b>VALLE Major</b><br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # Dibujar triángulos de divergencia (señales de entrada)
    if divergences is not None and not divergences.empty:
        # Filtrar solo puntos de ENTRY (el último punto de cada divergencia)
        df_entries = divergences[divergences['tag'].str.contains('ENTRY', na=False)].copy()

        if not df_entries.empty:
            print(f"[INFO] Dibujando {len(df_entries)} señales de divergencia en el gráfico")

            fig.add_trace(go.Scatter(
                x=df_entries['price_timestamp'],
                y=df_entries['price'],
                mode='markers',
                name='Divergencia ENTRY',
                marker=dict(
                    color='lime',
                    size=12,
                    symbol='triangle-up',
                    line=dict(color='darkgreen', width=2)
                ),
                hovertemplate='<b>DIVERGENCIA ENTRY</b><br>Tag: %{text}<br>Time: %{x}<br>Price: %{y:.2f}<extra></extra>',
                text=df_entries['tag']
            ), row=1, col=1)

    # Añadir niveles Fibonacci para TODOS los movimientos alcistas
    if fibo_levels is not None and 'upward_moves' in fibo_levels:
        upward_moves = fibo_levels['upward_moves']
        print(f"Dibujando niveles Fibonacci para {len(upward_moves)} movimientos alcistas")

        for move_idx, move in enumerate(upward_moves):
            move_levels = move['levels']
            swing_high_ts = move['swing_high_timestamp']
            retracement_end_ts = move.get('retracement_end_timestamp')

            # Si no hay siguiente fractal, dibujamos hasta el final del dataframe
            if retracement_end_ts is None:
                # Usar el último timestamp disponible en el dataframe
                end_ts = df['timestamp'].iloc[-1]
            else:
                end_ts = retracement_end_ts

            for level, price in move_levels.items():
                # Solo dibujar niveles clave: 38.2%, 50%, 61.8%
                if level in [0.382, 0.5, 0.618]:
                    # Color diferente por nivel
                    if level == 0.382:
                        color = 'orange'
                    elif level == 0.5:
                        color = 'darkorange'
                    else:  # 0.618
                        color = 'orangered'

                    # Dibujar línea diagonal desde swing_high_ts hasta retracement_end_ts
                    fig.add_trace(go.Scatter(
                        x=[swing_high_ts, end_ts],
                        y=[price, price],
                        mode='lines',
                        name=f'Fibo {level*100:.1f}%',
                        line=dict(color=color, width=1.5, dash='solid'),
                        showlegend=False,
                        hovertemplate=f'Fibo {level*100:.1f}%<br>Price: {price:.2f}<extra></extra>'
                    ), row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rsi'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=1),
        showlegend=False
    ), row=2, col=1)

    # Detectar fractales directamente en el RSI y marcarlos con puntos violeta
    # Solo mostrar VALLES (fractal low) que están por debajo de RSI_OVERSOLD
    df_rsi_fractals = detect_rsi_fractals(df, min_change_pct=MIN_CHANGE_PCT_RSI)
    if not df_rsi_fractals.empty:
        # Filtrar solo fractales VALLE por debajo de RSI_OVERSOLD
        df_rsi_valles = df_rsi_fractals[
            (df_rsi_fractals['type'] == 'VALLE') &
            (df_rsi_fractals['rsi_value'] < RSI_OVERSOLD)
        ]

        # Dibujar VALLES del RSI en violeta (solo si están por debajo de OVERSOLD)
        if not df_rsi_valles.empty:
            fig.add_trace(go.Scatter(
                x=df_rsi_valles['timestamp'],
                y=df_rsi_valles['rsi_value'],
                mode='markers',
                name='RSI Valles <OS',
                marker=dict(
                    color='violet',
                    size=8,
                    symbol='circle'
                ),
                hovertemplate='<b>RSI VALLE <OS</b><br>Time: %{x}<br>RSI: %{y:.2f}<extra></extra>'
            ), row=2, col=1)

    # Dibujar triángulos de divergencia en el RSI (señales de entrada)
    if divergences is not None and not divergences.empty:
        # Filtrar solo puntos de ENTRY
        df_entries = divergences[divergences['tag'].str.contains('ENTRY', na=False)].copy()

        if not df_entries.empty:
            # Para cada entrada, encontrar el valor del RSI en ese timestamp
            rsi_entry_values = []
            rsi_entry_timestamps = []

            for idx, row in df_entries.iterrows():
                entry_ts = pd.to_datetime(row['rsi_timestamp'])
                rsi_val = row['rsi_value']

                if pd.notna(rsi_val):
                    rsi_entry_values.append(rsi_val)
                    rsi_entry_timestamps.append(entry_ts)

            # Dibujar triángulos en el RSI
            if rsi_entry_timestamps:
                fig.add_trace(go.Scatter(
                    x=rsi_entry_timestamps,
                    y=rsi_entry_values,
                    mode='markers',
                    name='Divergencia ENTRY RSI',
                    marker=dict(
                        color='lime',
                        size=12,
                        symbol='triangle-up',
                        line=dict(color='darkgreen', width=2)
                    ),
                    hovertemplate='<b>DIVERGENCIA ENTRY RSI</b><br>Time: %{x}<br>RSI: %{y:.2f}<extra></extra>'
                ), row=2, col=1)

    # CÓDIGO COMENTADO - Marcar fractales VALLE de precio con RSI < RSI_OVERSOLD
    # (No se usa actualmente, pero se mantiene por si se necesita en el futuro)
    """
    df_fractals_to_use = df_fractals_minor if MINOR_ZIG_ZAG_RSI else df_fractals_major
    fractal_type_label = "MINOR" if MINOR_ZIG_ZAG_RSI else "MAJOR"

    if df_fractals_to_use is not None and not df_fractals_to_use.empty:
        df_valles_rsi = df_fractals_to_use[df_fractals_to_use['type'] == 'VALLE'].copy()

        if not df_valles_rsi.empty:
            # Para cada fractal VALLE, encontrar el valor del RSI en ese timestamp
            rsi_values = []
            timestamps_match = []

            for idx, row in df_valles_rsi.iterrows():
                fractal_ts = row['timestamp']
                # Buscar el valor del RSI en el df principal para ese timestamp
                df_match = df[df['timestamp'] == fractal_ts]
                if not df_match.empty:
                    rsi_val = df_match['rsi'].iloc[0]
                    # Solo guardar si RSI < RSI_OVERSOLD
                    if pd.notna(rsi_val) and rsi_val < RSI_OVERSOLD:
                        rsi_values.append(rsi_val)
                        timestamps_match.append(fractal_ts)

            # Dibujar puntos rojos en el RSI para esos fractales
            if timestamps_match:
                fig.add_trace(go.Scatter(
                    x=timestamps_match,
                    y=rsi_values,
                    mode='markers',
                    name=f'VALLE {fractal_type_label} RSI<OS',
                    marker=dict(
                        color='red',
                        size=6,
                        symbol='circle'
                    ),
                    hovertemplate=f'<b>VALLE {fractal_type_label} RSI<OS</b><br>Time: %{{x}}<br>RSI: %{{y:.2f}}<extra></extra>'
                ), row=2, col=1)
    """

    # Líneas de sobrecompra y sobreventa
    fig.add_hline(y=RSI_OVERBOUGHT, line=dict(color='lightblue', width=1, dash='solid'),
                  row=2, col=1, annotation_text=str(RSI_OVERBOUGHT),
                  annotation_position="right")
    fig.add_hline(y=RSI_OVERSOLD, line=dict(color='lightblue', width=1, dash='solid'),
                  row=2, col=1, annotation_text=str(RSI_OVERSOLD),
                  annotation_position="right")
    fig.add_hline(y=50, line=dict(color='lightblue', width=0.5, dash='dot'),
                  row=2, col=1)

    # Añadir parámetros al título del gráfico día
    # Determinar regla efectiva de resample (misma lógica que en la sección de cálculo RSI)
    if isinstance(RSI_RESAMPLE, bool) and RSI_RESAMPLE is True:
        effective_resample_day = RSI_RESAMPLE_TO_PERIOD
    elif isinstance(RSI_RESAMPLE, str) and RSI_RESAMPLE:
        effective_resample_day = RSI_RESAMPLE
    else:
        effective_resample_day = None

    title_params_day = (
        f"RSI_p={RSI_PERIOD}, smooth={RSI_SMOOTH_PERIOD}, resample={effective_resample_day}, "
        f"RSI_OB/OS={RSI_OVERBOUGHT}/{RSI_OVERSOLD}, MIN_IMP={MINIMUM_IMPULSE_FACTOR}%"
    )

    fig.update_layout(
        title=f'GL {dia} | {title_params_day}',
        template='plotly_white',
        hovermode=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family='Arial',
            size=12,
            color='#333333'
        ),
        height=940,
        showlegend=True
    )

    # Configurar eje X del subplot de precio (row 1)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        row=1, col=1
    )

    # Configurar eje Y del subplot de precio (row 1)
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#f0f0f0',
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        side='left',
        row=1, col=1
    )

    # Configurar eje X del subplot RSI (row 2)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        row=2, col=1
    )

    # Configurar eje Y del subplot RSI (row 2)
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='gray',
        tickcolor='gray',
        tickfont=dict(color='gray'),
        side='left',
        range=[0, 100],
        row=2, col=1
    )

    # Crear carpeta de salida si no existe
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Guardar gráfico
    output_html = os.path.join(output_dir, f'gc_{dia}.html')
    print(f"Guardando gráfico en: {output_html}")
    fig.write_html(output_html)
    print(f"Gráfico guardado exitosamente")

    # Mostrar en navegador
    import webbrowser
    webbrowser.open(f'file://{os.path.abspath(output_html)}')
    print(f"Abriendo en navegador...")

    return {
        'dia': dia,
        'output_path': output_html,
        'total_records': len(df),
        'rsi_calculated': True
    }

if __name__ == "__main__":
    from find_fractals import load_date_range
    from analyze_rsi import calculate_rsi

    print(f"Generando gráfico para: {START_DATE} -> {END_DATE}")

    # Cargar datos del rango
    df = load_date_range(START_DATE, END_DATE)
    if df is None:
        print("Error cargando datos")
        exit(1)

    # Calcular RSI con suavizado (posible resample definido en config)
    print(f"[INFO] RSI resample rule: {rsi_resample}")
    df['rsi'] = calculate_rsi(df, period=RSI_PERIOD, smooth_period=RSI_SMOOTH_PERIOD, resample_rule=rsi_resample)

    # Cargar fractales
    date_range_str = f"{START_DATE}_{END_DATE}"
    fractal_minor_path = FRACTALS_DIR / f"gc_fractals_minor_{date_range_str}.csv"
    fractal_major_path = FRACTALS_DIR / f"gc_fractals_major_{date_range_str}.csv"

    df_fractals_minor = None
    df_fractals_major = None

    if fractal_minor_path.exists():
        df_fractals_minor = pd.read_csv(fractal_minor_path)
    if fractal_major_path.exists():
        df_fractals_major = pd.read_csv(fractal_major_path)

    plot_range_chart(df, df_fractals_minor, df_fractals_major, START_DATE, END_DATE)

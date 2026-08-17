"""
Junta los CSVs de movement_logs/<video>/ de varios videos ya procesados
(cada video tiene su propia carpeta) y genera graficas comparativas
(velocidad, excentricidad, area por intervalo de frames, en cm).

Los histogramas de excentricidad por video individual se generan en
app.py durante el procesamiento de cada video, no aqui.

Uso:
    python compare_videos.py
    (toma automaticamente todos los videos que tengan <video>_movement.csv
     y <video>_area.csv dentro de movement_logs/<video>/)
"""
import glob
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG_COLOR = '#1a1a2e'
TEXT_COLOR = 'white'
MINUTOS_POR_FRAME = 5  # cada frame representa 5 minutos reales


def apply_dark_style(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')


def frames_a_horas(frames):
    return frames * MINUTOS_POR_FRAME / 60


def calcular_velocidad(df_movement):
    resultados = []
    for tid, grupo in df_movement.groupby('track_id'):
        grupo = grupo.sort_values('frame').reset_index(drop=True)
        for i in range(1, len(grupo)):
            dx_diff = grupo.loc[i, 'dx_cm'] - grupo.loc[i - 1, 'dx_cm']
            dy_diff = grupo.loc[i, 'dy_cm'] - grupo.loc[i - 1, 'dy_cm']
            vel = math.sqrt(dx_diff ** 2 + dy_diff ** 2)
            resultados.append({'track_id': tid, 'frame': grupo.loc[i, 'frame'], 'velocidad': vel})
    return pd.DataFrame(resultados)


def calcular_excentricidad(df_area):
    df = df_area.copy()
    df['excentricidad'] = df['w_cm'] / df['h_cm'].replace(0, np.nan)
    return df


def cargar_video(video_name, log_dir='movement_logs'):
    carpeta = os.path.join(log_dir, video_name, 'csv')
    df_mov = pd.read_csv(os.path.join(carpeta, f'{video_name}_movement.csv'))
    df_area = pd.read_csv(os.path.join(carpeta, f'{video_name}_area.csv'))
    return df_mov, df_area


def serie_por_intervalo(df_mov, df_area, intervalo=10):
    """Promedia velocidad/excentricidad/area entre TODOS los physarums
    detectados en cada ventana de frames, para tener una serie de
    'comportamiento global del video' a lo largo del tiempo."""
    df_vel = calcular_velocidad(df_mov)
    df_exc = calcular_excentricidad(df_area)

    frame_max = max(df_mov['frame'].max(), df_area['frame'].max())
    filas = []
    for inicio in range(0, int(frame_max) + 1, intervalo):
        fin = inicio + intervalo
        vel_ventana = df_vel[(df_vel['frame'] >= inicio) & (df_vel['frame'] < fin)]
        exc_ventana = df_exc[(df_exc['frame'] >= inicio) & (df_exc['frame'] < fin)]
        area_ventana = df_area[(df_area['frame'] >= inicio) & (df_area['frame'] < fin)]

        filas.append({
            'intervalo': inicio,
            'velocidad_prom': vel_ventana['velocidad'].mean() if not vel_ventana.empty else np.nan,
            'excentricidad_prom': exc_ventana['excentricidad'].mean() if not exc_ventana.empty else np.nan,
            'area_prom': area_ventana['area_cm2'].mean() if not area_ventana.empty else np.nan,
        })
    return pd.DataFrame(filas)


def calcular_continuidad(df_mov, video_name):
    """Para cada physarum, que tan 'constante' fue su deteccion: que
    fraccion de su propio rango de vida (desde que aparece hasta que se
    pierde por ultima vez) realmente tiene frames detectados, y cual fue
    el hueco mas largo que tuvo. Cobertura cercana a 1 = casi sin huecos."""
    filas = []
    for tid, grupo in df_mov.groupby('track_id'):
        frames = sorted(grupo['frame'].unique())
        frame_inicio = frames[0]
        frame_final = frames[-1]
        duracion = frame_final - frame_inicio + 1
        frames_totales = len(frames)
        cobertura = frames_totales / duracion if duracion > 0 else 1.0

        hueco_max = 0
        for i in range(1, len(frames)):
            hueco = frames[i] - frames[i - 1] - 1
            hueco_max = max(hueco_max, hueco)

        filas.append({
            'video': video_name,
            'track_id': tid,
            'frame_inicio': frame_inicio,
            'frame_final': frame_final,
            'duracion_frames': duracion,
            'frames_detectados': frames_totales,
            'cobertura': round(cobertura, 3),
            'hueco_max_frames': hueco_max,
        })
    return pd.DataFrame(filas)


def generar_comparativas(video_names, intervalo=10, log_dir='movement_logs', out_dir=None):
    out_dir = out_dir or os.path.join(log_dir, 'comparativas')
    os.makedirs(out_dir, exist_ok=True)

    series = {}
    continuidad_filas = []

    for video in video_names:
        df_mov, df_area = cargar_video(video, log_dir)
        series[video] = serie_por_intervalo(df_mov, df_area, intervalo)
        continuidad_filas.append(calcular_continuidad(df_mov, video))

    df_continuidad = pd.concat(continuidad_filas, ignore_index=True)
    df_continuidad = df_continuidad.sort_values(['video', 'cobertura'], ascending=[True, False])
    df_continuidad.to_csv(os.path.join(out_dir, 'continuidad_physarums.csv'), index=False, encoding='utf-8')

    filas_csv = [
        {'video': video, **fila.to_dict()}
        for video, df_serie in series.items()
        for _, fila in df_serie.iterrows()
    ]
    df_combinado = pd.DataFrame(filas_csv)
    df_combinado.to_csv(os.path.join(out_dir, 'comparativa_intervalos.csv'), index=False, encoding='utf-8')

    # El promedio global solo tiene sentido mientras TODOS los videos
    # siguen teniendo datos. Una vez el video mas corto termina, seguir
    # promediando mezclaria cada vez menos videos sin avisarlo.
    n_videos = len(series)
    conteo_por_intervalo = df_combinado.groupby('intervalo').size()
    intervalos_completos = conteo_por_intervalo[conteo_por_intervalo == n_videos].index

    df_promedio_global = (
        df_combinado[df_combinado['intervalo'].isin(intervalos_completos)]
        .groupby('intervalo')[['velocidad_prom', 'excentricidad_prom', 'area_prom']]
        .mean()
        .reset_index()
    )
    df_promedio_global.to_csv(os.path.join(out_dir, 'comparativa_promedio_global.csv'), index=False, encoding='utf-8')

    def _grafica_metrica(columna, titulo, ylabel, nombre_archivo):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG_COLOR)
        apply_dark_style(ax)
        for video, df_serie in series.items():
            horas = df_serie['intervalo'].apply(frames_a_horas)
            ax.plot(horas, df_serie[columna], linewidth=1.5, alpha=0.8, label=video)
        horas_prom = df_promedio_global['intervalo'].apply(frames_a_horas)
        ax.plot(horas_prom, df_promedio_global[columna], linewidth=2.5, color='white',
                linestyle='--', label='Promedio global')
        ax.set_title(titulo)
        ax.set_xlabel('Tiempo (horas)')
        ax.set_ylabel(ylabel)
        ax.legend(facecolor='#2a2a4e', labelcolor=TEXT_COLOR, fontsize=8)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, nombre_archivo), dpi=150)
        plt.close()

    _grafica_metrica('velocidad_prom', 'Velocidad promedio por intervalo (todos los videos)',
                      'Velocidad (cm/frame)', 'comparativa_velocidad.png')
    _grafica_metrica('excentricidad_prom', 'Excentricidad promedio por intervalo (todos los videos)',
                      'Excentricidad', 'comparativa_excentricidad.png')
    _grafica_metrica('area_prom', 'Area promedio por intervalo (todos los videos)',
                      'Area (cm2)', 'comparativa_area.png')

    print(f'Comparativas guardadas en: {out_dir}')
    print('Top physarums mas constantes por video (mayor cobertura):')
    for video in video_names:
        top = df_continuidad[df_continuidad['video'] == video].head(3)
        print(f'  {video}: {list(zip(top["track_id"], top["cobertura"]))}')
    return out_dir


if __name__ == '__main__':
    LOG_DIR = 'movement_logs'
    archivos = glob.glob(os.path.join(LOG_DIR, '*', 'csv', '*_movement.csv'))
    video_names = sorted({os.path.basename(f).replace('_movement.csv', '') for f in archivos})
    print('Videos detectados:', video_names)
    generar_comparativas(video_names, intervalo=10, log_dir=LOG_DIR)

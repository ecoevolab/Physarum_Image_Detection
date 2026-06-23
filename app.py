import matplotlib
matplotlib.use('Agg')

import os
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, request,
    redirect, url_for, send_from_directory
)
from ultralytics import YOLO
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. Configuracion
# =============================================================================
app = Flask(__name__)
app.secret_key = '777'

model = YOLO("yolo11_custom_4.pt")
names = model.model.names

BG_COLOR   = '#1a1a2e'
TEXT_COLOR = 'white'

# Umbral de re-identificacion por distancia (para IDs que desaparecieron)
REID_THRESHOLD = 100
# Porcentaje minimo del box nuevo que debe estar dentro del padre para ser division
SPLIT_OVERLAP_RATIO = 0.35 
# Grace period antes de eliminar un ID perdido
GRACE_PERIOD = 15
MIN_PERSISTENCE = 25  # Frames minimos para empezar a registrar movimiento (evita ruido inicial)

# Altura promedio del area capturada por la camara, usada para convertir
# medidas de pixeles a centimetros. Varia un poco video a video, pero
# este promedio es suficientemente bueno para el analisis.
ALTURA_CAPTURA_CM = 28

# =============================================================================
# 2. Utilidades
# =============================================================================

def apply_dark_style(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

MINUTOS_POR_FRAME = 5  # cada frame representa 5 minutos reales
def frames_a_horas(frames, fps=None):
    """Convierte frames a horas reales considerando que cada frame = 5 minutos."""
    return frames * MINUTOS_POR_FRAME / 60

def classify_direction(dx, dy, threshold=10):
    dist = math.sqrt(dx**2 + dy**2)
    if dist < threshold:
        return "Estatico", dist
    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5:                return "Derecha", dist
    elif 22.5 <= angle < 67.5:               return "Arriba-Der", dist
    elif 67.5 <= angle < 112.5:              return "Arriba", dist
    elif 112.5 <= angle < 157.5:             return "Arriba-Izq", dist
    elif angle >= 157.5 or angle < -157.5:   return "Izquierda", dist
    elif -157.5 <= angle < -112.5:           return "Abajo-Izq", dist
    elif -112.5 <= angle < -67.5:            return "Abajo", dist
    else:                                    return "Abajo-Der", dist


def draw_direction_arrow(frame, origin, current, label):
    if origin is None or current is None:
        return
    ox, oy = origin
    cx, cy = current
    if math.sqrt((cx - ox)**2 + (cy - oy)**2) < 5:
        return
    cv2.arrowedLine(frame, (ox, oy), (cx, cy), (0, 200, 255), 2, tipLength=0.3)
    cv2.putText(frame, label, (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 1)


# =============================================================================
# 3. Logica de solapamiento para detectar division
# =============================================================================

def porcentaje_dentro(box_nuevo, box_padre):
    """
    Retorna qué fracción del area de box_nuevo está dentro de box_padre.
    boxes en formato [x1, y1, x2, y2]
    Si el resultado > SPLIT_OVERLAP_RATIO -> es una division.
    """
    ix1 = max(box_nuevo[0], box_padre[0])
    iy1 = max(box_nuevo[1], box_padre[1])
    ix2 = min(box_nuevo[2], box_padre[2])
    iy2 = min(box_nuevo[3], box_padre[3])

    interseccion = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_nuevo   = (box_nuevo[2] - box_nuevo[0]) * (box_nuevo[3] - box_nuevo[1])

    if area_nuevo == 0:
        return 0.0
    return interseccion / area_nuevo


def resolver_nuevo_id(yolo_id, cx, cy, box_nuevo,
                      last_coords, last_boxes, id_grace, ids_activos,
                      id_remap, child_counts, already_claimed):
    """
    Para un yolo_id nunca visto, decide entre:
      A) Division:          box_nuevo está dentro del box de un padre activo
      B) Re-identificacion: padre en grace period y centroide cercano
      C) ID genuinamente nuevo

    Retorna (canonical_label, evento)
    """

    # --- Caso A: Division (prioridad sobre reid) ---
    # Busca padre ACTIVO cuyo box contenga al nuevo box
    mejor_padre_split  = None
    mejor_overlap      = 0.0

    for canon_label in ids_activos:
        if canon_label in already_claimed:
            continue
        if canon_label not in last_boxes:
            continue
        # Bloquear hijos de generar nietos — jerarquia maxima 1 nivel
        if '.' in canon_label:
            continue
        overlap = porcentaje_dentro(box_nuevo, last_boxes[canon_label])
        if overlap > SPLIT_OVERLAP_RATIO and overlap > mejor_overlap:
            mejor_overlap      = overlap
            mejor_padre_split  = canon_label

    if mejor_padre_split is not None:
        padre = mejor_padre_split
        already_claimed.add(padre)

        if padre not in child_counts:
            # Primera division: padre -> padre.1 y padre.2
            child_counts[padre] = 2
            hijo1 = f"{padre}.1"
            hijo2 = f"{padre}.2"

            # Clonar estado del padre hacia hijo1
            if padre in last_coords:
                last_coords[hijo1]  = last_coords[padre]
            if padre in last_boxes:
                last_boxes[hijo1]   = last_boxes[padre]
            if padre in id_persistence_ref[0]:
                id_persistence_ref[0][hijo1] = id_persistence_ref[0][padre]
            if padre in initial_coords_ref[0]:
                initial_coords_ref[0][hijo1] = initial_coords_ref[0][padre]

            # Reasignar remaps del padre a hijo1
            for k, v in list(id_remap.items()):
                if v == padre:
                    id_remap[k] = hijo1

            # Limpiar padre
            for d in [last_coords, last_boxes]:
                d.pop(padre, None)
            id_persistence_ref[0].pop(padre, None)
            ids_activos.discard(padre)

            return hijo2, ("split_first", padre, hijo1, hijo2)
        else:
            n = child_counts[padre] + 1
            child_counts[padre] = n
            nuevo_hijo = f"{padre}.{n}"
            return nuevo_hijo, ("split", padre, nuevo_hijo)

    # --- Caso B: Re-identificacion (padre en grace period, cercano) ---
    mejor_reid  = None
    mejor_dist  = float('inf')

    for canon_label, grace_count in id_grace.items():
        if canon_label in already_claimed:
            continue
        if canon_label not in last_coords:
            continue
        lx, ly = last_coords[canon_label]
        dist   = math.sqrt((cx - lx)**2 + (cy - ly)**2)
        if dist < REID_THRESHOLD and dist < mejor_dist:
            mejor_dist = dist
            mejor_reid = canon_label

    if mejor_reid is not None:
        already_claimed.add(mejor_reid)
        id_grace.pop(mejor_reid, None)
        return mejor_reid, "reid"

    # --- Caso C: ID genuinamente nuevo ---
    return str(yolo_id), "new"


# Referencias mutables para que resolver_nuevo_id pueda acceder a dicts del loop
# (evita pasar demasiados argumentos; se asignan antes del loop principal)
initial_coords_ref  = [{}]
id_persistence_ref  = [{}]


# =============================================================================
# 4. Calculos para graficas
# =============================================================================

def calcular_velocidad(df_movement):
    resultados = []
    for tid, grupo in df_movement.groupby('track_id'):
        grupo = grupo.sort_values('frame').reset_index(drop=True)
        for i in range(1, len(grupo)):
            dx_diff = grupo.loc[i, 'dx_cm'] - grupo.loc[i-1, 'dx_cm']
            dy_diff = grupo.loc[i, 'dy_cm'] - grupo.loc[i-1, 'dy_cm']
            vel     = math.sqrt(dx_diff**2 + dy_diff**2)
            resultados.append({
                'track_id': tid, 'frame': grupo.loc[i, 'frame'], 'velocidad': vel
            })
    return pd.DataFrame(resultados)


def calcular_excentricidad(df_area):
    if 'w_cm' in df_area.columns and 'h_cm' in df_area.columns:
        df = df_area.copy()
        df['excentricidad'] = df['w_cm'] / df['h_cm'].replace(0, np.nan)
        return df[['track_id', 'frame', 'excentricidad']]
    return None


def calcular_angulo(df_movement):
    df = df_movement.copy()
    df['angulo'] = df.apply(
        lambda r: math.degrees(math.atan2(r['dy_cm'], r['dx_cm'])), axis=1
    )
    return df[['track_id', 'frame', 'angulo']]


# =============================================================================
# 5. Graficas
# =============================================================================

def _plot_save(fig, ax, title, xlabel, ylabel, video_name, suffix, log_dir):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(facecolor='#2a2a4e', labelcolor=TEXT_COLOR, fontsize=7)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{video_name}_{suffix}.png'), dpi=150)
    plt.close()


VENTANA_SUAVIZADO = 5  # frames reales a cada lado, para limpiar ruido de deteccion

def suavizar_por_frame(frames, valores, ventana=VENTANA_SUAVIZADO):
    """Media movil centrada que respeta el frame real, no solo la posicion
    en la lista. Cuando un physarum se pierde unos frames y reaparece,
    los huecos se rellenan con NaN antes de promediar, para que la ventana
    de 'N vecinos' represente N frames de tiempo real (y no N detecciones
    que en realidad estan separadas por un hueco). No afecta los CSVs
    crudos, solo lo que se dibuja."""
    frames = list(frames)
    serie = pd.Series(list(valores), index=frames)
    frame_min, frame_max = int(min(frames)), int(max(frames))
    serie_completa = serie.reindex(range(frame_min, frame_max + 1))
    suavizada = serie_completa.rolling(window=ventana, center=True, min_periods=1).mean()
    return suavizada.loc[frames].values


def grafica_velocidad(df_vel, video_name, log_dir, fps=24):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_vel.groupby('track_id'):
        g = g.sort_values('frame')
        ax.plot(g['frame'].apply(lambda f: frames_a_horas(f, fps)),
                suavizar_por_frame(g['frame'], g['velocidad']), linewidth=1.5, alpha=0.85, label=tid)
    _plot_save(fig, ax, 'Velocidad vs Tiempo',
               'Tiempo (horas)', 'Velocidad (cm/frame)', video_name, 'velocidad', log_dir)


def grafica_angulo(df_ang, video_name, log_dir, fps=24):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_ang.groupby('track_id'):
        ax.plot(g['frame'].apply(lambda f: frames_a_horas(f, fps)),
                g['angulo'], linewidth=1.5, alpha=0.85, label=tid)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_yticklabels(['-180 izq', '-90 abajo', '0 der', '90 arriba', '180 izq'],
                       color=TEXT_COLOR, fontsize=8)
    _plot_save(fig, ax, 'Angulo de movimiento vs Tiempo',
               'Tiempo (horas)', 'Angulo (grados)', video_name, 'angulo', log_dir)


def grafica_excentricidad(df_exc, video_name, log_dir, fps=24):
    if df_exc is None: return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_exc.groupby('track_id'):
        g = g.sort_values('frame')
        ax.plot(g['frame'].apply(lambda f: frames_a_horas(f, fps)),
                suavizar_por_frame(g['frame'], g['excentricidad']), linewidth=1.5, alpha=0.85, label=tid)
    ax.axhline(1.0, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
    _plot_save(fig, ax, 'Excentricidad vs Tiempo',
               'Tiempo (horas)', 'Excentricidad', video_name, 'excentricidad', log_dir)


def grafica_area(grouped_a, video_name, log_dir, fps=24):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, data in grouped_a.items():
        horas = [frames_a_horas(f, fps) for f in data['frames']]
        areas_suavizadas = suavizar_por_frame(data['frames'], data['areas'])
        ax.plot(horas, areas_suavizadas, linewidth=1.5, label=tid)
    _plot_save(fig, ax, 'Area aproximada vs Tiempo',
               'Tiempo (horas)', 'Area (cm2)', video_name, 'area', log_dir)


def grafica_histograma_excentricidad(df_exc, video_name, log_dir):
    if df_exc is None or df_exc.empty: return
    vals = df_exc['excentricidad'].dropna()
    if vals.empty: return
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    ax.hist(vals, bins=25, color='#7B5EA7', edgecolor='#444466', alpha=0.85)
    media = vals.mean()
    ax.axvline(media, color='#FFB060', linewidth=1.5, linestyle='--',
               label=f'Media: {media:.2f}')
    _plot_save(fig, ax, f'Distribucion de excentricidad - {video_name}',
               'Excentricidad', 'Frecuencia', video_name, 'histograma_excentricidad', log_dir)


def grafica_vel_vs_exc(df_vel, df_exc, video_name, log_dir):
    if df_exc is None: return
    merged = pd.merge(df_vel, df_exc, on=['track_id', 'frame'])
    if merged.empty: return
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, merged['track_id'].nunique()))
    for (tid, g), color in zip(merged.groupby('track_id'), colors):
        ax.scatter(g['excentricidad'], g['velocidad'],
                   label=tid, alpha=0.6, s=20, color=color)
    ax.axvline(1.0, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
    _plot_save(fig, ax, 'Velocidad vs Excentricidad',
               'Excentricidad', 'Velocidad (cm/frame)', video_name, 'vel_vs_exc', log_dir)


def grafica_trayectoria(grouped, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, data in grouped.items():
        xs, ys = data['dx'], data['dy']
        ax.plot(xs, ys, linewidth=1.5, alpha=0.8, label=tid)
        ax.scatter([xs[0]], [ys[0]], marker='o', s=60, zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], marker='*', s=120, zorder=5)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='white', linewidth=0.5, alpha=0.4)
    _plot_save(fig, ax, 'Trayectoria 2D (desde punto inicial)',
               'dx (cm)', 'dy (cm)', video_name, 'trajectory2D', log_dir)


def grafica_distancia(grouped, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, data in grouped.items():
        ax.plot(data['horas'], data['dist'], linewidth=1.5, label=tid)  # <-- horas
    _plot_save(fig, ax, 'Distancia al punto inicial vs Tiempo',
               'Tiempo (horas)', 'Distancia (cm)', video_name, 'distance', log_dir)


def grafica_rosa_vientos(grouped, video_name, log_dir):
    all_dirs = [d for data in grouped.values() for d in data['dirs']]
    dir_counts = {}
    for d in all_dirs:
        dir_counts[d] = dir_counts.get(d, 0) + 1
    if not dir_counts: return
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG_COLOR); ax.set_facecolor(BG_COLOR)
    labels = list(dir_counts.keys())
    values = list(dir_counts.values())
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
    _, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                  colors=colors, startangle=90)
    for t in texts + autotexts:
        t.set_color(TEXT_COLOR)
    ax.set_title('Distribucion de direcciones', color=TEXT_COLOR, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{video_name}_directions.png'), dpi=150)
    plt.close()


def generar_tabla_resumen(movement_log, area_log, video_name, log_dir, fps=24):
    if not movement_log:
        return

    df_mov  = pd.DataFrame(movement_log,
                           columns=['track_id','frame','dx_cm','dy_cm','direction','distance_cm'])
    df_area = pd.DataFrame(area_log,
                           columns=['track_id','frame','area_cm2','w_cm','h_cm'])

    df_exc = calcular_excentricidad(df_area)
    df_vel = calcular_velocidad(df_mov)
    df_ang = calcular_angulo(df_mov)

    filas = []
    for tid in sorted(df_mov['track_id'].unique(), key=lambda x: df_mov[df_mov['track_id']==x]['frame'].min()):
        mov_tid  = df_mov[df_mov['track_id'] == tid].sort_values('frame')
        area_tid = df_area[df_area['track_id'] == tid].sort_values('frame')
        vel_tid  = df_vel[df_vel['track_id'] == tid].sort_values('frame') if not df_vel.empty else pd.DataFrame()
        exc_tid  = df_exc[df_exc['track_id'] == tid].sort_values('frame') if df_exc is not None else pd.DataFrame()
        ang_tid  = df_ang[df_ang['track_id'] == tid].sort_values('frame')

        ultimo     = mov_tid.iloc[-1]
        ultimo_area = area_tid.iloc[-1] if not area_tid.empty else None
        ultimo_vel  = vel_tid.iloc[-1]['velocidad'] if not vel_tid.empty else None
        ultimo_exc  = exc_tid.iloc[-1]['excentricidad'] if not exc_tid.empty else None
        ultimo_ang  = ang_tid.iloc[-1]['angulo'] if not ang_tid.empty else None

        filas.append({
            'ID':               tid,
            'frame inicio':     int(mov_tid.iloc[0]['frame']),
            'frame final':      int(ultimo['frame']),
            'frames totales':   len(mov_tid),
            'dx final (cm)':    round(ultimo['dx_cm'], 2),
            'dy final (cm)':    round(ultimo['dy_cm'], 2),
            'distancia (cm)':   round(ultimo['distance_cm'], 2),
            'direccion':        ultimo['direction'],
            'angulo (deg)':     round(ultimo_ang, 1) if ultimo_ang is not None else '-',
            'velocidad final':  round(ultimo_vel, 2)  if ultimo_vel is not None else '-',
            'area final (cm2)': round(ultimo_area['area_cm2'], 2) if ultimo_area is not None else '-',
            'excentricidad':    round(ultimo_exc, 2)  if ultimo_exc is not None else '-',
        })

    df_resumen = pd.DataFrame(filas)

    df_resumen.to_csv(
        os.path.join(log_dir, f'{video_name}_resumen.csv'),
        index=False, encoding='utf-8'
    )

    n_cols = len(df_resumen.columns)
    n_rows = len(df_resumen)
    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.4), max(2, n_rows * 0.5 + 1.2)))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    tabla = ax.table(
        cellText=df_resumen.values,
        colLabels=df_resumen.columns,
        cellLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.6)

    for (row, col), cell in tabla.get_celld().items():
        cell.set_edgecolor('#444466')
        if row == 0:
            cell.set_facecolor('#2a2a4e')
            cell.set_text_props(color=TEXT_COLOR, fontweight='bold')
        else:
            tid_fila = df_resumen.iloc[row - 1]['ID']
            if '.' in str(tid_fila):
                cell.set_facecolor('#3a2010')
                cell.set_text_props(color='#FFB060')
            else:
                cell.set_facecolor('#12122a')
                cell.set_text_props(color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(
        os.path.join(log_dir, f'{video_name}_resumen.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

def calcular_metricas_por_intervalo(df_movement, df_area, intervalo=10):
    """
    Divide el video en intervalos de N frames y calcula para cada physarum
    su velocidad promedio y distancia recorrida en ese intervalo.
    Retorna un DataFrame con una fila por (track_id, intervalo).
    """
    resultados = []
    frame_max = df_movement['frame'].max()

    for inicio in range(0, int(frame_max), int(intervalo)):
        fin = inicio + intervalo
        ventana = df_movement[
            (df_movement['frame'] >= inicio) &
            (df_movement['frame'] < fin)
        ]

        for tid, grupo in ventana.groupby('track_id'):
            grupo = grupo.sort_values('frame').reset_index(drop=True)
            if len(grupo) < 5:
                continue

            # Distancia recorrida en el intervalo (suma de pasos consecutivos)
            dist_total = 0
            for i in range(1, len(grupo)):
                dx_diff = grupo.loc[i, 'dx_cm'] - grupo.loc[i-1, 'dx_cm']
                dy_diff = grupo.loc[i, 'dy_cm'] - grupo.loc[i-1, 'dy_cm']
                dist_total += math.sqrt(dx_diff**2 + dy_diff**2)

            vel_promedio = dist_total / intervalo

            resultados.append({
                'intervalo':     inicio,
                'track_id':      tid,
                'distancia':     round(dist_total, 2),
                'velocidad_prom': round(vel_promedio, 2),
            })

    return pd.DataFrame(resultados)


def grafica_histogramas_intervalo(df_movement, area_log, video_name, log_dir, fps=24, intervalo=10):
    df_area = pd.DataFrame(area_log, columns=['track_id','frame','area_cm2','w_cm','h_cm'])
    df_int  = calcular_metricas_por_intervalo(df_movement, df_area, intervalo)

    if df_int.empty:
        return

    # Eliminar outliers del 2% superior
    vel_p90  = df_int['velocidad_prom'].quantile(0.90)
    dist_p90 = df_int['distancia'].quantile(0.90)

    df_vel_clean  = df_int[df_int['velocidad_prom'] <= vel_p90]
    df_dist_clean = df_int[df_int['distancia']      <= dist_p90]

    n_outliers_vel  = len(df_int) - len(df_vel_clean)
    n_outliers_dist = len(df_int) - len(df_dist_clean)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        apply_dark_style(ax)

    # Histograma velocidad
    axes[0].hist(df_vel_clean['velocidad_prom'], bins=20,
                 color='#7B5EA7', edgecolor='#444466', alpha=0.85)
    axes[0].set_title(f'Distribucion de velocidad promedio\n(intervalos de {intervalo} frames, sin top 10%)')
    axes[0].set_xlabel('Velocidad promedio (cm/frame)')
    axes[0].set_ylabel('Frecuencia (# physarums)')
    media_vel = df_vel_clean['velocidad_prom'].mean()
    axes[0].axvline(media_vel, color='#FFB060', linewidth=1.5,
                    linestyle='--', label=f'Media: {media_vel:.2f}\n({n_outliers_vel} outliers removidos)')
    axes[0].legend(facecolor='#2a2a4e', labelcolor=TEXT_COLOR, fontsize=8)

    # Histograma distancia
    axes[1].hist(df_dist_clean['distancia'], bins=20,
                 color='#3A7EBF', edgecolor='#444466', alpha=0.85)
    axes[1].set_title(f'Distribucion de distancia recorrida\n(intervalos de {intervalo} frames, sin top 10%)')
    axes[1].set_xlabel('Distancia recorrida (cm)')
    axes[1].set_ylabel('Frecuencia (# physarums)')
    media_dist = df_dist_clean['distancia'].mean()
    axes[1].axvline(media_dist, color='#FFB060', linewidth=1.5,
                    linestyle='--', label=f'Media: {media_dist:.2f}\n({n_outliers_dist} outliers removidos)')
    axes[1].legend(facecolor='#2a2a4e', labelcolor=TEXT_COLOR, fontsize=8)

    plt.suptitle('Comportamiento colectivo de physarums', color=TEXT_COLOR,
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{video_name}_histogramas.png'), dpi=150)
    plt.close()

    df_int.to_csv(
        os.path.join(log_dir, f'{video_name}_intervalos.csv'),
        index=False, encoding='utf-8'
    )
def generar_todas_las_graficas(movement_log, area_log, video_name, log_dir, fps=24):
    if not movement_log: return

    grouped = {}
    for tid, fr, dx, dy, direction, dist in movement_log:
        if tid not in grouped:
            grouped[tid] = {'frames': [], 'horas': [], 'dx': [], 'dy': [], 'dist': [], 'dirs': []}
        grouped[tid]['frames'].append(fr)
        grouped[tid]['horas'].append(frames_a_horas(fr, fps))  # <-- nuevo
        grouped[tid]['dx'].append(dx)
        grouped[tid]['dy'].append(dy)
        grouped[tid]['dist'].append(dist)
        grouped[tid]['dirs'].append(direction)

    grouped_a = {}
    for tid, fr, area, w, h in area_log:
        if tid not in grouped_a:
            grouped_a[tid] = {'frames': [], 'areas': []}
        grouped_a[tid]['frames'].append(fr)
        grouped_a[tid]['areas'].append(area)

    df_movement  = pd.DataFrame(movement_log,
                                columns=['track_id', 'frame', 'dx_cm', 'dy_cm',
                                         'direction', 'distance_cm'])
    df_area_full = pd.DataFrame(area_log,
                                columns=['track_id', 'frame', 'area_cm2', 'w_cm', 'h_cm'])

    df_vel = calcular_velocidad(df_movement)
    df_exc = calcular_excentricidad(df_area_full)
    df_ang = calcular_angulo(df_movement)

    grafica_trayectoria(grouped, video_name, log_dir)
    grafica_distancia(grouped, video_name, log_dir)       # ya usa 'horas'
    grafica_rosa_vientos(grouped, video_name, log_dir)
    grafica_area(grouped_a, video_name, log_dir, fps)
    if not df_vel.empty:
        grafica_velocidad(df_vel, video_name, log_dir, fps)
        grafica_angulo(df_ang, video_name, log_dir, fps)
        grafica_excentricidad(df_exc, video_name, log_dir, fps)
        grafica_vel_vs_exc(df_vel, df_exc, video_name, log_dir)
        grafica_histogramas_intervalo(df_movement, area_log, video_name, log_dir, fps)
        grafica_histograma_excentricidad(df_exc, video_name, log_dir)
        generar_tabla_resumen(movement_log, area_log, video_name, log_dir, fps)


# =============================================================================
# 6. Procesamiento de video principal
# =============================================================================

def detect_objects_from_video(video_path, max_detections=100):
    global initial_coords_ref, id_persistence_ref

    cap = cv2.VideoCapture(video_path)
    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps             = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer el video"); return

    display_w, display_h = 1020, 600
    roi_selection = cv2.selectROI(
        "Selecciona el area (ENTER para confirmar)",
        cv2.resize(first_frame, (display_w, display_h)), fromCenter=False
    )
    cv2.destroyWindow("Selecciona el area (ENTER para confirmar)")

    x_s, y_s, w_s, h_s = roi_selection
    scale_x = original_width / display_w
    scale_y = original_height / display_h
    x_roi = int(x_s * scale_x); y_roi = int(y_s * scale_y)
    w_roi = int(w_s * scale_x); h_roi = int(h_s * scale_y)
    if w_roi == 0 or h_roi == 0:
        x_roi, y_roi, w_roi, h_roi = 0, 0, original_width, original_height

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Estado del tracker ---
    initial_coords = {}   # label -> (cx, cy) punto de origen absoluto
    last_coords    = {}   # label -> (cx, cy) ultima posicion
    last_boxes     = {}   # label -> [x1,y1,x2,y2] ultimo bounding box  *** NUEVO ***
    id_persistence = {}   # label -> frames visibles
    id_grace       = {}   # label -> frames desde que desaparecio
    persistence_buffer = {}  # label -> [(frame, dx, dy, dir, dist, area, w, h)]
    ids_activos    = set()

    id_remap       = {}   # yolo_id (int) -> canonical label (str)
    child_counts   = {}   # label -> cuantos hijos tiene

    split_log      = []
    movement_log   = [] 
    area_log       = []

    # Exponer refs para resolver_nuevo_id
    initial_coords_ref[0] = initial_coords
    id_persistence_ref[0] = id_persistence

    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    save_dir    = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join('movement_logs', video_name)
    os.makedirs(log_dir, exist_ok=True)

    # Calibracion px -> cm: la camara captura ~28cm de alto en promedio
    # (varia un poco por video, pero esta aproximacion es suficiente).
    cm_por_px = ALTURA_CAPTURA_CM / original_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(
        os.path.join(save_dir, f"{video_name}_annotated.mp4"),
        fourcc, 20.0, (w_roi, h_roi)
    )

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        results   = model.track(frame_roi, persist=True, conf=0.4, iou=0.6) 

        ids_vistos_este_frame = set()
        already_claimed       = set()

        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes     = res.boxes.xyxy.int().cpu().tolist()
            class_ids = res.boxes.cls.int().cpu().tolist()

            for box, class_id, yolo_id in zip(boxes, class_ids, track_ids):
                if names.get(class_id, "").lower() != "physarum":
                    continue

                x1, y1, x2, y2 = box
                cx      = (x1 + x2) // 2
                cy      = (y1 + y2) // 2
                w_box   = x2 - x1
                h_box   = y2 - y1
                area_px = w_box * h_box

                # Resolver identidad si es un ID nuevo
                if yolo_id not in id_remap:
                    canonical, evento = resolver_nuevo_id(
                        yolo_id, cx, cy, box,
                        last_coords, last_boxes, id_grace, ids_activos,
                        id_remap, child_counts, already_claimed
                    )
                    id_remap[yolo_id] = canonical

                    # Registrar evento de division en el log
                    if isinstance(evento, tuple) and evento[0] in ("split_first", "split"):
                        if evento[0] == "split_first":
                            _, padre, hijo1, hijo2 = evento
                            split_log.append((frame_count, padre, hijo1, hijo2))
                        else:
                            _, padre, nuevo_hijo = evento
                            split_log.append((frame_count, padre, padre, nuevo_hijo))

                canonical = id_remap[yolo_id]
                ids_vistos_este_frame.add(canonical)
                # Seguridad: si por alguna razon es nieto, subirlo a hijo
                if canonical.count('.') > 1:
                    raiz = canonical.split('.')[0]
                    n = child_counts.get(raiz, 1) + 1
                    child_counts[raiz] = n
                    canonical = f"{raiz}.{n}"
                    id_remap[yolo_id] = canonical

                # Inicializar si es primera vez
                if canonical not in initial_coords:
                    initial_coords[canonical] = (cx, cy)
                    id_persistence[canonical] = 0

                id_grace[canonical]    = 0
                last_coords[canonical] = (cx, cy)
                last_boxes[canonical]  = box        # *** guardar box actual ***
                id_persistence[canonical] = id_persistence.get(canonical, 0) + 1

                ox, oy = initial_coords[canonical]
                dx     = cx - ox
                dy     = -(cy - oy)
                direction, dist = classify_direction(dx, dy)

                # Convertir a centimetros para el log/graficas (la logica
                # de tracking arriba sigue usando pixeles)
                dx_cm   = dx * cm_por_px
                dy_cm   = dy * cm_por_px
                dist_cm = dist * cm_por_px
                area_cm2 = area_px * (cm_por_px ** 2)
                w_cm    = w_box * cm_por_px
                h_cm    = h_box * cm_por_px

                # Buffer hasta confirmar MIN_PERSISTENCE
                entrada_mov  = (canonical, frame_count, round(dx_cm, 3), round(dy_cm, 3),
                                 direction, round(dist_cm, 3))
                entrada_area = (canonical, frame_count, round(area_cm2, 3),
                                 round(w_cm, 3), round(h_cm, 3))

                if id_persistence[canonical] < MIN_PERSISTENCE:
                    # Acumular en buffer sin guardar en log
                    if canonical not in persistence_buffer:
                        persistence_buffer[canonical] = []
                    persistence_buffer[canonical].append((entrada_mov, entrada_area))

                elif id_persistence[canonical] == MIN_PERSISTENCE:
                    # Confirmar — volcar todo el buffer al log
                    if canonical in persistence_buffer:
                        for em, ea in persistence_buffer[canonical]:
                            movement_log.append(em)
                            area_log.append(ea)
                        del persistence_buffer[canonical]
                    movement_log.append(entrada_mov)
                    area_log.append(entrada_area)

                else:
                    movement_log.append(entrada_mov)
                    area_log.append(entrada_area)

                # Dibujo
                is_child  = '.' in canonical
                box_color = (255, 150, 0) if is_child else (0, 255, 0)
                cv2.rectangle(frame_roi, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame_roi, (cx, cy), 4, (255, 255, 0), -1)
                cv2.putText(frame_roi, canonical, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 60), 1)
                cv2.putText(frame_roi, f'A:{area_cm2:.1f}cm2', (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 60), 1)
                draw_direction_arrow(
                    frame_roi,
                    initial_coords[canonical], (cx, cy),
                    f"{canonical}: {direction}"
                )

        # Actualizar ids activos
        ids_activos = set(ids_vistos_este_frame)

        # Grace period
        for label in list(id_grace.keys()):
            if label not in ids_vistos_este_frame:
                id_grace[label] = id_grace.get(label, 0) + 1
                if id_grace[label] > GRACE_PERIOD:
                    for d in [initial_coords, last_coords, last_boxes, id_persistence, id_grace]:
                        d.pop(label, None)
                    ids_activos.discard(label)
                    for k, v in list(id_remap.items()):
                        if v == label:
                            del id_remap[k]

        out.write(frame_roi)
        _, buffer = cv2.imencode('.jpg', frame_roi)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    out.release()

    # CSVs
    with open(os.path.join(log_dir, f'{video_name}_movement.csv'), 'w',
              newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(
            [['track_id', 'frame', 'dx_cm', 'dy_cm', 'direction', 'distance_cm']] + movement_log
        )
    with open(os.path.join(log_dir, f'{video_name}_area.csv'), 'w',
              newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(
            [['track_id', 'frame', 'area_cm2', 'w_cm', 'h_cm']] + area_log
        )
    if split_log:
        with open(os.path.join(log_dir, f'{video_name}_splits.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(
                [['frame', 'padre', 'hijo1', 'hijo2']] + split_log
            )

    generar_todas_las_graficas(movement_log, area_log, video_name, log_dir, fps)


# =============================================================================
# 7. Procesamiento de imagenes
# =============================================================================

def process_image_files(files):
    processed_filenames = []
    os.makedirs('uploads', exist_ok=True)
    for file in files:
        if file.filename == '': continue
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True, tracker="botsort.yaml")
        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            for box, class_id, track_id in zip(
                res.boxes.xyxy.int().cpu().tolist(),
                res.boxes.cls.int().cpu().tolist(),
                res.boxes.id.int().cpu().tolist()
            ):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        processed_filename = f"processed_{file.filename}"
        cv2.imwrite(os.path.join('uploads', processed_filename), frame)
        processed_filenames.append(processed_filename)
    return processed_filenames


# =============================================================================
# 8. Rutas Flask
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video_form')
def upload_video_form():
    return render_template('upload_video.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        files = request.files.getlist('files')
        return render_template('show_image.html', filenames=process_image_files(files))
    return render_template('upload_image.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(
        detect_objects_from_video(os.path.join('uploads', filename)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/uploads_image/<filename>')
def send_image(filename):
    return send_from_directory('uploads', filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)

@app.route('/annotated_video/<filename>')
def send_annotated_video(filename):
    video_name = os.path.splitext(filename)[0]
    path = os.path.join('detected_frames', video_name, f"{video_name}_annotated.mp4")
    if os.path.exists(path):
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    return "Video anotado no encontrado", 404

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)
    file = request.files['file']
    os.makedirs('uploads', exist_ok=True)
    file.save(os.path.join('uploads', file.filename))
    return redirect(url_for('play_video', filename=file.filename))

@app.route('/upload_video/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/movement_logs/<filename>')
def serve_log_file(filename):
    return send_from_directory('movement_logs', filename)


# =============================================================================
# 9. Ejecucion
# =============================================================================
if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
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
MIN_PERSISTENCE = 20

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
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


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
            dx_diff = grupo.loc[i, 'dx'] - grupo.loc[i-1, 'dx']
            dy_diff = grupo.loc[i, 'dy'] - grupo.loc[i-1, 'dy']
            vel     = math.sqrt(dx_diff**2 + dy_diff**2)
            resultados.append({
                'track_id': tid, 'frame': grupo.loc[i, 'frame'], 'velocidad': vel
            })
    return pd.DataFrame(resultados)


def calcular_excentricidad(df_area):
    if 'w' in df_area.columns and 'h' in df_area.columns:
        df = df_area.copy()
        df['excentricidad'] = df['w'] / df['h'].replace(0, np.nan)
        return df[['track_id', 'frame', 'excentricidad']]
    return None


def calcular_angulo(df_movement):
    df = df_movement.copy()
    df['angulo'] = df.apply(
        lambda r: math.degrees(math.atan2(r['dy'], r['dx'])), axis=1
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


def grafica_velocidad(df_vel, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_vel.groupby('track_id'):
        ax.plot(g['frame'], g['velocidad'], linewidth=1.5, alpha=0.85, label=tid)
    _plot_save(fig, ax, 'Velocidad (px/frame) vs Frame',
               'Frame', 'Velocidad (px/frame)', video_name, 'velocidad', log_dir)


def grafica_excentricidad(df_exc, video_name, log_dir):
    if df_exc is None: return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_exc.groupby('track_id'):
        ax.plot(g['frame'], g['excentricidad'], linewidth=1.5, alpha=0.85, label=tid)
    ax.axhline(1.0, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
    _plot_save(fig, ax, 'Excentricidad (ancho/alto) vs Frame',
               'Frame', 'Excentricidad', video_name, 'excentricidad', log_dir)


def grafica_angulo(df_ang, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, g in df_ang.groupby('track_id'):
        ax.plot(g['frame'], g['angulo'], linewidth=1.5, alpha=0.85, label=tid)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_yticklabels(['-180 izq', '-90 abajo', '0 der', '90 arriba', '180 izq'],
                       color=TEXT_COLOR, fontsize=8)
    _plot_save(fig, ax, 'Angulo de movimiento vs Frame',
               'Frame', 'Angulo (grados)', video_name, 'angulo', log_dir)


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
               'Excentricidad', 'Velocidad (px/frame)', video_name, 'vel_vs_exc', log_dir)


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
               'dx (px)', 'dy (px)', video_name, 'trajectory2D', log_dir)


def grafica_distancia(grouped, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, data in grouped.items():
        ax.plot(data['frames'], data['dist'], linewidth=1.5, label=tid)
    _plot_save(fig, ax, 'Distancia al punto inicial vs Frame',
               'Frame', 'Distancia (px)', video_name, 'distance', log_dir)


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


def grafica_area(grouped_a, video_name, log_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG_COLOR); apply_dark_style(ax)
    for tid, data in grouped_a.items():
        ax.plot(data['frames'], data['areas'], linewidth=1.5, label=tid)
    _plot_save(fig, ax, 'Area aproximada (px2) vs Frame',
               'Frame', 'Area (px2)', video_name, 'area', log_dir)

def generar_tabla_resumen(movement_log, area_log, video_name, log_dir):
    if not movement_log:
        return

    df_mov  = pd.DataFrame(movement_log,
                           columns=['track_id','frame','dx','dy','direction','distance_px'])
    df_area = pd.DataFrame(area_log,
                           columns=['track_id','frame','area_px','w','h'])

    df_exc = calcular_excentricidad(df_area)
    df_vel = calcular_velocidad(df_mov)
    df_ang = calcular_angulo(df_mov)

    # Construir info basica por ID
    info = {}
    for tid in df_mov['track_id'].unique():
        mov_tid  = df_mov[df_mov['track_id'] == tid].sort_values('frame')
        area_tid = df_area[df_area['track_id'] == tid].sort_values('frame')
        vel_tid  = df_vel[df_vel['track_id'] == tid].sort_values('frame') if not df_vel.empty else pd.DataFrame()
        exc_tid  = df_exc[df_exc['track_id'] == tid].sort_values('frame') if df_exc is not None else pd.DataFrame()
        ang_tid  = df_ang[df_ang['track_id'] == tid].sort_values('frame')

        info[tid] = {
            'frame_inicio':  int(mov_tid.iloc[0]['frame']),
            'frame_final':   int(mov_tid.iloc[-1]['frame']),
            'total_frames':  len(mov_tid),
            'ultimo_mov':    mov_tid.iloc[-1],
            'ultimo_area':   area_tid.iloc[-1] if not area_tid.empty else None,
            'ultimo_vel':    vel_tid.iloc[-1]['velocidad'] if not vel_tid.empty else None,
            'ultimo_exc':    exc_tid.iloc[-1]['excentricidad'] if not exc_tid.empty else None,
            'ultimo_ang':    ang_tid.iloc[-1]['angulo'] if not ang_tid.empty else None,
        }

    ids = list(info.keys())

    # Puntos inicial y final de cada ID para calcular distancia
    primeros_puntos = {}
    ultimos_puntos  = {}
    for tid in ids:
        mov_tid = df_mov[df_mov['track_id'] == tid].sort_values('frame')
        primeros_puntos[tid] = (float(mov_tid.iloc[0]['dx']),  float(mov_tid.iloc[0]['dy']))
        ultimos_puntos[tid]  = (float(mov_tid.iloc[-1]['dx']), float(mov_tid.iloc[-1]['dy']))

    # Detectar posibles continuaciones por tiempo Y distancia
    FRAME_THRESHOLD = 30
    DIST_THRESHOLD  = 50

    posible_continuacion = {}
    for tid_a in ids:
        frame_final_a = info[tid_a]['frame_final']
        ux_a, uy_a   = ultimos_puntos[tid_a]
        candidatos    = []

        for tid_b in ids:
            if tid_b == tid_a:
                continue
            frame_inicio_b = info[tid_b]['frame_inicio']
            diff = frame_inicio_b - frame_final_a
            if not (0 < diff <= FRAME_THRESHOLD):
                continue
            px_b, py_b = primeros_puntos[tid_b]
            dist = math.sqrt((px_b - ux_a)**2 + (py_b - uy_a)**2)
            if dist <= DIST_THRESHOLD:
                candidatos.append((diff, dist, tid_b))

        if candidatos:
            candidatos.sort()
            mejor = candidatos[0]
            posible_continuacion[tid_a] = f"{mejor[2]} (d={mejor[1]:.0f}px)"
        else:
            posible_continuacion[tid_a] = '-'

    # Calcular diferencia de frames para ordenar — los mas sospechosos primero
    def diff_frames(tid):
        cont = posible_continuacion[tid]
        if cont == '-':
            return float('inf')  # sin continuacion van al final
        # extraer el ID del string "ID60 (d=30px)" -> buscar en info
        tid_cont = cont.split(' ')[0]
        if tid_cont in info:
            return info[tid_cont]['frame_inicio'] - info[tid]['frame_final']
        return float('inf')

    # Construir filas ordenadas por diferencia de frames (menor diferencia primero)
    filas = []
    for tid in sorted(ids, key=diff_frames):
        d      = info[tid]
        ultimo = d['ultimo_mov']
        fila   = {
            'ID':               tid,
            'frame inicio':     d['frame_inicio'],
            'frame final':      d['frame_final'],
            'frames totales':   d['total_frames'],
            'dx final (px)':    int(ultimo['dx']),
            'dy final (px)':    int(ultimo['dy']),
            'distancia (px)':   round(ultimo['distance_px'], 1),
            'direccion':        ultimo['direction'],
            'angulo (deg)':     round(d['ultimo_ang'], 1) if d['ultimo_ang'] is not None else '-',
            'velocidad final':  round(d['ultimo_vel'], 2)  if d['ultimo_vel'] is not None else '-',
            'area final (px2)': int(d['ultimo_area']['area_px']) if d['ultimo_area'] is not None else '-',
            'excentricidad':    round(d['ultimo_exc'], 2)  if d['ultimo_exc'] is not None else '-',
            'continua en ID':   posible_continuacion[tid],
        }
        filas.append(fila)

    df_resumen = pd.DataFrame(filas)

    # CSV
    df_resumen.to_csv(
        os.path.join(log_dir, f'{video_name}_resumen.csv'),
        index=False, encoding='utf-8'
    )

    # PNG
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
            cont     = df_resumen.iloc[row - 1]['continua en ID']
            if '.' in str(tid_fila):
                cell.set_facecolor('#3a2010')
                cell.set_text_props(color='#FFB060')
            elif cont != '-':
                cell.set_facecolor('#0d1f3a')
                cell.set_text_props(color='#60B0FF')
            else:
                cell.set_facecolor('#12122a')
                cell.set_text_props(color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(
        os.path.join(log_dir, f'{video_name}_resumen.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()

def generar_todas_las_graficas(movement_log, area_log, video_name, log_dir):
    if not movement_log: return

    grouped = {}
    for tid, fr, dx, dy, direction, dist in movement_log:
        if tid not in grouped:
            grouped[tid] = {'frames': [], 'dx': [], 'dy': [], 'dist': [], 'dirs': []}
        grouped[tid]['frames'].append(fr); grouped[tid]['dx'].append(dx)
        grouped[tid]['dy'].append(dy);     grouped[tid]['dist'].append(dist)
        grouped[tid]['dirs'].append(direction)

    grouped_a = {}
    for tid, fr, area, w, h in area_log:
        if tid not in grouped_a:
            grouped_a[tid] = {'frames': [], 'areas': []}
        grouped_a[tid]['frames'].append(fr)
        grouped_a[tid]['areas'].append(area)

    df_movement  = pd.DataFrame(movement_log,
                                columns=['track_id', 'frame', 'dx', 'dy',
                                         'direction', 'distance_px'])
    df_area_full = pd.DataFrame(area_log,
                                columns=['track_id', 'frame', 'area_px', 'w', 'h'])

    df_vel = calcular_velocidad(df_movement)
    df_exc = calcular_excentricidad(df_area_full)
    df_ang = calcular_angulo(df_movement)

    grafica_trayectoria(grouped, video_name, log_dir)
    grafica_distancia(grouped, video_name, log_dir)
    grafica_rosa_vientos(grouped, video_name, log_dir)
    grafica_area(grouped_a, video_name, log_dir)
    if not df_vel.empty:
        grafica_velocidad(df_vel, video_name, log_dir)
        grafica_angulo(df_ang, video_name, log_dir)
        grafica_excentricidad(df_exc, video_name, log_dir)
        grafica_vel_vs_exc(df_vel, df_exc, video_name, log_dir)
        generar_tabla_resumen(movement_log, area_log, video_name, log_dir)


# =============================================================================
# 6. Procesamiento de video principal
# =============================================================================

def detect_objects_from_video(video_path, max_detections=100):
    global initial_coords_ref, id_persistence_ref

    cap = cv2.VideoCapture(video_path)
    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    log_dir = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)

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
        results   = model.track(frame_roi, persist=True, conf=0.3, iou=0.6)

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

                if id_persistence[canonical] >= MIN_PERSISTENCE:
                    movement_log.append((canonical, frame_count, dx, dy,
                                         direction, round(dist, 1)))
                    area_log.append((canonical, frame_count, area_px, w_box, h_box))

                # Dibujo
                is_child  = '.' in canonical
                box_color = (255, 150, 0) if is_child else (0, 255, 0)
                cv2.rectangle(frame_roi, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame_roi, (cx, cy), 4, (255, 255, 0), -1)
                cv2.putText(frame_roi, canonical, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
                cv2.putText(frame_roi, f'A:{area_px}px', (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
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
            [['track_id', 'frame', 'dx', 'dy', 'direction', 'distance_px']] + movement_log
        )
    with open(os.path.join(log_dir, f'{video_name}_area.csv'), 'w',
              newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(
            [['track_id', 'frame', 'area_px', 'w', 'h']] + area_log
        )
    if split_log:
        with open(os.path.join(log_dir, f'{video_name}_splits.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(
                [['frame', 'padre', 'hijo1', 'hijo2']] + split_log
            )

    generar_todas_las_graficas(movement_log, area_log, video_name, log_dir)


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
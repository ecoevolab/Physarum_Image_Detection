"""
Genera una tabla de resumen (mínimos y máximos) por video
a partir de todos los CSVs en movement_logs/*/csv/
"""

import os
import csv
import math
from pathlib import Path


def leer_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def calcular_resumen(video_id, movement_rows, area_rows):
    areas       = [safe_float(r['area_cm2']) for r in area_rows]
    areas       = [v for v in areas if v is not None]

    widths      = [safe_float(r['w_cm']) for r in area_rows]
    heights     = [safe_float(r['h_cm']) for r in area_rows]
    eccents = []
    for w, h in zip(widths, heights):
        if w is not None and h is not None and h > 0:
            eccents.append(w / h)

    # velocidad = distancia por frame (distance_cm)
    vels = [safe_float(r['distance_cm']) for r in movement_rows]
    vels = [v for v in vels if v is not None]

    # distancia total por track_id
    dist_por_track = {}
    for r in movement_rows:
        tid = r['track_id']
        d = safe_float(r['distance_cm'])
        if d is not None:
            dist_por_track[tid] = dist_por_track.get(tid, 0.0) + d
    totales = list(dist_por_track.values())

    def fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else 'N/A'

    return {
        'video':             video_id,
        'n_tracks':          len(dist_por_track),
        'area_min_cm2':      fmt(min(areas)    if areas   else None),
        'area_max_cm2':      fmt(max(areas)    if areas   else None),
        'veloc_min_cm_fr':   fmt(min(vels)     if vels    else None),
        'veloc_max_cm_fr':   fmt(max(vels)     if vels    else None),
        'eccent_min':        fmt(min(eccents)  if eccents else None),
        'eccent_max':        fmt(max(eccents)  if eccents else None),
        'dist_min_cm':       fmt(min(totales)  if totales else None),
        'dist_max_cm':       fmt(max(totales)  if totales else None),
    }


VIDEOS_INCLUIDOS = {
    '28.12.25', '27.09.25', '25.10.25', '22.12.25',
    '18.12.25', '13.11.25', '02.09.25', '01_10_25',
}


def main():
    base = Path('movement_logs')
    if not base.exists():
        print("No se encontró la carpeta movement_logs/")
        return

    resultados = []

    for video_dir in sorted(base.iterdir()):
        csv_dir = video_dir / 'csv'
        if not csv_dir.is_dir():
            continue

        vid = video_dir.name

        if vid not in VIDEOS_INCLUIDOS:
            continue

        movement_file = csv_dir / f'{vid}_movement.csv'
        area_file     = csv_dir / f'{vid}_area.csv'

        if not movement_file.exists() or not area_file.exists():
            print(f"  [omitido] {vid} — falta movement o area CSV")
            continue

        m_rows = leer_csv(movement_file)
        a_rows = leer_csv(area_file)

        if not m_rows or not a_rows:
            print(f"  [vacío]   {vid}")
            continue

        res = calcular_resumen(vid, m_rows, a_rows)
        resultados.append(res)
        print(f"  [ok]      {vid}")

    if not resultados:
        print("\nNo se encontraron datos para resumir.")
        return

    # ── Imprimir tabla en consola ──────────────────────────────────────────
    col_w = {
        'video':           14,
        'n_tracks':         8,
        'area_min_cm2':    12,
        'area_max_cm2':    12,
        'veloc_min_cm_fr': 16,
        'veloc_max_cm_fr': 16,
        'eccent_min':      12,
        'eccent_max':      12,
        'dist_min_cm':     12,
        'dist_max_cm':     12,
    }
    headers = {
        'video':           'Video',
        'n_tracks':        'Tracks',
        'area_min_cm2':    'Área min',
        'area_max_cm2':    'Área max',
        'veloc_min_cm_fr': 'Vel min (cm/fr)',
        'veloc_max_cm_fr': 'Vel max (cm/fr)',
        'eccent_min':      'Excent min',
        'eccent_max':      'Excent max',
        'dist_min_cm':     'Dist min (cm)',
        'dist_max_cm':     'Dist max (cm)',
    }

    sep = '+' + '+'.join('-' * (col_w[k] + 2) for k in col_w) + '+'
    header_row = '|' + '|'.join(f" {headers[k]:<{col_w[k]}} " for k in col_w) + '|'

    print()
    print(sep)
    print(header_row)
    print(sep)
    for r in resultados:
        row = '|' + '|'.join(f" {str(r[k]):<{col_w[k]}} " for k in col_w) + '|'
        print(row)
    print(sep)

    # ── Exportar CSV ──────────────────────────────────────────────────────
    out_path = Path('movement_logs') / 'resumen_global.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(col_w.keys()))
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nTabla exportada -> {out_path}")


if __name__ == '__main__':
    main()

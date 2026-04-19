import matplotlib
matplotlib.use('Agg')

import os
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, request,
    redirect, url_for, send_from_directory, jsonify
)
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt
import math

# =============================================================================
# 1. Configuración
# =============================================================================
app = Flask(__name__)
app.secret_key = '777'

model = YOLO("yolo11_custom_4.pt")  # <-- tu modelo de detection original
names = model.model.names

# =============================================================================
# 2. Utilidades de dirección
# =============================================================================

def classify_direction(dx, dy, threshold=10):
    dist = math.sqrt(dx**2 + dy**2)
    if dist < threshold:
        return "Estatico", dist
    angle = math.degrees(math.atan2(dy, dx))
    if -22.5 <= angle < 22.5:                  return "Derecha", dist
    elif 22.5 <= angle < 67.5:                 return "Arriba-Der", dist
    elif 67.5 <= angle < 112.5:                return "Arriba", dist
    elif 112.5 <= angle < 157.5:               return "Arriba-Izq", dist
    elif angle >= 157.5 or angle < -157.5:     return "Izquierda", dist
    elif -157.5 <= angle < -112.5:             return "Abajo-Izq", dist
    elif -112.5 <= angle < -67.5:              return "Abajo", dist
    else:                                      return "Abajo-Der", dist


def draw_direction_arrow(frame, origin, current, track_id, direction_label):
    if origin is None or current is None:
        return
    ox, oy = origin
    cx, cy = current
    if math.sqrt((cx-ox)**2 + (cy-oy)**2) < 5:
        return
    cv2.arrowedLine(frame, (ox, oy), (cx, cy), (0, 200, 255), 2, tipLength=0.3)
    cv2.putText(frame, f"ID{track_id}: {direction_label}",
                (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


# =============================================================================
# 3. Procesamiento de video
# =============================================================================

def detect_objects_from_video(video_path, max_detections=100):
    cap = cv2.VideoCapture(video_path)
    original_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer el video")
        return

    display_w, display_h = 1020, 600
    first_frame_display = cv2.resize(first_frame, (display_w, display_h))
    roi_selection = cv2.selectROI(
        "Selecciona el area (ENTER para confirmar)", first_frame_display, fromCenter=False
    )
    cv2.destroyWindow("Selecciona el area (ENTER para confirmar)")

    x_s, y_s, w_s, h_s = roi_selection
    scale_x = original_width  / display_w
    scale_y = original_height / display_h
    x_roi = int(x_s * scale_x); y_roi = int(y_s * scale_y)
    w_roi = int(w_s * scale_x); h_roi = int(h_s * scale_y)
    if w_roi == 0 or h_roi == 0:
        x_roi, y_roi, w_roi, h_roi = 0, 0, original_width, original_height

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    initial_coords  = {}
    last_coords     = {}
    id_persistence  = {}
    id_grace        = {}
    GRACE_PERIOD    = 15
    MIN_PERSISTENCE = 10

    movement_log = []
    area_log     = []

    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    save_dir    = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)
    log_dir     = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)

    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(save_dir, f"{video_name}_annotated.mp4")
    out         = cv2.VideoWriter(output_path, fourcc, 20.0, (w_roi, h_roi))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_roi = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        results = model.track(frame_roi, persist=True, conf=0.4, iou=0.3)

        ids_seen_this_frame = set()

        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes     = res.boxes.xyxy.int().cpu().tolist()
            class_ids = res.boxes.cls.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names.get(class_id, "unknown").lower()
                if class_name != "physarum":
                    continue

                ids_seen_this_frame.add(track_id)

                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area_px = (x2 - x1) * (y2 - y1)

                if track_id not in initial_coords:
                    initial_coords[track_id] = (cx, cy)
                    id_persistence[track_id] = 0

                id_grace[track_id]    = 0
                last_coords[track_id] = (cx, cy)
                id_persistence[track_id] = id_persistence.get(track_id, 0) + 1

                ox, oy = initial_coords[track_id]
                dx     = cx - ox
                dy     = -(cy - oy)
                direction, dist = classify_direction(dx, dy)

                if id_persistence[track_id] >= MIN_PERSISTENCE:
                    movement_log.append((track_id, frame_count, dx, dy, direction, round(dist, 1)))
                    area_log.append((track_id, frame_count, area_px))

                cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_roi, (cx, cy), 4, (255, 255, 0), -1)
                cv2.putText(frame_roi, f'ID:{track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                cv2.putText(frame_roi, f'A:{area_px}px', (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                draw_direction_arrow(frame_roi, initial_coords[track_id], (cx, cy),
                                     track_id, direction)

        # Grace period
        for tid in list(id_grace.keys()):
            if tid not in ids_seen_this_frame:
                id_grace[tid] = id_grace.get(tid, 0) + 1
                if id_grace[tid] > GRACE_PERIOD:
                    for d in [initial_coords, last_coords, id_persistence, id_grace]:
                        d.pop(tid, None)

        out.write(frame_roi)
        _, buffer = cv2.imencode('.jpg', frame_roi)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    out.release()

    # --- CSVs ---
    with open(os.path.join(log_dir, f'{video_name}_movement.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame', 'dx', 'dy', 'direction', 'distance_px'])
        writer.writerows(movement_log)

    with open(os.path.join(log_dir, f'{video_name}_area.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame', 'area_px'])
        writer.writerows(area_log)

    # --- Gráficas ---
    if movement_log:
        grouped = {}
        for tid, fr, dx, dy, direction, dist in movement_log:
            if tid not in grouped:
                grouped[tid] = {'frames': [], 'dx': [], 'dy': [], 'dist': [], 'dirs': []}
            grouped[tid]['frames'].append(fr)
            grouped[tid]['dx'].append(dx)
            grouped[tid]['dy'].append(dy)
            grouped[tid]['dist'].append(dist)
            grouped[tid]['dirs'].append(direction)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
        for tid, data in grouped.items():
            xs, ys = data['dx'], data['dy']
            ax.plot(xs, ys, linewidth=1.5, alpha=0.8, label=f'ID {tid}')
            ax.scatter([xs[0]], [ys[0]], marker='o', s=60, zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], marker='*', s=120, zorder=5)
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.4)
        ax.set_title('Trayectoria 2D (desde punto inicial)', color='white', fontsize=13)
        ax.set_xlabel('Δx (px, + = derecha)', color='white')
        ax.set_ylabel('Δy (px, + = arriba)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{video_name}_trajectory2D.png'), dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
        for tid, data in grouped.items():
            ax.plot(data['frames'], data['dist'], linewidth=1.5, label=f'ID {tid}')
        ax.set_title('Distancia al punto inicial vs Frame', color='white')
        ax.set_xlabel('Frame', color='white'); ax.set_ylabel('Distancia (px)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{video_name}_distance.png'), dpi=150)
        plt.close()

        all_dirs = [d for data in grouped.values() for d in data['dirs']]
        dir_counts = {}
        for d in all_dirs:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        if dir_counts:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
            labels = list(dir_counts.keys())
            values = list(dir_counts.values())
            colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(labels)))
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            for t in texts + autotexts:
                t.set_color('white')
            ax.set_title('Distribución de direcciones', color='white', fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'{video_name}_directions.png'), dpi=150)
            plt.close()

    if area_log:
        grouped_a = {}
        for tid, fr, area in area_log:
            if tid not in grouped_a:
                grouped_a[tid] = {'frames': [], 'areas': []}
            grouped_a[tid]['frames'].append(fr)
            grouped_a[tid]['areas'].append(area)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
        for tid, data in grouped_a.items():
            ax.plot(data['frames'], data['areas'], linewidth=1.5, label=f'ID {tid}')
        ax.set_title('Área aproximada (px²) vs Frame', color='white')
        ax.set_xlabel('Frame', color='white'); ax.set_ylabel('Área (px²)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2a2a4e', labelcolor='white', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{video_name}_area.png'), dpi=150)
        plt.close()


# =============================================================================
# 4. Procesamiento de imágenes
# =============================================================================

def process_image_files(files):
    processed_filenames = []
    os.makedirs('uploads', exist_ok=True)
    for file in files:
        if file.filename == '':
            continue
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True, tracker="botsort.yaml")
        res = results[0]
        if res.boxes is not None and res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes     = res.boxes.xyxy.int().cpu().tolist()
            class_ids = res.boxes.cls.int().cpu().tolist()
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
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
# 5. Rutas Flask
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
        processed_filenames = process_image_files(files)
        return render_template('show_image.html', filenames=processed_filenames)
    return render_template('upload_image.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(
        detect_objects_from_video(video_path),
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
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
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
# 6. Ejecución
# =============================================================================
if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
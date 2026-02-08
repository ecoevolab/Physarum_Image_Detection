import matplotlib
# CRITICAL: Use the 'Agg' backend to avoid GUI conflicts in Flask's background thread
matplotlib.use('Agg') 
# =============================================================================
import os
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, request,
    redirect, url_for, send_from_directory, session, jsonify
)
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt 

# =============================================================================
# 1. Configuración de la Aplicación y Carga del Modelo
# =============================================================================
app = Flask(__name__)
app.secret_key = '777' 

# Carga el modelo YOLOv11
model = YOLO("yolo11_custom_3.pt")
names = model.model.names

# =============================================================================
# 2. Funciones de Procesamiento de Video e Imagen
# =============================================================================
def detect_objects_from_video(video_path, max_detections=100):
    """
    Procesa video con:
    1. ROI interactivo escalado (sin distorsión).
    2. Buffer de persistencia para recuperar frames iniciales.
    3. Gráficas consolidadas por eje.
    """
    global initial_coords
    global id_persistence_count
    
    cap = cv2.VideoCapture(video_path)
    
    # --- 1. CONFIGURACIÓN DE DIMENSIONES Y ROI ---
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("Error al leer el video")
        return

    # Ventana de visualización para el usuario
    display_w, display_h = 1020, 600
    first_frame_display = cv2.resize(first_frame, (display_w, display_h))
    
    # Instrucciones: Seleccionar área y presionar ENTER
    roi_selection = cv2.selectROI("Selecciona el area (ENTER para confirmar)", first_frame_display, fromCenter=False)
    cv2.destroyWindow("Selecciona el area (ENTER para confirmar)")

    x_s, y_s, w_s, h_s = roi_selection
    
    # Escalar coordenadas de la ventana al tamaño real del video
    scale_x = original_width / display_w
    scale_y = original_height / display_h
    
    x_roi = int(x_s * scale_x)
    y_roi = int(y_s * scale_y)
    w_roi = int(w_s * scale_x)
    h_roi = int(h_s * scale_y)

    # Si no hay selección, usar todo el video
    if w_roi == 0 or h_roi == 0:
        x_roi, y_roi, w_roi, h_roi = 0, 0, original_width, original_height

    # Reiniciar video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # --- Inicialización de variables ---
    count = 0
    movement_log = [] 
    size_log = [] 
    temp_data_buffer = {} 
    initial_coords = {} 
    id_persistence_count = {}
    MIN_PERSISTENCE_FRAMES = 10 

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(save_dir, f"{video_name}_annotated.mp4")
    # El video guardado tendrá el tamaño EXACTO del recorte
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w_roi, h_roi))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        count += 1
        # Procesar frames pares para velocidad
        if count % 2 != 0: continue
        
        # --- 2. APLICAR RECORTE SIN DISTORSIÓN ---
        # Cortamos directamente del frame original
        frame_roi = frame[y_roi : y_roi+h_roi, x_roi : x_roi+w_roi]
        
        # YOLO analiza el recorte puro
        results = model.track(frame_roi, persist=True, conf=0.5, iou=0.6) 
        ids_in_frame = set() 

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_data = results[0].boxes
            track_ids = boxes_data.id.int().cpu().tolist()
            boxes = boxes_data.xyxy.int().cpu().tolist()
            class_ids = boxes_data.cls.int().cpu().tolist()
            
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names.get(class_id, "unknown").lower()
                ids_in_frame.add(track_id)
                
                if class_name == "physarum":
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    id_persistence_count[track_id] = id_persistence_count.get(track_id, 0) + 1
                    
                    if track_id not in initial_coords:
                        initial_coords[track_id] = (center_x, center_y)
                    
                    orig_x, orig_y = initial_coords[track_id]
                    rx, ry = center_x - orig_x, -(center_y - orig_y)
                    area = (x2 - x1) * (y2 - y1)

                    # --- 3. LÓGICA DE BUFFER Y PERSISTENCIA ---
                    if id_persistence_count[track_id] < MIN_PERSISTENCE_FRAMES:
                        if track_id not in temp_data_buffer:
                            temp_data_buffer[track_id] = []
                        temp_data_buffer[track_id].append([count, rx, ry, area])
                    
                    elif id_persistence_count[track_id] == MIN_PERSISTENCE_FRAMES:
                        if track_id in temp_data_buffer:
                            for old_f, old_rx, old_ry, old_a in temp_data_buffer[track_id]:
                                movement_log.append((track_id, old_f, old_rx, old_ry))
                                size_log.append((track_id, old_f, old_a))
                            del temp_data_buffer[track_id]
                        movement_log.append((track_id, count, rx, ry))
                        size_log.append((track_id, count, area))
                    else:
                        movement_log.append((track_id, count, rx, ry))
                        size_log.append((track_id, count, area))

                # Dibujo etiquetas
                cv2.rectangle(frame_roi, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame_roi, f'ID:{track_id}', (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Limpiar IDs perdidos
        for tid in list(id_persistence_count.keys()):
            if tid not in ids_in_frame:
                del id_persistence_count[tid]
                if tid in temp_data_buffer: del temp_data_buffer[tid]

        out.write(frame_roi)
        _, buffer = cv2.imencode('.jpg', frame_roi)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()
    out.release()
    
    # --- 4. GRÁFICAS CONSOLIDADAS ---
    if movement_log:
        grouped = {}
        for tid, f, rx, ry in movement_log:
            if tid not in grouped: grouped[tid] = []
            grouped[tid].append((f, rx, ry))
        
        # Gráfica X
        plt.figure(figsize=(10, 5))
        for tid, data in grouped.items():
            plt.plot([d[0] for d in data], [d[1] for d in data], label=f'ID {tid}')
        plt.title('Movimiento X'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'{video_name}_x.png')); plt.close()

        # Gráfica Y
        plt.figure(figsize=(10, 5))
        for tid, data in grouped.items():
            plt.plot([d[0] for d in data], [d[2] for d in data], label=f'ID {tid}')
        plt.title('Movimiento Y'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'{video_name}_y.png')); plt.close()

    if size_log:
        grouped_s = {}
        for tid, f, a in size_log:
            if tid not in grouped_s: grouped_s[tid] = []
            grouped_s[tid].append((f, a))
        
        plt.figure(figsize=(10, 5))
        for tid, data in grouped_s.items():
            plt.plot([d[0] for d in data], [d[1] for d in data], label=f'ID {tid}')
        plt.title('Área'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'{video_name}_area.png')); plt.close()
            

def process_image_files(files):
    """Procesa una lista de archivos de imagen subidos."""
    processed_filenames = []
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    for file in files:
        if file.filename == '':
            continue
        
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True, tracker = "botsort.yaml")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        processed_filename = f"processed_{file.filename}"
        processed_path = os.path.join('uploads', processed_filename)
        cv2.imwrite(processed_path, frame)
        processed_filenames.append(processed_filename)
    
    return processed_filenames

# =============================================================================
# 3. Rutas de la Aplicación
# =============================================================================

### Rutas principales y de navegación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

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


### Rutas para streams de video y archivos estáticos
@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    """
    Inicia el stream de detección de video con configuraciones por defecto.
    Ya no lee 'points' o 'max_detections' de la URL.
    """
    max_detections = 100
    
    video_path = os.path.join('uploads', filename)
    
    # 3. La función detect_objects_from_video ahora recibe el único parámetro necesario
    return Response(
        detect_objects_from_video(video_path, max_detections),
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
    annotated_video_path = os.path.join('detected_frames', video_name, f"{video_name}_annotated.mp4")
    if os.path.exists(annotated_video_path):
        return send_from_directory(os.path.dirname(annotated_video_path), os.path.basename(annotated_video_path))
    else:
        return "Video anotado no encontrado", 404

### Rutas de API para manejo de datos

# ... (código anterior) ...

### Rutas de API para manejo de datos

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # --- LÍNEA CORREGIDA ---
    # Redirige directamente a la página de reproducción del video.
    return redirect(url_for('play_video', filename=file.filename))

# ELIMINAR: def get_video_dimensions(video_path):

# ELIMINAR: @app.route('/set_point_page/<filename>')
# def set_point_page(filename):
#     # Esta ruta ya no tiene sentido
#     return redirect(url_for('play_video', filename=filename)) 

# ELIMINAR: @app.route('/get_first_frame/<filename>')
# def get_first_frame(filename):
#     # Esta ruta ya no tiene sentido
#     return # ... (código anterior)

@app.route('/upload_video/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)


# =============================================================================
# 4. Ejecución del Servidor
# =============================================================================
if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
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
def detect_objects_from_video(video_path, max_detections=2):
    """Procesa un archivo de video usando SOLO detección (NO tracking persistente) para evitar el error 'with_reid'."""
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    size_log = [] 

    print(f"Límite de detecciones establecido en: {max_detections}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(save_dir, f"{video_name}_annotated.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1020, 600))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        
        frame = cv2.resize(frame, (1020, 600))
        
        # --- CAMBIO CRÍTICO: Usar SOLO detección para saltar el error del tracker ---
        results = model(frame) # Usa model() en lugar de model.track()
        
        # Como no hay tracking, asignamos un ID temporal para las gráficas.
        track_id_counter = 1 
        
        if results[0].boxes is not None: # El ID ya no existe en el resultado de model()
            boxes_data = results[0].boxes
            
            sorted_indices = boxes_data.conf.argsort(descending=True)
            top_detections_indices = sorted_indices[:max_detections]
            
            boxes = boxes_data.xyxy[top_detections_indices].int().cpu().tolist()
            class_ids = boxes_data.cls[top_detections_indices].int().cpu().tolist()
            
            # Los track_ids ya no se obtienen del modelo, los simulamos
            
            for box, class_id in zip(boxes, class_ids):
                track_id = track_id_counter # Usar un ID temporal
                track_id_counter += 1
                
                class_name = names.get(class_id, "unknown").lower()
                x1, y1, x2, y2 = box
                
                if class_name == "physarum":
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    size_log.append((track_id, count, area))
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{track_id} - {class_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        out.write(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    out.release()
    
    # --- LÓGICA DE GRÁFICOS Y CSV PARA TAMAÑO (size_log) ---
    # El código de las gráficas es robusto y usará los IDs temporales que generamos
    if size_log:
        # ... (tu código de agrupación y generación de gráficas size_log aquí) ...
        grouped_sizes = {}
        for track_id, frame, area in size_log:
            if track_id not in grouped_sizes:
                grouped_sizes[track_id] = []
            grouped_sizes[track_id].append((frame, area))

        for track_id, data in grouped_sizes.items():
            size_log_path = os.path.join(log_dir, f"{video_name}_track_{track_id}_size.csv")
            with open(size_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'area'])
                writer.writerows(data)

            frames = [row[0] for row in data]
            areas = [row[1] for row in data]

            plt.figure(figsize=(10, 5))
            plt.plot(frames, areas, label='Área del recuadro')
            plt.xlabel('Frame')
            plt.ylabel('Área (píxeles)')
            plt.title(f'Tamaño del recuadro del objeto ID {track_id}')
            plt.legend()
            plt.grid(True)

            size_plot_path = os.path.join(log_dir, f'{video_name}_track_{track_id}_size_plot.png')
            plt.savefig(size_plot_path)
            plt.close()

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
    # Establece valores por defecto directamente. max_detections es 2 por defecto.
    max_detections = 2 
    
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
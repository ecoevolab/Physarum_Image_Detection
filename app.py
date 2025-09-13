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
def detect_objects_from_webcam():
    """Procesa el feed de la cámara web con detección y tracking de objetos."""
    count = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_objects_from_video(video_path, reference=None, max_detections=2):
    """Procesa un archivo de video con detección, tracking y registro de movimiento."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    movement_log = []

    # The max_detections value is now passed as a parameter
    print(f"Límite de detecciones establecido en: {max_detections}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)

    log_dir = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{video_name}.csv")

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
        results = model.track(frame, persist=True)
        
        # ... (Rest of the function logic is the same) ...
        # (Your filtering code should now work without the session call)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_data = results[0].boxes
            
            # Ordena las detecciones por confianza en orden descendente
            sorted_indices = boxes_data.conf.argsort(descending=True)
            
            # Limita las detecciones al número máximo del usuario
            top_detections_indices = sorted_indices[:max_detections]
            
            # ... (Rest of your detection and drawing logic) ...
            boxes = boxes_data.xyxy[top_detections_indices].int().cpu().tolist()
            class_ids = boxes_data.cls[top_detections_indices].int().cpu().tolist()
            track_ids = boxes_data.id[top_detections_indices].int().cpu().tolist()
            
            if reference:
                ref_x, ref_y = int(reference['x']), int(reference['y'])
                cv2.circle(frame, (ref_x, ref_y), 5, (0, 0, 255), -1)

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id].lower()
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if class_name == "physarum" and reference:
                    dx = center_x - ref_x
                    dy = center_y - ref_y
                    movement_log.append((track_id, count, dx, dy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {class_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        out.write(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
    
    if movement_log:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame', 'dx', 'dy'])
            writer.writerows(movement_log)

        dx_vals = [row[2] for row in movement_log]
        dy_vals = [row[3] for row in movement_log]
        frames = [row[1] for row in movement_log]

        plt.figure(figsize=(10, 5))
        plt.plot(frames, dx_vals, label='dx')
        plt.plot(frames, dy_vals, label='dy')
        plt.xlabel('Frame')
        plt.ylabel('Desplazamiento relativo')
        plt.title('Movimiento relativo del objeto respecto al punto medio')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(log_dir, f'{video_name}_plot.png')
        plt.savefig(plot_path)
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
        results = model.track(frame, persist=True)

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
    # Retrieve values from the session within the route function
    reference = session.get('reference_point')
    max_detections = session.get('max_detections', 2)
    video_path = os.path.join('uploads', filename)
    # Pass the values as arguments to the processing function
    return Response(detect_objects_from_video(video_path, reference, max_detections),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
@app.route('/set_reference_point', methods=['POST'])
def set_reference_point():
    data = request.get_json()
    if data and 'x' in data and 'y' in data:
        session['reference_point'] = {'x': data['x'], 'y': data['y']}
        return jsonify({'status': 'ok', 'message': 'Punto guardado'})
    return jsonify({'status': 'error', 'message': 'Datos inválidos'}), 400

# Modifica la función upload_video
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

    # Redirige a una nueva ruta para seleccionar el punto de referencia
    return redirect(url_for('set_point_page', filename=file.filename))

@app.route('/set_point_page/<filename>')
def set_point_page(filename):
    return render_template('set_point.html', filename=filename)

@app.route('/get_first_frame/<filename>')
def get_first_frame(filename):
    video_path = os.path.join('uploads', filename)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error al abrir el video", 500

    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return "No se pudo leer el primer fotograma", 500

    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')

# Coloca esta nueva ruta junto a las otras rutas de tu aplicación.
@app.route('/set_max_detections', methods=['POST'])
def set_max_detections():
    data = request.get_json()
    if data and 'max_detections' in data:
        try:
            max_num = int(data['max_detections'])
            session['max_detections'] = max_num
            return jsonify({'status': 'ok', 'message': 'Límite guardado'})
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Valor inválido'}), 400
    return jsonify({'status': 'error', 'message': 'Datos inválidos'}), 400

@app.route('/upload_video/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)


# =============================================================================
# 4. Ejecución del Servidor
# =============================================================================
if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, session, jsonify
from ultralytics import YOLO
import csv

# Load the YOLOv11 model
model = YOLO("yolo11_custom_3.pt")
names = model.model.names

app = Flask(__name__)
app.secret_key = '777' 

@app.route('/set_reference_point', methods=['POST'])
def set_reference_point():
    data = request.get_json()
    if data and 'x' in data and 'y' in data:
        session['reference_point'] = {'x': data['x'], 'y': data['y']}
        return jsonify({'status': 'ok', 'message': 'Punto guardado'})
    return jsonify({'status': 'error', 'message': 'Datos inválidos'}), 400

@app.route('/get_reference_point')
def get_reference_point():
    # Opcional: para que puedas obtener el punto si quieres mostrarlo al cargar la página
    ref = session.get('reference_point')
    return jsonify(ref if ref else {})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

def detect_objects_from_webcam():
    count=0
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1  # Increment the global count
        if count % 2 != 0:
           continue
        # Resize the frame to (1020, 600)
        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLOv8 tracking on the frame
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

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return redirect(request.url)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        processed_filenames = []

        for file in files:
            image_path = os.path.join('uploads', file.filename)
            file.save(image_path)

            # Leer y procesar imagen
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

        return render_template('show_image.html', filenames=processed_filenames)

    return render_template('upload_image.html')


@app.route('/uploads_image/<filename>')
def send_image(filename):
    return send_from_directory('uploads', filename)


@app.route('/upload_video_form')
def upload_video_form():
    return render_template('upload_video.html')
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file to the uploads folder
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Redirect to the video playback page after upload
    return redirect(url_for('play_video', filename=file.filename))

@app.route('/upload_video/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)


def process_video(video_path, reference=None):
    cap = cv2.VideoCapture(video_path)
    ...
    # Lo mismo de antes, pero sin `yield`
    ...
    print("Procesamiento completo")

def detect_objects_from_video(video_path, reference=None):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0
    movement_log = []

    # Extraer el nombre del video para la carpeta
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join('detected_frames', video_name)
    os.makedirs(save_dir, exist_ok=True)

    # Carpeta para guardar el CSV y la gráfica
    log_dir = 'movement_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{video_name}.csv")

    # Configurar VideoWriter para guardar video anotado
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

        # Resize the frame
        frame = cv2.resize(frame, (1020, 600))

        # Run YOLOv11 tracking
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            if reference:
                ref_x, ref_y = int(reference['x']), int(reference['y'])

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = names[class_id].lower()
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if class_name == "physarum" and reference:
                    dx = center_x - ref_x
                    dy = center_y - ref_y
                    movement_log.append((track_id, count, dx, dy))

                # Dibujar detección
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {class_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Guardar el frame en carpeta (opcional)
            frame_filename = os.path.join(save_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        # Guardar el frame anotado en el video
        out.write(frame)

        # Codificar el frame para el stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()

    # Guardar log de movimiento
    if movement_log:
        import csv
        import matplotlib.pyplot as plt

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame', 'dx', 'dy'])
            writer.writerows(movement_log)

        # Graficar desplazamientos dx y dy
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

    print(f"Total cuadros guardados con detecciones: {saved_count}")

@app.route('/video_feed/<filename>')
def video_feed(filename):
    reference = session.get('reference_point')  # Captura el punto medio
    video_path = os.path.join('uploads', filename)
    return Response(detect_objects_from_video(video_path, reference),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/annotated_video/<filename>')
def send_annotated_video(filename):
    video_name = os.path.splitext(filename)[0]
    annotated_video_path = os.path.join('detected_frames', video_name, f"{video_name}_annotated.mp4")
    if os.path.exists(annotated_video_path):
        return send_from_directory(os.path.dirname(annotated_video_path), os.path.basename(annotated_video_path))
    else:
        return "Video anotado no encontrado", 404

if __name__ == '__main__':
    app.run('0.0.0.0',debug=False, port=8080)
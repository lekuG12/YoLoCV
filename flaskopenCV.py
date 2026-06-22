import cv2 as cv

from flask import Flask, render_template, jsonify, Response
import atexit

from ultralytics import YOLO

model = YOLO('best.pt')
app = Flask(__name__)
cap = None  # Initialize cap as None

# Add initialization and cleanup functions
def init_camera():
    if cap is None or not cap.isOpened():
        cap = cv.VideoCapture(0)
    return cap.isOpened()

@atexit.register
def cleanup():
    if cap is not None and cap.isOpened():
        cap.release()

def captureFrames():
    if not init_camera():
        print('Could not initialize camera')
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                results = model(frame, save=True)
                annotated = results[0].plot()
                ret, buffer = cv.imencode('.jpg', annotated)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break
    except Exception as e:
        print(f"Error in captureFrames: {str(e)}")
        if cap is not None:
            cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    if init_camera():
        return render_template('index.html')
    return jsonify({'error': 'Failed to start camera'}), 500


@app.route('/stop', methods=['POST'])
def stop():
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    return render_template('stop.html')


@app.route('/video_capture')
def video_capture():
    return Response(captureFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    init_camera()  # Initialize camera at startup
    app.run(debug=True, port=2000)
import cv2 as cv
from flask import Flask, render_template, jsonify, Response
import atexit
from ultralytics import YOLO

model = YOLO('best.pt')
app = Flask(__name__)


# ── Camera wrapper — no global variables ─────────────────────────────────────
class Camera:
    def __init__(self):
        self._cap = None

    def start(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv.VideoCapture(0)
        return self._cap.isOpened()

    def stop(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    def is_open(self):
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()


camera = Camera()


@atexit.register
def cleanup():
    camera.stop()


def captureFrames():
    if not camera.start():
        print('Could not initialize camera')
        return

    try:
        while True:
            ret, frame = camera.read()
            if ret:
                results = model(frame, save=True)
                annotated = results[0].plot()
                ret, buffer = cv.imencode('.jpg', annotated)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                break
    except Exception as e:
        print(f"Error in captureFrames: {str(e)}")
        camera.stop()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    if camera.start():
        return render_template('index.html')
    return jsonify({'error': 'Failed to start camera'}), 500


@app.route('/stop', methods=['POST'])
def stop():
    camera.stop()
    return render_template('stop.html')


@app.route('/video_capture')
def video_capture():
    return Response(captureFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    camera.start()
    app.run(debug=True, port=2000)
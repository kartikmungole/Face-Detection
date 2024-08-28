from django.shortcuts import render
import cv2
from django.http import StreamingHttpResponse



def gen():
    face_cap = cv2.CascadeClassifier("C:/Users/Kartik Mungole/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, video_data = video_capture.read()
        color = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(
            color,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', video_data)
        if ret:
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    video_capture.release()


def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'face_detection/video.html')
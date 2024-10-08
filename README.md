## Django Face Detection Project ##

This project integrates face detection into a Django web application using OpenCV. It provides a live video feed with face detection capabilities through a web interface.

**Features**:
Live Video Feed: Displays real-time video from the webcam.
Face Detection: Uses OpenCV to detect faces and draw bounding boxes around them.

**Requirements**
Python 3.11 or higher
Django
OpenCV

**Project Structure**
face_detection_project/: The root directory of the Django project.
face_detection/: The app containing face detection functionality.
migrations/: Database migrations for the app.
templates/face_detection/: HTML templates for the app.
__init__.py: Initialization file for the app.
admin.py: Admin configuration for the app.
apps.py: App configuration.
models.py: Data models for the app (currently not used).
tests.py: Test cases for the app.
urls.py: URL routing for the app.
views.py: View functions for handling requests.
face_detection_project/: Main project configuration.
__init__.py: Initialization file for the project.
asgi.py: ASGI configuration for the project.
settings.py: Settings for the project.
urls.py: URL routing for the project.
wsgi.py: WSGI configuration for the project.

**Troubleshooting**
Video Feed Not Displaying: Ensure your webcam is properly connected and accessible. Check the OpenCV installation and verify that the haarcascade_frontalface_default.xml file path is correct.

**Acknowledgments**
OpenCV: For providing the face detection functionality.
Django: For being the web framework used in this project.

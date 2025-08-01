# Real-Time Object Detection System Using YOLOv5 and Flask

This project is a real-time object detection system that leverages the YOLOv5 model for detecting objects in a video stream from a webcam or other video input. The system is built using a Flask web application to serve the video feed, providing an interactive, real-time experience directly in a browser. YOLOv5's efficiency and speed make it ideal for real-time applications where both performance and accuracy are critical.


### Features
- Real-Time Object Detection: Detects and classifies objects in real-time using YOLOv5.
- Web-Based Interface: Access video feed with detections via a browser-based interface built with HTML and CSS.
- Responsive UI: An intuitive and modern UI design provides a user-friendly experience.

### How It Works
- Video Capture: Captures frames from a webcam or other video input.
- Object Detection: Each frame is passed through the YOLOv5 model to detect objects.
- Annotation: Detected objects are labeled and bounded in the frames.
- Streaming to Web Interface: The Flask app streams these annotated frames in real-time to a web page, accessible from any browser.


### Installation
1. Clone the repository
   ```sh
   git clone https://github.com/letsdoitbycode/Objectify-YOLOv5
   cd Objectify-YOLOv5
   ```

3. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install torch torchvision opencv-python matplotlib
   pip install requirements.txt   #else you can do this directly
   ```

4. Set Up YOLOv5 in the Project:
   ```sh
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   ```

5. Return to the main project folder:
   ```sh
   cd..
   ```
   
6. Run the Flask app:
    ```sh
   python app.py
    ```

### Project Structure
```plaintext
Objectify-YOLOv5/
│
├── templates/
│   └── index.html          # HTML file for UI 
├── static/
│   └── style.css          # CSS file for styling
├── venv                    # Virtual environment
├── yolov5                  # Model YOLOv5 
├── README.md               # This README file
├── app.py                  # Main Flask application
├── yolov5s.pt              # mirror of the YOLOv5 project
└── requirements.txt        # requirement file


```

### Acknowledgements
- Ultralytics for the YOLOv5 model: YOLOv5 GitHub Repository
- OpenCV for video processing
- Flask for powering the web interface

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.


## DEMP APPLICATION
----

https://github.com/Taffazzul-Bin-Azam/AI-Based-Object-Detection-System/blob/main/Screenshot%20(506).png

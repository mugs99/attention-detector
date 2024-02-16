# Face Attention Monitoring System

This project utilizes computer vision to monitor and assess a person's face for attention and distance parameters. It uses facial landmarks and measurements to determine the face size, eye distance, and eyebrow distance, providing insights into the subject's attention level and proximity to the camera.

## Features

- **Real Time Monitoring**
- **Face Size Estimation**: Calculates the size of the subject's face.
- **Eye Distance Measurement**: Determines the distance between the eyes.
- **Eyebrow Distance Measurement**: Measures the distance between the eyebrows.
- **Face Orientation**: Estimates the orientation of the face in degrees.
- **Distance Estimation**: Uses known parameters to estimate the subject's distance from the camera.
- **Attention Level Check**: Analyzes face size, orientation, and distance to assess attention level.
- **Database Integration**: Stores face data, including timestamp, face angle, zone text, and attention text, in an SQLite database.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- NumPy
- SQLite3

## Note
The threshold values might vary depending on the person, adjust accordingly.
You will also need to capture the reference image yourself beforehand.

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/mugs99/face-attention-monitoring.git
   cd face-attention-monitoring

2. Install the required dependencies:

   pip install -r requirements.txt

3. run the script:

   python attention.py

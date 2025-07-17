# Football Analysis System

## Description
This project implements a comprehensive football analysis system leveraging advanced machine learning, deep learning, and computer vision techniques. The system processes football match videos to analyze player movements, track objects (players, referees, and the ball), and measure key performance metrics such as player speed and distance covered. It integrates various computer vision tasks to provide deep insights into game dynamics.

## Key Features & Objectives
The system is designed to achieve the following:

* **Object Detection:** Detect players, referees, and footballs in video frames. The system utilizes YOLO models (e.g., YOLOv8 for detection, with a provided notebook for fine-tuning a YOLOv5 model on custom datasets).
* **Object Tracking:** Track detected objects (players, referees, and ball) across video frames using algorithms like ByteTrack, ensuring continuity of identification.
* **Team Assignment:** Automatically assign players to their respective teams based on t-shirt color clustering using the KMeans algorithm applied to pixel segmentation.
* **Ball Possession:** Assign ball possession to specific players frame-by-frame based on proximity to the ball.
* **Camera Movement Estimation:** Measure and account for camera movement using optical flow techniques (Lucas-Kanade method) to stabilize player positions.
* **Perspective Transformation:** Transform player and ball positions from pixel coordinates to real-world measurements (meters) on the court, enabling accurate distance and speed calculations.
* **Speed and Distance Measurement:** Calculate player speed (km/h) and total distance covered (meters) during the match.
* **Data Persistence:** The system can read and write processed tracking data and camera movement estimates to stub files (`.pkl`) for faster processing in subsequent runs.
* **Visual Annotations:** It generates output videos with visual annotations for bounding boxes, player IDs, team colors, ball possession, camera movement, speed, and distance.

## Technologies Used
* **Python:** The core programming language for the entire system.
* **YOLOv5/YOLOv8:** Used for real-time object detection of players, referees, and the ball. A training notebook is provided for fine-tuning YOLOv5.
* **Ultralytics:** Provides the framework for implementing and utilizing YOLO models.
* **Supervision:** Utilized for object detection utilities and tracking algorithms (e.g., ByteTrack).
* **OpenCV (`cv2`):** Extensive use for video reading/writing, image processing, optical flow (Lucas-Kanade), and perspective transformations.
* **Scikit-learn (`sklearn`):** Specifically for KMeans clustering to assign player teams based on color.
* **NumPy:** Essential for numerical operations and array manipulation across various modules.
* **Pandas:** Used for efficient data handling, particularly in interpolating missing ball positions.
* **Roboflow:** Used in the training notebook for dataset management and downloading.

## Installation & Setup
To set up and run the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd football-analysis-system
    ```
    *(Note: Replace `<repository_url>` with the actual URL of the repository.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Install the necessary Python libraries using pip:
    ```bash
    pip install opencv-python scikit-learn numpy pandas ultralytics supervision roboflow
    ```
    *(Note: The exact versions of libraries are not specified in the provided files, so using the latest compatible versions might be necessary.)*

4.  **Download pre-trained models and datasets (Optional for full training):**
    * The `main.py` script expects a YOLO model at `models/best.pt`. You may need to train your own model or acquire a pre-trained one. The `football_training_yolo_v5.ipynb` notebook provides steps to train a YOLOv5 model using a Roboflow dataset.
    * To utilize the `football_training_yolo_v5.ipynb` notebook for training, ensure you have a Roboflow API key.

## Usage
To run the football analysis system and generate an annotated video:

1.  **Place your input video:** Put your video file (e.g., `08fd33_4.mp4`) in the `input_videos/` directory. If your video has a different filename, you will need to modify the `video_path` variable in `main.py`.

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **View the output:** The analyzed video with all annotations will be saved as `output_video.avi` in the `output_videos/` directory.

## Contribution
Contributions to this project are welcome! Please feel free to fork the repository, open issues for bugs or feature requests, or submit pull requests with improvements.

## License
This project is open-source and available under an appropriate license (e.g., MIT License).

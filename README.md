# Face Swapping 

This repository contains a real-time face-swapping application built using Streamlit, OpenCV, MediaPipe, and InsightFace. The app allows users to capture video from their Mac's camera (or ManyCam) and swap faces using an uploaded image. Additionally, the repository includes a Jupyter Notebook (`gdsc-face-swap.ipynb`) for experimenting with face-swapping techniques.

## Features
- **Real-time Face Detection** using MediaPipe
- **Facial Landmarks Extraction** for accurate face mapping
- **Face Swapping using Delaunay Triangulation**
- **Integration with ManyCam** to work with Google Meet & other video apps
- **Screenshot & Merge Feature** to blend real-time feed with uploaded images

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-swap-app.git
cd face-swap-app
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# For Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running the Streamlit App
```bash
streamlit run app.py
```

## Running the Jupyter Notebook
If you want to experiment with face-swapping manually, open the notebook:
```bash
jupyter notebook gdsc-face-swap.ipynb
```


## Troubleshooting
- If **no facial landmarks** are detected, ensure proper lighting & a clear face in the camera.
- If **ManyCam is not detected**, ensure it's set as the default video source in Google Meet.
- For performance issues, use a GPU (`ctx_id=0`) instead of CPU (`ctx_id=-1`) in InsightFace.

## Contributing
Feel free to submit issues or pull requests to improve this project.

## License
This project is licensed under the MIT License.

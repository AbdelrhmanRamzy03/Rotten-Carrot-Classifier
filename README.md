# 🥕 Carrot Freshness Classifier

This Python script detects carrots in an image and evaluates their freshness based on visual cues of rot or decay. It uses OpenCV to segment the carrot region, identify rotten areas, and determine the overall condition of the carrot.

---

## 📌 Features

- Detects carrots in an input image using HSV color masking.
- Identifies potentially rotten regions (dark, gray, or white areas).
- Classifies the carrot as:
  - **Fresh**
  - **Not Fresh**
  - **Not a carrot or unclear**
- Visualizes the results with masks and annotations.

---

## 🧰 Requirements

- Python 3.7+
- OpenCV
- NumPy

### Install Dependencies
pip install opencv-python numpy

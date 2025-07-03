# 🖐️ Hand Gesture Detection System with MobileNetV2
This project recognizes dynamic hand gestures from sequences of frames using a pretrained MobileNetV2 and LSTM. It uses the hand gesture dataset from Kaggle by marusagar for training and evaluation.

----

## 🚀 How to Run

#### 📥 1. Clone the Repository

```bash
git clone <repository_url>
cd Hand_Gesture_Detection_System_With_MobileNetV2
```
📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

----

## 📁 Dataset

  - `Source`: Original dataset ( **763 sequences** ) downloaded from Kaggle via **kagglehub**: [marusagar/hand-gesture-detection-system](https://www.kaggle.com/code/marusagar/hand-gesture-recognition-system)
  - `Classes`: 5 hand gestures (`Left_Swipe`, `Right_Swipe`, `Stop`, `Thumbs_Down`, `Thumbs_Up`).
  - `Structure`: Each gesture class contains subfolders with sequences of .png frames (30 frames per sequence).
  - `Data Split`:
      - `train`: Majority of sequences for training.
      - `val`: Subset of sequences for validation.
      - `test` :
  - `Preprocessing`: Frames are resized to 224×224, converted to RGB, and normalized to [0, 1].

### 📊 Class Distribution :
  <p align="center">
  <img src="https://github.com/user-attachments/assets/b65883b2-c86a-45d3-94d9-dbc797cacfb6"
 width="60%" height="300px" />
  </p>

### 🖼️ Train Sample : 
![Sans titre](https://github.com/user-attachments/assets/bce6a890-20f3-4801-b509-89a36b2d1424)




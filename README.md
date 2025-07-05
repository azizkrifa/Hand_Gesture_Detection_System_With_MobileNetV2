# 🖐️ Hand Gesture Detection System with MobileNetV2
This project recognizes `dynamic hand gestures` from `sequences of frames` using a pretrained `MobileNetV2` for spatial feature extraction and a `GRU (Gated Recurrent Unit)` for modeling `temporal dependencies across frames`. GRU is a type of `recurrent neural network (RNN)` that efficiently captures sequential patterns while being lighter than `LSTM (Long Short-Term Memory)`. The model is trained and evaluated on the hand gesture dataset from `Kaggle`, created by `marusagar`.

----

## 🚀 How to Run

#### 📥 1. Clone the Repository

```bash
git clone https://github.com/azizkrifa/Hand_Gesture_Detection_System_With_MobileNetV2.git
cd Hand_Gesture_Detection_System_With_MobileNetV2
```
#### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

----

## 📁 Dataset

  - `Source`: Original dataset ( **763 sequences with 30 frames for each **  ) downloaded from Kaggle via **kagglehub**: [marusagar/hand-gesture-detection-system](https://www.kaggle.com/code/marusagar/hand-gesture-recognition-system)
  - `Classes`: 5 hand gestures (`Left_Swipe`, `Right_Swipe`, `Stop`, `Thumbs_Down`, `Thumbs_Up`).
  - `Data Split`:
      - `train(643)`: Majority of sequences for training.
      - `val(100)`: Subset of sequences for validation.
      - `test(20)` :
  - `Preprocessing`: Frames are resized to 224×224, converted to RGB, and normalized to [0, 1].

### 📚 Directory structure: 
Each gesture class contains subfolders with sequences of .png frames (30 frames per sequence).

```bash
Dataset/
└── fingers_dataset(split)/
    ├── train/
    │    ├── Left_Swipe/
    │    │     ├── sample_001/
    │    │     │     ├── frame_0001.png
    │    │     │     ├── frame_0002.png
    │    │     │     └── ...
    │    │     ├── sample_002/
    │    │     │     └── ...
    │    │     └── ...
    │    ├── Right_Swipe/
    │    │     └── ...
    │    ├── Stop/
    │    │     └── ...
    │    ├── Thumbs_Down/
    │    │     └── ...
    │    └── Thumbs_Up/
    │          └── ...
    ├── val/
    │    ├── Left_Swipe/
    │    │     └── ...
    │    ├── Right_Swipe/
    │    │     └── ...
    │    └── ...
    └── test/
         ├── Left_Swipe/
         │     └── ...
         └── ...

```


### 📊 Class Distribution(Train Set) :
  <p align="center">
  <img src="https://github.com/user-attachments/assets/b65883b2-c86a-45d3-94d9-dbc797cacfb6"
 width="60%" height="300px" />
  </p>
  
The classes are relatively `balanced`, with counts ranging from `119` to `133` samples each. This balance helps ensure that the model receives sufficient examples from each gesture category during training, which can contribute to `better overall performance` and generalization across different hand gestures.

### 🖼️ Train Sample 

- Originally, each training sequence consists of `30 frames` :
  
![Sans titre](https://github.com/user-attachments/assets/bce6a890-20f3-4801-b509-89a36b2d1424)

- To reduce `computation and speed up training`, we downsample each sequence by selecting every `2nd frame` (i.e., 15 frames per sample) :
  


  

![image](https://github.com/user-attachments/assets/9ddb32f8-f918-4e82-9ce9-8a3e20fd0bda)





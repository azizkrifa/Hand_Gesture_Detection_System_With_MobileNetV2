# 🖐️ Hand Gesture Detection System with MobileNetV2
This project recognizes `dynamic hand gestures` from `sequences of frames` using a pretrained `MobileNetV2` for spatial feature extraction and a `GRU (Gated Recurrent Unit)` for modeling `temporal dependencies across frames`. GRU is a type of `recurrent neural network (RNN)` that efficiently captures sequential patterns while being lighter than `LSTM (Long Short-Term Memory)`. The model is trained and evaluated on the hand gesture dataset from `Kaggle`, created by `marusagar`.

----

## 🧠 Overview

  - Downloaded the Hand Gesture dataset from `KaggleHub` and organized it into `train`, `validation`, and `test` sets.

  - Visualized random `gesture sequences` and `class distributions` to ensure `balanced` data.

  - Preprocessed frames by `resizing`, `normalizing`, and `structuring` them into sequences for `temporal` modeling.

  - Built a `hybrid` deep learning model combining `MobileNetV2 (CNN)` for spatial feature extraction and `Bidirectional GRU` for temporal sequence learning.

  - Trained the model with `early stopping`, `learning rate reduction`, `model checkpointing`, and `CSV logging`.

  - Visualized `training vs. validation` metrics `(accuracy/loss)` to monitor model convergence.

  - Evaluated performance on the test set, reporting `loss`, `accuracy`, `classification report`, and `confusion matrix`.


----

## 🚀 How to Run

#### 📥 1. Clone the Repository

```bash
git clone https://github.com/azizkrifa/Hand_Gesture_Detection_System_With_MobileNetV2.git
```
```bash
cd Hand_Gesture_Detection_System_With_MobileNetV2
```
#### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 🧪 3. Run the Pipeline Step-by-Step

   #####  🧹 3.1: Data Preparation

  Load and preprocess the dataset :

   📄 Run: `data_preprocessing.ipynb`

  ##### 🧠 3.2: Train the Model

  Train the model on the prepared dataset: 
  
   📄 Run: `Training.ipynb`
    
  ##### 📊 3.3: Evaluate the Model

  Evaluate model performance and visualize results: 
  
   📄 Run: `Evaluation.ipynb`
     
-----

## 📁 Dataset

  - `Source`: Original dataset ( **763 sequences with 30 frames for each **  ) downloaded from Kaggle via **kagglehub**: [marusagar/hand-gesture-detection-system](https://www.kaggle.com/code/marusagar/hand-gesture-recognition-system)
  - `Classes`: 5 hand gestures (`Left_Swipe`, `Right_Swipe`, `Stop`, `Thumbs_Down`, `Thumbs_Up`).
  - `Data Split`:
      - `train(643)`: Majority of sequences for training.
      - `val(100)`: Subset of sequences for validation.
      - `test(20)` :
  - `Preprocessing`: Frames are resized to `224×224`, converted to `RGB` and normalized to `[0, 1]`.

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
 width="70%" height="300px" />
  </p>
  
The classes are relatively `balanced`, with counts ranging from `119` to `133` samples each. This balance helps ensure that the model receives sufficient examples from each gesture category during training, which can contribute to `better overall performance` and generalization across different hand gestures.

### 🖼️ Train Sample 

- Originally, each training sequence consists of `30 frames` :
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/bce6a890-20f3-4801-b509-89a36b2d1424">
</p>


- To reduce `computational cost` and `accelerate training`, we downsample each input sequence by selecting every `2nd frame`, resulting in `15 frames per sample`. This decision is based on `prior experiments` using 30-frame sequences, which showed `marginal performance gains` relative to the increased training time and memory consumption. Thus, 15 frames offered a more efficient trade-off between `accuracy` and `computational efficiency`:

<p align="center">
  <img src="https://github.com/user-attachments/assets/468927c7-dcb4-4e25-b77a-3dd16e1d8133" >
</p>

-----
## ⚙️ Model Setup

- **Base model**: MobileNetV2 (pretrained on ImageNet, `include_top=False`, fine-tuned).

- **Input**: Sequence of `15 frames`, each of size 224×224×3 (**RGB**).

- **Architecture**:

  - `TimeDistributed(MobileNetV2)`
  - `TimeDistributed(BatchNormalization)`
  - `TimeDistributed(GlobalAveragePooling2D)`
  - **Bidirectional(GRU)**(128)
  - **Dropout**(0.25)
  - **Dense**(128, **ReLU**) + **Dropout**(0.25)
  - **Dense**(5, **Softmax**)

- **Optimizer**: Adam (`learning rate = 1e-4`)

- **Loss function**: Sparse Categorical Crossentropy

- **Metrics**: Accuracy

-----

## 🏋️‍♂️ Training Details

- **Epochs**: 40

- **Batch size**: 4 sequences per step (`each sequence = 15 frames`).

- **Training data**: Preprocessed image sequences (`15-frame inputs`), optionally augmented.

- **Validation data**: Clean validation set (no augmentation).

- **Callbacks**:

  - **Checkpoints**: Best model saved based on validation accuracy (`ModelCheckpoint`).
  - **Early stopping**: Stops training if validation loss does not improve for 10 consecutive epochs (`EarlyStopping` with `restore_best_weights=True`).
  - **ReduceLROnPlateau**: Reduces learning rate by `factor of 0.5` if validation loss stagnates for`3 epochs` (minimum LR = 1e-7).
  - **CSVLogger**: Saves training history to CSV (`training_log.csv`).
 
-----

## 📈 Training Curves

Plots of **training** & **validation** (`accuracy / loss`) across **30** epochs :

<img width="1388" height="490" alt="Sans titre" src="https://github.com/user-attachments/assets/587d8eba-8253-4f75-9c38-a341186276e3" />


-  The training and validation curves show consistent improvement over epochs. The model achieves a validation accuracy of `~89%`, with decreasing loss and no major signs of overfitting — indicating good generalization to unseen data.

---

## 📊 Evaluation Results

The test set consists of only `20 samples`. Although the model achieved a test accuracy of `90%`, such a small sample size limits the `reliability` and `generalizability` of the result. Therefore, we also present the `validation` set metrics, which are based on a `larger dataset` and provide a more stable and informative evaluation of the model’s performance.

<div align="center">
<table>
  <tr>
    <th>val set</th>
    <th>test set</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a2d1114e-003f-4fbc-bf44-ba09d3b48dff" width="400px" height="300px" ></td>
    <td><img src="https://github.com/user-attachments/assets/a06a0ed3-e100-4446-8473-87471894b5b2" width="400"  height="300px"></td>
  </tr>
</table>
</div>

➡️ The `Outputs` folder contains the `confusion matrix` for both the `validation` and `test` sets, along with the saved best model (`best_model.keras`).


----


## 🎯 Sample Predictions

<div align="center">
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/a4c03611-49f7-4dc0-9880-6a3b417b5137" />
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/26e6a1d1-c984-4f44-8fbc-12fef89675d9" />
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/810e0bf8-fe3f-4735-8927-13d077cead66" />
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/c9284e6a-e5ee-4bde-9716-0c21252716bf" />
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/c683b44c-ce7f-4417-9a55-84a3566d1177" />
  <img width="500" height="400" alt="Sans titre" src="https://github.com/user-attachments/assets/b5f492fb-0e2a-4b05-909e-385a69867389" />
</div>











  







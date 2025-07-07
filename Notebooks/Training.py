# ---
# 📦 Import Necessary Libraries
# ---
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import nbimporter
from data_preprocessing import DATA_DIR

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D ,BatchNormalization ,Flatten ,Bidirectional , GRU , Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger ,ReduceLROnPlateau

# ---
# 🗂️ Load Frame Sequences and Create TensorFlow Dataset
# ---

def load_sequence_frames(sample_path, seq_len=15, step = 2, size=(224, 224)):
    frames = []
    image_files = sorted(sample_path.glob('*.png'))
    selected_files = image_files[::step][:seq_len]

    for img_path in selected_files:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        frames.append(img)

    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames

def data_generator(root_dir):
    root = Path(root_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples, labels = [], []
    for c in classes:
        for sample_folder in (root / c).iterdir():
            if sample_folder.is_dir():
                samples.append(sample_folder)
                labels.append(class_to_idx[c])

    def gen():
        for sample_path, label in zip(samples, labels):
            seq = load_sequence_frames(sample_path)
            yield seq, label

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(15, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

# ---
# 📊 Create Training and Validation Datasets
# ---
train_dataset = data_generator(f"{DATA_DIR}/train")\
    .shuffle(100)\
    .batch(4)\
    .repeat()\
    .prefetch(tf.data.AUTOTUNE)

val_dataset = data_generator(f"{DATA_DIR}/val")\
    .batch(4)\
    .prefetch(tf.data.AUTOTUNE)

# ---
# 🧠 Build the MobileNetV2 + Bidirectional GRU Model
# ---
num_classes = 5
sequence_length = 15
image_size = (224, 224, 3)

inputs = Input(shape=(sequence_length, *image_size))
cnn_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=image_size)
cnn_base.trainable = True
x = TimeDistributed(cnn_base)(inputs)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = Bidirectional(GRU(128), kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.25)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.25)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---
# 🔢 Calculate Steps per Epoch and Validation Steps
# ---
original_train_dataset = data_generator(f"{DATA_DIR}/train")
num_train_samples = sum(1 for _ in original_train_dataset)

original_val_dataset = data_generator(f"{DATA_DIR}/val")
num_val_samples = sum(1 for _ in original_val_dataset)

steps_per_epoch = num_train_samples // 4
validation_steps = num_val_samples // 4

# ---
# 🏋️ Train the Model with Callbacks
# ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(f"{DATA_DIR}/Outputs/best_model.keras", save_best_only=True, monitor='val_accuracy')
csv_logger = CSVLogger(f"{DATA_DIR}/Outputs/training_log.csv", append=False)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-7
)

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=40,
    callbacks=[early_stop, checkpoint , csv_logger , reduce_lr]
)

# ---
# 📈 Visualize Training & Validation Accuracy and Loss
# ---
history = pd.read_csv(f"{DATA_DIR}/Outputs/training_log.csv")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

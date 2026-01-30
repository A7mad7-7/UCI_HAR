import numpy as np
import matplotlib.pyplot as plt
import os

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Load Processed Data
# ==========================================
print("Loading processed data... 📂")

data_dir = "processed_data"

X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_cv = np.load(os.path.join(data_dir, 'X_cv.npy'))
y_cv = np.load(os.path.join(data_dir, 'y_cv.npy'))

print(f"✅ Training set:   {X_train.shape[0]} samples")
print(f"✅ Validation set: {X_cv.shape[0]} samples")

# ==========================================
# 2. Build Model
# ==========================================
print("\nBuilding model... 🏗️")

model = Sequential([
    Dense(128, activation='relu', input_shape=(561,)),
    Dense(64, activation='relu'),
    Dense(6, activation='linear')  # 6 classes, logits output
])

model.compile(
    optimizer=Adam(learning_rate=0.0006),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 3. Train Model
# ==========================================
print("\nTraining model... 🚀")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_cv, y_cv),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 4. Save Model
# ==========================================
model_path = "model.keras"
model.save(model_path)
print(f"\n✅ Model saved to '{model_path}'")

# ==========================================
# 5. Plot Learning Curves
# ==========================================
print("\nGenerating learning curves... 📈")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss Plot
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss vs Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy Plot
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy vs Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plot_path = "learning_curves.png"
plt.savefig(plot_path, dpi=150)
print(f"✅ Learning curves saved to '{plot_path}'")

plt.show()

# ==========================================
# 6. Final Training Summary
# ==========================================
print("\n" + "=" * 50)
print("📊 TRAINING SUMMARY")
print("=" * 50)
print(f"🔹 Total epochs run: {len(history.history['loss'])}")
print(f"🔹 Final training loss:     {history.history['loss'][-1]:.4f}")
print(f"🔹 Final validation loss:   {history.history['val_loss'][-1]:.4f}")
print(f"🔹 Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"🔹 Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print("=" * 50)

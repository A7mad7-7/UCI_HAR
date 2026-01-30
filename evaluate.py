import numpy as np
import os

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ==========================================
# 1. Load Test Data
# ==========================================
print("Loading test data... 📂")

data_dir = "processed_data"
model_path = "model.keras"

X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

print(f"✅ Test set: {X_test.shape[0]} samples")

# ==========================================
# 2. Load Saved Model
# ==========================================
print(f"\nLoading model from '{model_path}'... 🔄")

model = load_model(model_path)
print("✅ Model loaded successfully!")

# ==========================================
# 3. Evaluate on Test Set
# ==========================================
print("\nEvaluating on test set... 🧪")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

# ==========================================
# 4. Results
# ==========================================
print("\n" + "=" * 50)
print("📊 TEST RESULTS")
print("=" * 50)
print(f"🔹 Test Loss:     {test_loss:.4f}")
print(f"🔹 Test Accuracy: {test_accuracy*100:.2f}%")
print("=" * 50)

# ==========================================
# 5. Predictions (Optional Details)
# ==========================================
# Get predictions
y_pred_logits = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_logits, axis=1)

# Activity labels
activity_labels = [
    "WALKING",
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

# Per-class accuracy
print("\n📊 PER-CLASS ACCURACY:")
print("-" * 40)
for i, label in enumerate(activity_labels):
    mask = (y_test == i)
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == i).mean() * 100
        print(f"  {label:<20}: {class_acc:.1f}%")
print("-" * 40)

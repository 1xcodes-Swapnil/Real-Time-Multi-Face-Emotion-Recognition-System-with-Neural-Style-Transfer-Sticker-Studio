import numpy as np
import cv2, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
from grad_cam import GradCAM

MODEL_PATH = os.path.join(os.path.dirname(__file__),
    'models', 'saved_model', 'emotion_model.h5')

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"Input : {model.input_shape}")
print(f"Output: {model.output_shape}")

# res_conv4 confirmed present — test GradCAM
gc = GradCAM(model, "res_conv4")

# Use a real-looking test image (gradient pattern, not zeros)
test_img = np.random.rand(1, 48, 48, 1).astype(np.float32)
heatmap  = gc.compute(test_img, class_idx=3)
print(f"Heatmap min/max: {heatmap.min()} / {heatmap.max()}")
print("Result:", "OK — colors showing!" if heatmap.max() > 10 else "STILL BLACK")

# Also check with a brighter input
test_img2 = np.ones((1, 48, 48, 1), dtype=np.float32) * 0.5
heatmap2  = gc.compute(test_img2, class_idx=3)
print(f"Heatmap2 min/max: {heatmap2.min()} / {heatmap2.max()}")

# Save test heatmap
cv2.imwrite('test_heatmap.png', heatmap)
print("Saved test_heatmap.png — check if it has colors")
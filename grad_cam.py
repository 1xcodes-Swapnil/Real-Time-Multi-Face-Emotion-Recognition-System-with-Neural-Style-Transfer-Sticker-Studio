"""
grad_cam.py
GradCAM using full model — works with residual connections (Add layers).
Does NOT split the model — uses a functional sub-model approach instead.
"""
import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str):
        self.model     = model
        self.layer_name = last_conv_layer_name
        self._grad_model = None
        self._ready      = False
        self._build()

    def _build(self):
        try:
            # Build a model that outputs BOTH the conv layer AND predictions
            # This works with residual/skip connections unlike the split approach
            conv_layer = self.model.get_layer(self.layer_name)
            self._grad_model = tf.keras.Model(
                inputs  = self.model.inputs,
                outputs = [conv_layer.output, self.model.output]
            )
            self._ready = True
            print(f"[GradCAM] Ready — targeting '{self.layer_name}'")
        except Exception as e:
            print(f"[GradCAM] Build failed: {e}")
            self._ready = False

    def compute(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        img_array : preprocessed input (1, H, W, C)
        Returns   : BGR heatmap uint8 same size as input face
        """
        if not self._ready:
            return np.zeros((48, 48, 3), dtype=np.uint8)

        try:
            img_tensor = tf.cast(img_array, tf.float32)

            with tf.GradientTape() as tape:
                # Watch conv output AND compute predictions in one forward pass
                conv_out, predictions = self._grad_model(
                    img_tensor, training=False)
                tape.watch(conv_out)

                if class_idx is None:
                    class_idx = int(tf.argmax(predictions[0]))
                loss = predictions[:, class_idx]

            # Gradients of class score w.r.t conv feature maps
            grads = tape.gradient(loss, conv_out)

            if grads is None:
                print("[GradCAM] No gradients — trying activation fallback")
                # Fallback: use raw activation map
                conv_np  = conv_out[0].numpy()
                heatmap  = np.mean(np.abs(conv_np), axis=-1)
            else:
                # Pool gradients spatially → importance weight per channel
                pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
                conv_np = conv_out[0].numpy()

                # Weighted sum of feature maps
                heatmap = np.zeros(conv_np.shape[:2], dtype=np.float32)
                for i, w in enumerate(pooled):
                    heatmap += w * conv_np[:, :, i]

                # ReLU — keep only positive activations
                heatmap = np.maximum(heatmap, 0)

                # If still zero, fall back to mean absolute activation
                if heatmap.max() < 1e-8:
                    heatmap = np.mean(np.abs(conv_np), axis=-1)

            # Normalize to [0, 1]
            if heatmap.max() > 1e-8:
                heatmap = heatmap / heatmap.max()
            else:
                return np.zeros((48, 48, 3), dtype=np.uint8)

            # Gamma correction to boost mid-range values
            heatmap = np.power(heatmap, 0.6)

            # Resize and colorize
            h = cv2.resize(heatmap, (48, 48))
            return cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)

        except Exception as e:
            print(f"[GradCAM] Error: {e}")
            return np.zeros((48, 48, 3), dtype=np.uint8)

    def overlay(self, original_bgr, heatmap_bgr, alpha=0.5):
        h, w = original_bgr.shape[:2]
        hm   = cv2.resize(heatmap_bgr, (w, h))
        return cv2.addWeighted(original_bgr, 1-alpha, hm, alpha, 0)
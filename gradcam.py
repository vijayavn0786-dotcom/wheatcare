"""
gradcam.py - Explainable AI (XAI) using Gradient-weighted Class Activation Mapping (Grad-CAM)
Specifically adapted for the Wheat Leaf Disease Custom CNN.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib

# Set non-interactive writable cache directory for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))


class GradCAMExplainer:
    """Computes Grad-CAM heatmaps and visual explanations for Keras CNN models."""

    def __init__(self, model, last_conv_layer_name="conv2d_7", img_size=(224, 224)):
        self.model = model
        self.img_size = img_size
        self.last_conv_layer_name = last_conv_layer_name

        # Find target layer index
        matching_indices = [
            i for i, layer in enumerate(self.model.layers)
            if layer.name == last_conv_layer_name
        ]

        if not matching_indices:
            # Fallback: look for the last Conv2D layer in the model
            conv_layers = [
                (i, layer.name) for i, layer in enumerate(self.model.layers)
                if "conv2d" in layer.name.lower()
            ]
            if conv_layers:
                target_idx = conv_layers[-1][0]
                self.last_conv_layer_name = conv_layers[-1][1]
            else:
                raise ValueError("No Conv2D layer found in the model for Grad-CAM.")
        else:
            target_idx = matching_indices[0]

        self.target_idx = target_idx

        # Decompose Sequential model into feature extractor and classifier
        self.feature_extractor = tf.keras.Sequential(self.model.layers[: target_idx + 1])
        self.classifier = tf.keras.Sequential(self.model.layers[target_idx + 1 :])

    def generate_heatmap(self, preprocessed_image_array, target_class_index=None):
        """
        Calculates raw Grad-CAM activation heatmap normalized to [0, 1].

        Args:
            preprocessed_image_array: np.ndarray of shape (1, 224, 224, 3) in range [0, 1]
            target_class_index: Optional integer for target class. If None, top prediction is used.

        Returns:
            np.ndarray: 2D heatmap of shape (H_conv, W_conv) normalized in [0, 1]
        """
        tensor_input = tf.convert_to_tensor(preprocessed_image_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs = self.feature_extractor(tensor_input)
            tape.watch(conv_outputs)
            predictions = self.classifier(conv_outputs)

            if target_class_index is None:
                target_class_index = tf.argmax(predictions[0])

            loss = predictions[:, target_class_index]

        # Calculate gradients of the predicted class score w.r.t feature maps
        gradients = tape.gradient(loss, conv_outputs)
        if gradients is None:
            # Return uniform activation fallback if gradient computation fails
            return np.zeros((self.img_size[0] // 8, self.img_size[1] // 8), dtype=np.float32)

        # Global average pooling over spatial dimensions
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

        # Weighted sum of feature maps
        conv_output_0 = conv_outputs[0]
        cam = tf.reduce_sum(conv_output_0 * pooled_gradients, axis=-1).numpy()

        # Apply ReLU to keep only features that contribute positively to the class
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        original_pil_image,
        cam_heatmap,
        colormap_name="jet",
        alpha=0.45,
    ):
        """
        Superimposes the Grad-CAM heatmap onto the original image.

        Args:
            original_pil_image: PIL.Image in RGB
            cam_heatmap: 2D np.ndarray in [0, 1]
            colormap_name: Matplotlib colormap ('jet', 'turbo', 'inferno', etc.)
            alpha: Transparency factor for overlay blend [0 to 1]

        Returns:
            tuple: (superimposed_pil_image, colored_heatmap_pil_image, infection_ratio)
        """
        # Resize original image to target size
        orig_resized = original_pil_image.resize(self.img_size, Image.Resampling.LANCZOS).convert("RGB")

        # Upscale heatmap to image resolution using bilinear interpolation
        heatmap_uint8 = np.uint8(255 * cam_heatmap)
        colormap = matplotlib.colormaps.get(colormap_name, matplotlib.colormaps["jet"])
        colormap_colors = colormap(np.arange(256))[:, :3]  # RGB floats [0, 1]
        colored_map = colormap_colors[heatmap_uint8]
        colored_map_uint8 = (colored_map * 255).astype(np.uint8)

        heatmap_pil = Image.fromarray(colored_map_uint8).resize(self.img_size, Image.Resampling.BILINEAR)

        # Superimpose
        superimposed_pil = Image.blend(orig_resized, heatmap_pil, alpha=alpha)

        # Estimate infected/salient area ratio (threshold > 0.35 of maximum activation)
        salient_pixels = np.sum(cam_heatmap > 0.35)
        total_pixels = cam_heatmap.size
        activation_ratio = float(salient_pixels / total_pixels)

        return superimposed_pil, heatmap_pil, activation_ratio

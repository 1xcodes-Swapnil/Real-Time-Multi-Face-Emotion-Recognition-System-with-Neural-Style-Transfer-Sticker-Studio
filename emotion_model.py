"""
emotion_model.py
Two architectures for emotion recognition:
  - build_cnn()       : Improved CNN with residual blocks (grayscale 48x48)
  - build_mobilenet() : MobileNetV2 fine-tuned (RGB 224x224)

Grad-CAM target layer: 'res_conv4' for CNN, use main_app LAST_CONV setting
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications

NUM_CLASSES          = 7
LAST_CONV_LAYER_NAME = "res_conv4"   # used by Grad-CAM


def _residual_block(x, filters, name_prefix):
    """Small residual block: 2x Conv + skip connection."""
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding="same", use_bias=False,
                      name=f"{name_prefix}_c1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3,3), padding="same", use_bias=False,
                      name=f"{name_prefix}_c2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    # Match channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def build_cnn(input_shape=(48, 48, 1), num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    Improved CNN with residual connections.
    Deeper than original, better generalization.
    Grad-CAM target: 'res_conv4'
    """
    inp = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, (3,3), padding="same", use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu")(x)

    # Block 1 — 32 filters
    x = _residual_block(x, 32, "res_block1")
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2 — 64 filters
    x = _residual_block(x, 64, "res_block2")
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3 — 128 filters  ← GradCAM hooks here
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False,
                      name="res_conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same", use_bias=False,
                      name="res_conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4 — 256 filters
    x = layers.Conv2D(256, (3,3), padding="same", use_bias=False,
                      name="res_conv5")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier head
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inp, out, name="EmotionCNN_Residual")


def build_mobilenet(input_shape=(224, 224, 3), num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    MobileNetV2 transfer learning — improved head.
    """
    base = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="EmotionMobileNetV2")


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
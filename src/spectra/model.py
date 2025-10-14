"""TensorFlow model for classifying stellar spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass
class ModelConfig:
    num_classes: int
    input_length: int
    filters: Sequence[int] = (32, 64, 128)
    kernel_sizes: Sequence[int] = (7, 5, 3)
    dropout_rate: float = 0.2
    dense_units: int = 128


class SpectralClassifier(tf.keras.Model):
    """A lightweight 1D CNN for stellar spectral classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(config.input_length, 1)),
                *[
                    _conv_block(filters, kernel)
                    for filters, kernel in zip(config.filters, config.kernel_sizes)
                ],
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(config.dense_units, activation="relu"),
                tf.keras.layers.Dropout(config.dropout_rate),
                tf.keras.layers.Dense(config.num_classes, activation="softmax"),
            ]
        )

    def call(self, inputs, training=False):
        if inputs.shape.rank == 2:
            inputs = tf.expand_dims(inputs, axis=-1)
        features = self.feature_extractor(inputs, training=training)
        return self.classifier(features, training=training)

    def compile_for_training(self, learning_rate: float = 1e-3) -> None:
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


def _conv_block(filters: int, kernel_size: int) -> tf.keras.layers.Layer:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(filters, kernel_size, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool1D(pool_size=2),
        ]
    )

"""Model definition for CNN-LSTM-Attention malware classifier."""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf


class AdditiveAttention(tf.keras.layers.Layer):
    """Additive attention mechanism for sequence data.

    This layer computes context vectors as described in Bahdanau et al. (2014)
    and is suitable for sequence outputs from recurrent networks. Given a
    sequence of hidden states ``H`` with shape ``(batch, timesteps, features)``
    it learns to assign attention weights that sum to one across timesteps and
    returns the weighted context vector ``(batch, features)``.
    """

    def __init__(self, units: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.W_h: tf.keras.layers.Layer | None = None
        self.W_s: tf.keras.layers.Layer | None = None
        self.v: tf.keras.layers.Layer | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        feature_dim = int(input_shape[-1])
        self.W_h = self.add_weight(
            name="W_h",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_s = self.add_weight(
            name="W_s",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs shape: (batch, timesteps, features)
        score = tf.tanh(tf.tensordot(inputs, self.W_h, axes=[2, 0]) + self.W_s)
        score = tf.tensordot(score, self.v, axes=[2, 0])
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        return context

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_cnn_lstm_attention_model(
    input_dim: int = 2381,
    *,
    conv_filters: tuple[int, int] = (64, 128),
    kernel_sizes: tuple[int, int] = (5, 3),
    pool_sizes: tuple[int, int] = (2, 2),
    lstm_units: int = 128,
    attention_units: int = 128,
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build the default CNN→BiLSTM→Attention malware classifier.

    Parameters
    ----------
    input_dim:
        Dimensionality of the EMBER vectorized features (default: 2381).
    conv_filters, kernel_sizes, pool_sizes:
        Hyperparameters for the two Conv1D + MaxPool blocks.
    lstm_units:
        Number of units in the bidirectional LSTM layer.
    attention_units:
        Dimensionality of the additive attention layer.
    dense_units:
        Units in the dense layer prior to the output.
    dropout_rate:
        Dropout applied before the final classification layer.
    learning_rate:
        Learning rate for the Adam optimizer.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training on binary classification tasks.
    """

    inputs = tf.keras.Input(shape=(input_dim,), name="ember_features")
    x = tf.keras.layers.Reshape((input_dim, 1), name="reshape_inputs")(inputs)

    for idx, (filters, kernel_size, pool_size) in enumerate(
        zip(conv_filters, kernel_sizes, pool_sizes)
    ):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_{idx+1}",
        )(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            name=f"max_pool_{idx+1}",
        )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm",
    )(x)
    context = AdditiveAttention(units=attention_units, name="attention")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_relu")(context)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_attention")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


__all__ = [
    "AdditiveAttention",
    "build_cnn_lstm_attention_model",
]

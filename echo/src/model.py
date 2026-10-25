"""The LSTM network (TensorFlow / Keras).

How it works, in plain language
-------------------------------
1. Input:  the last ``seq_len`` musical events, each an integer id.
2. Embedding: each id becomes a small vector of numbers the network can learn
   (similar-sounding events end up with similar vectors).
3. LSTM layer(s): read the sequence in order and keep a memory of what has
   happened so far - this is what lets the model pick up melodic patterns.
4. Dropout: randomly switches off some units during training so the model
   does not simply memorise the songs.
5. Dense + softmax: outputs a probability for every possible next event.

Training teaches it to give high probability to the event that really came
next in the music (sparse categorical cross-entropy loss, Adam optimiser).
"""

from __future__ import annotations

import config
from src.utils import require_module


def build_model(vocab_size: int, cfg: config.ModelConfig):
    """Create and compile the LSTM for a vocabulary of ``vocab_size`` events."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    cfg.validate()
    tf = require_module("tensorflow")
    keras = tf.keras

    inputs = keras.Input(shape=(cfg.seq_len,), dtype="int32", name="event_ids")
    x = keras.layers.Embedding(vocab_size, cfg.embedding_dim, name="event_embedding")(inputs)
    for layer_number in range(1, cfg.num_layers + 1):
        is_last = layer_number == cfg.num_layers
        # recurrent_dropout is left at 0 so TensorFlow can use its fast LSTM kernel.
        x = keras.layers.LSTM(
            cfg.lstm_units, return_sequences=not is_last, name=f"lstm_{layer_number}"
        )(x)
        x = keras.layers.Dropout(cfg.dropout, name=f"dropout_{layer_number}")(x)
    outputs = keras.layers.Dense(vocab_size, activation="softmax", name="next_event")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="echo_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

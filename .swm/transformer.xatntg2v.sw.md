---
title: Transformer
---
<SwmSnippet path="/BaseTransformer/BaseTransformerModules/transformer.py" line="14">

---

&nbsp;

```python
def build_transformer(input_shape, d_model, num_heads, dff, num_decoder_layers, rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)

    pos_encoding = positional_encoding(input_shape[0], d_model)
    x += pos_encoding[:, :input_shape[0], :]

    encoder_output = TransformerEncoderLayer(d_model, num_heads, dff, rate)(x, training=False, mask=None)

    decoder_output = encoder_output
    for _ in range(num_decoder_layers):
        decoder_output, _, _ = TransformerDecoderLayer(d_model, num_heads, dff, rate)(
            decoder_output, encoder_output, training=False, look_ahead_mask=None, padding_mask=None)

    x = tf.keras.layers.Flatten()(encoder_output)
    x = tf.keras.layers.Dense(6, activation='linear')(x)  # Output layer for all 6 features

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: TrainBaseTSF
---
<SwmSnippet path="/BaseTransformer/BaseTransformerUse/trainTSM.py" line="17">

---

&nbsp;

```python
# Download stock data
data = yf.download("BA")

# Apply purging and embargoing
data = purge_data(data, 2) # Purge data older than 2 years
data = embargo_data(data, 5) # Embargo the most recent 5 days

# Scale data
scaled_data, scaler = scale_data(data)

# Create sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into training and testing sets
X_train, y_train, X_test, y_test, train_size = train_test_split(X, y, 0.8)
```

---

</SwmSnippet>

Train

<SwmSnippet path="/BaseTransformer/BaseTransformerUse/trainTSM.py" line="40">

---

&nbsp;

```python
transformer = build_transformer(
    input_shape=(SEQ_LENGTH, X_train.shape[-1]), 
    d_model=d_model, 
    num_heads=num_heads, 
    dff=dff, 
    num_decoder_layers=num_decoder_layers, 
    rate=dropout_rate)

transformer.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')

transformer.summary()

history = transformer.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

test_loss = transformer.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

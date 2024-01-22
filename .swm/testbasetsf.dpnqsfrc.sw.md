---
title: TestBaseTSF
---
<SwmSnippet path="/BaseTransformer/BaseTransformerUse/testTSM.py" line="22">

---

&nbsp;

```python
transformer = tf.keras.models.load_model("models/baseTSF", custom_objects=custom_objects)

# Load the MinMaxScaler
scaler = joblib.load("models/baseTSF/scaler.save")

# After training, make predictions
predictions = transformer.predict(X_test)

# Directly apply inverse scaling
inverse_scaled_predictions = scaler.inverse_transform(predictions)

print(inverse_scaled_predictions[:5])

actual = scaler.inverse_transform(scaled_data)

# Plot the predictions for closing vs the actual values for closing
import matplotlib.pyplot as plt

plt.plot(inverse_scaled_predictions[:, 3], label='Predicted')
plt.plot(actual[train_size + SEQ_LENGTH:, 3], label='Actual')
plt.legend()
plt.show()
```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: Data Preprocessing
---
<SwmSnippet path="/BaseTransformer/BaseTransformerModules/utils.py" line="6">

---

&nbsp;

```python
def purge_data(data, years):
    """ Remove data older than a specified number of years. """
    purge_date = pd.Timestamp('today') - pd.DateOffset(years=years)
    return data[data.index >= purge_date]
```

---

</SwmSnippet>

Embargo

<SwmSnippet path="/BaseTransformer/BaseTransformerModules/utils.py" line="11">

---

&nbsp;

```python
def embargo_data(data, days):
    """ Exclude the most recent specified number of days. """
    return data[:-days]
```

---

</SwmSnippet>

Scale

<SwmSnippet path="/BaseTransformer/BaseTransformerModules/utils.py" line="15">

---

&nbsp;

```python
def scale_data(data):
    """ Scale data using MinMaxScaler. """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler
```

---

</SwmSnippet>

e

<SwmSnippet path="/BaseTransformer/BaseTransformerModules/utils.py" line="20">

---

&nbsp;

```python
def create_sequences(data, seq_length):
    """ Create sequences from the data. """
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
```

---

</SwmSnippet>

e

<SwmSnippet path="/BaseTransformer/BaseTransformerModules/utils.py" line="30">

---

&nbsp;

```python
def train_test_split(X, y, train_split):
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, y_train, X_test, y_test, train_size
```

---

</SwmSnippet>

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: Hyperparameters
---
<SwmSnippet path="/BaseTransformer/BaseTransformerModules/hyperparameters.py" line="1">

---

&nbsp;

```python
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
N_LAYERS = 6
SEQ_LENGTH = 96
```

---

</SwmSnippet>

### **Look-back Window Considerations**

- Existing solutions tend to overfit temporal noises instead of extracting temporal information if given a longer sequence, and the **input size 96 is exactly suitable for most Transformers.**
- While the temporal dynamics in the look-back window significantly impact the forecasting accuracy of short-term time series forecasting, **we hypothesize** that long-term forecasting depends on whether models can capture the trend and periodicity well only. That is, the \*farther the forecasting horizon, the less impact the look-back window itself has.
- Similar to the observations from previous studies \[27, 30\], existing Transformer-based models’ performance deteriorates or stays stable when the look-back window size increases. In contrast, the performances of all LTSF-Linear are **significantly boosted** with the increase of look-back window size.
- When the \[\[Look-Back Window (Input Length)|input length\]\] is 96 steps, and the \[\[Output Horizon\]\] is 336 steps, Transformers \[28, 30, 31\] fail to capture the scale and bias of the future data on Electricity and ETTh2. Moreover, they can hardly predict a proper trend on aperiodic data such as Exchange-Rate. These phenomena further indicate the inadequacy of existing Transformer-based solutions for the LTSF task.
- We notice that increasing the size of the look-back window positively affected the performance of the LSTM (cf. Figure 12). However, there was no noticeable change for the HFformer (cf. Figure 11).&nbsp;

### **Target Time Series Considerations**

The Autoformer and FEDformer require an additional input feature which is the date and time timestamp, as both of these models use a time embedding. The frequency of the target time series (e.g., seconds, minutes, and hours) is used as a hyperparameter.

### **Forecast Horizon**

### **Layer Number**

One of the key advantages Transformer holds in these fields is being able to increase prediction power through increasing model size. Usually, the model capacity is controlled by Transformer’s layer number, which is commonly set between 12 to 128. Yet as shown in the experiments of Table 3, when we compare the prediction result with different Transformer models with various numbers of layers, the Transformer with 3 to 6 layers often achieves better results. It raises a question about how to design a proper Transformer architecture with deeper layers to increase the model’s capacity and achieve better forecasting performance.

### **Miscellaneous**

We conduct the manual hyperparameter tuning for the best settings: batch size as 32, sequence length as 16, regarding a duration of 16 days, and for the transformer, we set the embedding dimension as 256 and employ 12 attention heads. We use the Adam optimizer with a default learning rate of 0.001. We conduct 10 training epochs for the decentralized SOLO method while conducting 10 global rounds with a single local epoch for both FedAtt and FedAvg.

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: Additions to Consider
---
 1. **LIKELY: Integration of Hierarchical Temporal Features**: Modify the transformer to incorporate global temporal information, such as hierarchical timestamps (week, month, year), as well as agnostic timestamps (holidays and events). This could be achieved by introducing additional input channels or embedding layers specifically designed to encode this kind of temporal information, which is often crucial in financial time series.

 2. **LIKELY: Seasonality and Periodicity Modules**: Implement modules within the transformer that can automatically detect and adapt to seasonal patterns. This might involve integrating components similar to those found in exponential smoothing models, which are adept at capturing such patterns.

 3. **Sparse Attention Mechanisms**: To handle the complexity issue, incorporate sparse attention mechanisms. These could reduce the computational complexity from O(L^2) to something more manageable, enabling the processing of longer sequences without a substantial increase in computational demand.

 4. **LIKELY: Online Normalization Integration**: Embed an online normalization process within the transformer architecture to normalize data in real-time. This is particularly valuable in financial time series where data properties can change rapidly.

 5. **Linear Decoder Enhancement**: Following insights from existing research, consider replacing the Transformer's decoder with a linear decoder for certain tasks, potentially improving performance metrics like the R2 score.

 6. **LIKELY: Spiking Activation Functions**: Experiment with spiking activation functions within the transformer layers. These could offer benefits in modeling the rapid changes often seen in financial data.

 7. **LIKELY: Incorporating Inductive Biases**: Design the transformer architecture with inductive biases that are specific to time series data. This could involve modifications that make the transformer more sensitive to the unique characteristics of financial time series, like auto-correlation and heteroskedasticity.

 8. **Adversarial Training Procedures**: Implement GAN-style adversarial training procedures to improve the stability and performance of the transformer model, especially in tasks like anomaly detection and trend prediction.

 9. **Frequency-Aware Modeling**: Tailor the transformer model to better handle the varying frequencies of financial time series data, from high-frequency (milliseconds) to low-frequency (days). This might involve adaptive sampling or frequency-specific attention mechanisms.

10. **Enhanced Positional Encoding**: Develop advanced positional encoding techniques that better capture the temporal dynamics of financial time series. This could involve more sophisticated embeddings that account for the non-linear nature of time in financial markets.

11. **LIKELY: Temporal Dependency Modeling**: Refine the model's ability to capture long-term dependencies and the weakening of these dependencies over time. This could involve attention mechanisms that focus on relative time lags or incorporate insights from methods like ARIMA or exponential smoothing.&nbsp;

12. **Noise and Overfitting Mitigation**: Implement strategies within the transformer architecture to reduce sensitivity to noise and prevent overfitting, which are critical in financial time series forecasting.

---

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

- Global temporal information, such as hierarchical timestamps (week, month, year) and agnostic timestamps (holidays and events), is also informative \[30\].

- Automatically extracting seasonal patterns has been proved to be critical for the success of forecasting \[5, 6, 36\].

  - However, the vanilla attention mechanism is unlikely able to learn these required periodic dependencies without any in-built prior structure.
  - \[\[ETSformer – Exponential Smoothing Transformers for Time-series Forecasting\]\]

- Sparsity to help with complexity

  - **Is efficiency really a top-level priority?**
  - Section in \[\[Are Transformers Effective for Time Series Forecasting\]\]Failure of Transformers in TSF
  - Novel Transformer designs \[27\] have focused on reducing the O(L^2) complexity by using causal convolutions or sparse bias in the Attention module to process longer sequences.

- Online Normalization

- Architectural changes

  - We notice from Figure 10 that a major improvement in the R2 score is achieved by replacing the Transformer decoder with a linear decoder. Adding a spiking activation and removing the positional encoding result in marginal improvements.

- Anomaly Detection

  - Transformer based architecture also benefits the time series anomaly detection task with the ability to model temporal dependency, which brings high detection quality
  - GAN style adversarial training procedure is designed by two Transformer encoders and two Transformer decoders to gain stability. Ablation study shows that, if Transformer-based encoder-decoder is replaced, F1 score drops nearly 11%, indicating the effect of Transformer architecture on time series anomaly detection.&nbsp;
  - Transformer is proved to be effective in various time series classification tasks due to its prominent capability in capturing long-term dependency

- Incorporate Inductive Biases

  - One future direction is to consider more effective ways to induce inductive biases into Transformers based on the understanding of time series data and characteristics of specific tasks.

  &nbsp;

- One future direction is to consider more architecture-level designs for Transformers specifically optimized for time series data and tasks.

&nbsp;

&nbsp;

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

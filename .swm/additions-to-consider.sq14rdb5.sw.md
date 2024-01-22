---
title: Additions to Consider
---
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

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

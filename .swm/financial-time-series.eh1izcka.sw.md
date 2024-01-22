---
title: Financial Time Series
---
1. **Frequency Range in Trading Strategies**: Trading strategies in financial time series (FTS) vary in frequency, typically from milliseconds to days. However, minute-frequency data is most commonly studied due to the high cost of millisecond data.

2. **FTS Characteristics**: FTS are characterized by properties like frequency, auto-correlation, heteroskedasticity, drift, and seasonality, making them complex for analysis and forecasting.

3. **Use of LSTM and Other Neural Networks**: LSTMs are widely successful in minute-frequency FTS forecasting, both for one- and multi-step forecasts. Additionally, Convolutional Neural Networks (CNNs) and LSTM-CNNs are utilized for extracting information from Limit Order Books (LOBs), and Spiking Neural Networks (SNNs) are employed for predicting price spikes in High-Frequency Trading (HFT) strategies.

4. **Transformation of Time Series Data**: To apply deep learning methods to FTS, it's essential to transform the time series to address non-stationarity. Common methods include price differencing, returns, and log-returns, with returns and log-returns being preferred for their unitless nature and convenience in finance.

5. **Serial Correlation and Data Leakage**: Financial time-series data often exhibit serial correlations, leading to potential data leakage if training and testing sets are not properly separated. Techniques like Purging (to remove overlapping observations) and Embargoing (to eliminate subsequent observations) are implemented to avoid immediate and delayed data leakage.

6. **Challenge of Forecasting Stock Market Trends**: The uncertainty, complexity, and noisy nature of stock market features make accurate forecasting extremely challenging. The continuously shifting trends, influenced by numerous, often unknown factors, defy consistent patterning, complicating the application of traditional machine learning approaches in non-stationary stock forecasting problems.

&nbsp;

---

&nbsp;

The frequency of trading strategies ranges from milliseconds to minutes and days. Most studies on FTS forecasting \[10, 11\] focus on minute frequencies as millisecond trading data is expensive.

FTS are characterized by properties including frequency, auto-correlation, heteroskedasticity, drift, and seasonality

Currently, LSTMs are successfully used for minute-frequency FTS forecasting \[21\] for one- and multistep forecasting \[22\]. Convolutional Neural Networks (CNNs) and LSTM-CNNs are used to extract information from LOBs \[10\]. Spiking Neural Networks (SNNs) predict price spikes for HFT strategies \[23\].

One challenge with FTS is that the range of values varies significantly, even for the same financial asset. If we want to apply a deep learning method to forecast FTS, we need to transform the time series to remove the non-stationarity. Usually, three price time series transformations are proposed: • price differencing: dt+1 = pt+1 − pt • returns: r ∗ t+1 = pt+1−pt pt • log-returns: rt+1 = log(pt+1) − log(pt) All three of these methods remove the non-stationarity of prices

- However, using price differencing preserves the units and will be proportional to the underlying price of the asset, which is undesirable in the case of Bitcoin because of its high volatility. On the contrary, returns and log-returns are unitless as it is a ratio of two prices. Returns represent the change in asset price as a percentage and are therefore widely used in finance for their convenience.

Financial time-series data often exhibit serial correlations, meaning that consecutive data points (like stock prices on consecutive days) are correlated with each other. This characteristic can lead to leakage if the training and testing sets are not properly segregated

- **Purging** removes overlapping observations to prevent immediate data leakage.
- **Embargo** goes a step further by removing subsequent observations to account for delayed effects and serial correlations, ensuring a more robust and leakage-free training process.

Due to the uncertainty of features involved and their complex and noisy nature, it becomes challenging to accurately forecast stock market trends.

- As the trend is continuously shifting under the influence of several factors, many of which remain unknown and uncontrollable, there is no consistent pattern to follow&nbsp;
- Thus, it is difficult to apply traditional ML approaches when dealing with non-stationary stock forecasting problems.
- &nbsp;

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

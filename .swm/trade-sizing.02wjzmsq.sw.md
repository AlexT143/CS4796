---
title: Trade Sizing
---
1. **Threshold-Based Trade Sizing**: The strategy employs thresholds to determine trade sizes. With two thresholds, 0.15 BTC is traded above the highest threshold, 0.1 BTC for the medium threshold, and 0.05 BTC for lower cases. For a five-threshold system, trade sizes vary from 0.15 BTC at the highest to 0.025 BTC below the lowest threshold.

2. **Experiment Setup**: The experiment uses seven signals with a primary forecasting horizon of 25 ticks to assess the impact of trade sizing on performance.

3. **Impact on Cumulative Profit and Loss (PnL)**: Introducing trade sizing increases the cumulative PnL. Specifically, with two thresholds, the standard deviation of cumulative PnL rises from 0.49 USDT (without trade sizing) to 0.58 USDT. However, with five thresholds, while the PnL doesn't increase significantly, the standard deviation decreases to 0.41 USDT.

4. **Benefits of More Thresholds**: Increasing the number of thresholds for trade sizing doesn't substantially boost PnL but does reduce its standard deviation. A lower standard deviation of PnL implies a better Sharpe ratio in the long term and lessens the volatility of the high-frequency trading strategy.

5. **Proposed Strategy Improvement**: To enhance the trading strategy, it's suggested to increase the quantity traded in proportion to the strength of the trading signal. This approach aligns trade size more closely with the confidence or strength of the signal.

---

2 thresholds: above the highest threshold 0.15 BTC is traded, 0.1 BTC for the medium threshold, and 0.05 BTC otherwise&nbsp;

5 thresholds: above the highest threshold, 0.15 BTC is traded, 0.125 BTC for the second highest threshold, 0.1 BTC for the third highest threshold, 0.075 for the fourth highest threshold, 0.05 for the fifth highest threshold, and 0.025 BTC otherwise

&nbsp;

![](https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM%3D%2F2b553a1b-2752-4a29-871f-c04264bf6c8f.png?alt=media&token=0c220341-1b27-485e-b797-af95cea86e9b)For this experiment, we use 7 signals with a main forecasting horizon of 25 ticks. We notice an increase in the cumulative PnL when using trade sizing. The standard deviation of the cumulative PnL increases from 0.49 USDT without trade sizing to 0.58 USDT with trade sizing using 2 thresholds. However, when the number of thresholds for trade sizing is increased to 5, the PnL does not increase significantly, but the standard deviation of the PnL drops to 0.41 USDT. A lower standard deviation of the PnL will yield a better Sharpe ratio long-term and reduce the volatility of the high-frequency strategy&nbsp;

- Therefore, one proposed improvement to the trading strategy is to increase the quantity traded proportionally to the strength of the trading signal.

![](https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM%3D%2Fc8f742be-9e6e-420d-a01f-21eda7b5e534.png?alt=media&token=b1f09e3b-7939-41a8-9b3e-157a7dd11fd2)

&nbsp;

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: Testing
---
1. **Dataset Splitting**: Datasets are divided chronologically into train, validation, and test sets. For ETT datasets, the split follows a 60/20/20 ratio, while for other datasets, it's a 70/10/20 ratio.

2. **Data Normalization**: Input data is normalized to have a zero mean. This is a standard preprocessing step to ensure consistent scaling across the datasets.

3. **Evaluation Metrics**: Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used as the primary metrics for evaluating model performance. These metrics are widely recognized for their effectiveness in quantifying forecasting accuracy.

4. **HFformer and Loss Functions**: The HFformer model is evaluated using MSE and MAE loss functions. Additionally, the quantile loss is considered for forecasting a range around the predicted value. This becomes particularly useful when the inter-quartile range exceeds the mean inter-quartile range, indicating lower certainty in the forecast and potentially influencing trade decisions.

5. **Normalization and Splitting Strategy**: Another approach involves min-max normalization of the data followed by splitting into 80% for training, 10% for validation, and 10% for testing.

6. **Sequence Separation**: Training, validation, and test sets are further divided into individual sequences. Each sequence spans 16 days, with 5 features per day including Volume, Open, High, Low, and Close. This segmentation aids in capturing daily trends and fluctuations relevant to the model's forecasting accuracy.

---

&nbsp;

&nbsp;

&nbsp;

For the main benchmark, datasets are split into train, validation, and test sets chronologically, following a 60/20/20 split for the ETT datasets and 70/10/20 split for other datasets. Inputs are zero-mean normalized and we use MSE and MAE as evaluation metrics.

The HFformer works with MSE and MAE loss. However, quantile loss allows forecasting an interval around the predicted value. This is helpful when the inter-quartile range becomes larger than the mean inter-quartile range, which could signify a low certainty about the forecasted value and therefore one might decide not to enter the trade

Then, the values are min-max normalized and split into 80% for training, 10% for validation and 10% for test set. Finally, the training, validation and test sets are separated into individual sequences with a length of 16 days and 5 features per sequence day i.e., Volume, Open, High, Low, and Close&nbsp;

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

---
title: Data Sourcing
---
 1. **Impact of Training Data Size**: Contrary to common belief, smaller datasets sometimes yield lower prediction errors in time series forecasting (TSF) with transformers. This could be due to clearer temporal features in smaller, complete datasets compared to larger, incomplete ones.

 2. **Performance of Different Models**: While the HFformer generally outperforms other Transformer-like architectures, it initially underperforms compared to LSTM. However, with increased training data size, HFformer's forecasting accuracy surpasses that of LSTM.

 3. **Normalization in Machine Learning**: Traditional normalization using training set parameters may not be effective for fluctuating data like BTC-USDT. Online Normalization is proposed as a solution to account for significant fluctuations in data like mean and standard deviation.

 4. **Requirement of Time Embedding**: Autoformer and FEDformer models necessitate an additional input feature - the date and time timestamp, using the frequency of the target time series as a hyperparameter.

 5. **Training and Validation Set Size**: Larger training and validation sets may lead to decreased absolute performance of models. However, larger sets are important for real-world application like live trading to avoid continuous retraining and overfitting.

 6. **Size of Datasets and Model Performance**: The LSTM shows better performance on smaller datasets, but as the size of training and validation sets increases, the HFformer's performance improves relative to LSTM.

 7. **Sample Independence in ML Models**: It's crucial to maintain independence between samples in time-series data. K-fold cross-validation is commonly used, but this can be challenging due to the time-ordered nature of time series data, leading to potential overfitting.

 8. **Challenges with Cross-Validation in Time Series**: Applying K-fold cross-validation to time series data raises issues. Time-ordering of data may not be preserved, and high correlation along the time axis can inflate performance metrics on validation sets.

 9. **Addressing Data Leak in Time Series**: Data leakage in time series can be mitigated through techniques like Purging and Embargoing, especially in transformer-based models.

10. **Requirement of Large Data for Transformers**: Transformers, being data-hungry models, require substantial data to train effectively with minimal overfitting issues. Federated Learning (FL) can be a viable approach in this context.

11. **Model Training on Historical Stock Data**: Training on historical daily stock data, using the day as a time feature in Time2Vec representation, and employing multiple transformer encoders can effectively predict output trends.

&nbsp;

&nbsp;

&nbsp;

"Some may argue that the poor performance of Transformer-based solutions is due to the small sizes of the benchmark datasets. Unlike computer vision or natural language processing tasks, TSF is performed on collected time series, and it is difficult to scale up the training data size. In fact, the size of the training data would indeed have a significant impact on the model performance. ...Unexpectedly, Table 7 presents that the prediction errors 7 with reduced training data are lower in most cases. This might because the whole-year data maintains more clear temporal features than a longer but incomplete data size. While we cannot conclude that we should use less data for training, it demonstrates that the training data scale is not the limiting reason for the performances of Autoformer and FEDformer."

COUNTER: The HFformer outperforms the other Transformer-like architectures, however, it underperforms the LSTM (cf. Figure 9), however, as the size of the training dataset increases, the HFformer attains better forecasting results than the LSTM during backtesting.&nbsp;

To improve the performance of machine learning methods, input data needs to be normalized. Traditionally, the normalization parameters from the training set are applied to the test set (e.g., the mean and standard deviation of the training data). However, as in the case of the BTC-USDT pair, the mean and standard deviation of the prices and quantities fluctuate significantly. Taking this into account, we propose to use \[\[Online Normalization\]\]

The Autoformer and FEDformer require an additional input feature which is the date and time timestamp, as both of these models use a time embedding. The frequency of the target time series (e.g., seconds, minutes, and hours) is used as a hyperparameter.

![](https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM%3D%2F143f876b-7fc9-4e06-af90-24cccfa4d161.png?alt=media&token=f0593c8e-9fa1-4c52-a112-da75ea6a1ec2)From Figure 13, we notice that as the training and validation sets increases the absolute performance of both models decreases.&nbsp;

However, it is important to consider larger training and validation sets, as during live trading, it can be technically challenging to continuously retrain the model and may lead to overfitting.

he LSTM outperforms the HFformer on smaller training and validation datasets.  #IMPORTANT The HFformer’s forecasting performance improves relative to the LSTM’s as the size of the training and validation sets increases.&nbsp;

In order to develop ML models based on time-series data, it is important to maintain independence between the samples, since machine learning generally assumes that the samples are independent.

\* As a result, K-fold cross-validation is commonly used in this scenario \[27\]. In K-fold cross-validation, selected samples serve as validation sets and the remaining samples serve as training sets.

\* The dataset is randomly divided into K equal-size subsets and each is used as a validation set and the remaining subsets as a training set in \[\[k-fold cross-validation|k-fold cross-validation\]\]&nbsp;

\* As a result of applying this algorithm to time series data, there are two problems; first, as time-series data are timeordered, , the assumption would not be preserved if crossvalidation were applied. Second, time-series data tends to be highly correlated with the time axis, consequently, in cases of overfitting, the performance metrics on the validation set are increased&nbsp;

This unintended data leak can be solved through \[\[Purging\]\] and \[\[Embargoing\]\] \[25\]![](https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM%3D%2F450087fc-0f3f-4db7-847e-0c3aa2381d26.png?alt=media&token=bf46e6d5-64ba-488e-ad61-f4bd719e93d1)\[\[Transformers with Attentive Federated Aggregation for Time Series Stock Forecasting\]\]

Nonetheless, large amount of data is required in training a data-hungry transformer with minimum overfitting issue

```
- \[\[Federated Learning (FL)\]\]
```

&nbsp;

Specifically, we train our model on historical daily stock data where the input integer (day) is used as the time feature for Time2Vec representation \[11\] and a number of transformer encoders are stacked above to predict the output trend

<SwmMeta version="3.0.0" repo-id="Z2l0aHViJTNBJTNBQ1M0Nzk2JTNBJTNBQWxleFQxNDM=" repo-name="CS4796"><sup>Powered by [Swimm](https://app.swimm.io/)</sup></SwmMeta>

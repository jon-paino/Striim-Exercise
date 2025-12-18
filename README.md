# Real-Time Credit Card Fraud Detection: From EDA to Robust Streaming Inference

## 1. Executive Summary

This project implements a low-latency, hybrid streaming anomaly detector for credit card fraud. The system evolved from a naive statistical profiler into a robust ensemble architecture (`RobustStreamDetectorV2`) capable of processing transactions in O(1) time while maintaining high precision on highly imbalanced data (0.17% fraud rate).

## 2. Exploratory Data Analysis (EDA) & Feature Engineering

Before architectural design, a deep analysis of the dataset revealed three critical characteristics that dictated my modeling strategy:

* **High-Dimensional Orthogonality:** The features `V1` through `V28` are Principal Components of the original transaction data. While this obfuscates semantic meaning, it guarantees feature orthogonality, reducing multicollinearity issues for linear models and distance-based metrics.
* **Extreme Class Imbalance:** Fraudulent transactions account for only ~0.17% of the dataset. This necessitated a shift from standard Accuracy metrics to **Precision-Recall (PR)** optimization and required specific sampling strategies (undersampling majority class) for supervised training.
* **Distributional Skew (The "Power Law" Problem):**
* **Amount:** As visualized in the initial histograms, the `Amount` feature is severely right-skewed (heavy-tailed). Most transactions are small, with rare, massive outliers. This observation was critical as naive Z-score models failed here because the data is *not* normally distributed.
* **Time:** The `Time` feature showed a bimodal distribution, reflecting the normal rhythm of transaction volume (day/night cycles).



## 3. System Evolution & Architecture

### Phase 1: The Naive Baseline (Z-Score & Heuristics)

My initial approach attempted to model anomalies using simple statistical profiling:

* **Mechanism:** Calculated a rolling Z-score for transaction amounts.
* **Failure Mode:** Due to the power-law distribution of `Amount`, legitimate high-value transactions resulted in massive Z-scores (Z > 20), triggering constant False Positives. The assumption of Gaussian normality was flawed.

### Phase 2: `FastStreamDetector` (Hybrid Prototype)

We evolved to a hybrid approach combining supervised and unsupervised logic:

* **Stack:** Random Forest (Supervised) + Isolation Forest (Unsupervised) + Rolling Statistics.
* **Optimization:** Utilized `collections.deque` for O(1) rolling window updates.
* **Shortcomings:** Reliance on hardcoded thresholds (`multivariate_dist > 20.0`) and raw feature inputs made the model brittle. It achieved reasonable Recall (67%) but suffered from poor Precision (16.8%) due to high false alarm rates.

### Phase 3: `RobustStreamDetectorv2` (The Production Candidate)

The final architecture introduced three engineering breakthroughs that tripled precision:

1. **Log-Space Transformation:** Addressing the EDA findings, we applied `np.log1p` to the `Amount` column. This compressed the heavy tail into a quasi-normal distribution, making Z-scores and Euclidean distance metrics mathematically valid.
2. **Dynamic Quantile Calibration:** Instead of "magic numbers," thresholds were calibrated dynamically (99.9th percentile) against a "Normal-Only" reference set. This bounded the theoretical False Positive Rate.
3. **Clean-Baseline Training:** We isolated non-fraud data to calculate baseline statistics, preventing fraud outliers from "polluting" the model's definition of normality.

**Performance Impact:**

* **False Positives:** Reduced by ~82% (1113 \to 199).
* **Precision:** Improved from 17% \to 52%.

## 4. Modeling Assumptions

* **Stationarity (Relaxed):** We assume the statistical properties of "normal" spending change slowly enough that a sliding window (N=1000) or periodic retraining can capture the drift.
* **Independence:** While the `Time` feature suggests temporal dependence, we modeled events primarily as independent vectors, assuming the *content* of the transaction (Amount + V-features) is more predictive than the timestamp alone.
* **Latency Constraints:** The system assumes a requirement for near real-time inference, necessitating O(1) complexity for state updates.

## 5. Future Roadmap: Scaling for Production

To move this prototype into a high-throughput production environment (10k+ TPS), the following evolution is required:

### Tech Stack Migration

* **Ingestion:** Replace CSV iteration with **Apache Kafka** or **AWS Kinesis** for event streaming.
* **State Management:** Move in-memory `deque` state to a low-latency feature store like **Redis** to handle state failover and distributed processing.
* **Compute:** Deploy the inference logic on **Apache Flink** or **Spark Structured Streaming** for horizontal scalability.

### MLOps & Monitoring

* **Drift Detection:** Implement Kolmogorov-Smirnov (KS) tests on the distribution of incoming `V` features to detect concept drift ( new fraud patterns).
* **Feedback Loop:** Build a pipeline to ingest confirmed chargeback data (labels) to retrain the Random Forest model daily/weekly.
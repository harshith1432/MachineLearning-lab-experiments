# Machine Learning Lab Experiments (ML for Cybersecurity)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Library-statsmodels-00599C.svg)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, organized collection of **12 Machine Learning Laboratory Experiments** demonstrating practical applications of machine learning, time-series forecasting, and unsupervised anomaly detection in **Cybersecurity and Network Defense**.

All 12 laboratory experiments are provided as clean, self-contained **Jupyter Notebooks (`exp1.ipynb` to `exp12.ipynb`)** ready to run in Jupyter Lab, Jupyter Notebook, VS Code, or Google Colab.

---

## 📌 Repository Overview & About Section

This repository serves as a hands-on laboratory manual and codebase for undergraduate/postgraduate courses in **Machine Learning**, **Data Science**, and **Cybersecurity Applications**. It addresses critical modern security challenges:
- **Network Traffic Profiling & Baselining**: Distinguishing legitimate seasonal variations from anomalous activity.
- **Volumetric Attack Detection**: Forecasting network volume with ARIMA to trigger early DDoS mitigation.
- **Intrusion Detection Systems (IDS)**: Multi-class attack classification (DoS, Probe, R2L, U2R).
- **Phishing & Malicious URL Detection**: Utilizing lexical tokenization, TF-IDF n-grams, and Support Vector Machines.
- **Unsupervised Anomaly Detection**: Identifying novel and zero-day threat patterns with Isolation Forest.
- **Email Security & Protocol Verification**: Evaluating SPF, DKIM, and DMARC headers alongside NLP-based spam filtering.
- **Database & Application Security**: Detecting SQL Injection (SQLi) patterns using Decision Trees.
- **IoT & Sensor Network Protection**: Detecting compromised nodes in Wireless Sensor Networks (WSNs).

---

## 📋 Experiments Index

| Exp # | Experiment Title | Model / Algorithm | Cybersecurity Domain | Notebook File |
| :---: | :--- | :--- | :--- | :--- |
| **01** | Network Traffic Analysis & Seasonal Decomposition | Additive Seasonal Decomposition (`statsmodels`) | Network Baseline & Profiling | [`exp1.ipynb`](exp1.ipynb) |
| **02** | DDoS Anomaly Detection & Traffic Forecasting | ARIMA Time Series Forecasting | DDoS & Volumetric Surge Detection | [`exp2.ipynb`](exp2.ipynb) |
| **03** | Multi-Class Cyber-Attack Classification | Random Forest Classifier | Intrusion Detection (IDS) | [`exp3.ipynb`](exp3.ipynb) |
| **04** | Malicious & Phishing URL Classification | TF-IDF Character n-grams + Logistic Regression | Web & URL Threat Filtering | [`exp4.ipynb`](exp4.ipynb) |
| **05** | Phishing URL Detection via Lexical Features | Linear Support Vector Machine (`LinearSVC`) | Web Security & Phishing Defense | [`exp5.ipynb`](exp5.ipynb) |
| **06** | Unsupervised URL Anomaly Detection | Isolation Forest | Zero-Day & Obfuscation Detection | [`exp6.ipynb`](exp6.ipynb) |
| **07** | Handwritten Digit Recognition & Visual Security | Random Forest Classifier | CAPTCHA Breaking & OCR Security | [`exp7.ipynb`](exp7.ipynb) |
| **08** | Email Spoofing & Phishing Header Analysis | Random Forest Classifier | Email Authentication (SPF/DKIM/DMARC) | [`exp8.ipynb`](exp8.ipynb) |
| **09** | Spam & Phishing Email Text Classification | TF-IDF Vectorizer + Multinomial Naive Bayes | Social Engineering & Spam Filtering | [`exp9.ipynb`](exp9.ipynb) |
| **10** | Network Intrusion & Event Clustering | K-Means Clustering + `StandardScaler` | Unsupervised Network Profiling | [`exp10.ipynb`](exp10.ipynb) |
| **11** | SQL Injection (SQLi) Query Detection | Decision Tree Classifier | Web Application & Database Defense | [`exp11.ipynb`](exp11.ipynb) |
| **12** | Wireless Sensor Network (WSN) Attack Detection | Random Forest Classifier | IoT & Wireless Sensor Security | [`exp12.ipynb`](exp12.ipynb) |

---

## 🔍 Detailed Experiment Breakdown

### Experiment 1: Network Traffic Analysis via Seasonal Decomposition
- **Objective**: Decompose complex network traffic logs into underlying trend, cyclical/seasonal patterns, and residual noise.
- **Algorithm**: `statsmodels.tsa.seasonal.seasonal_decompose` (Additive Model).
- **Features**: Daily timestamp series, aggregate network packet counts.
- **Cybersecurity Role**: Helps SOC analysts establish baseline normal operations and filter periodic diurnal spikes from actual anomalies.

### Experiment 2: DDoS Anomaly Detection & Traffic Forecasting
- **Objective**: Forecast future traffic loads and detect abrupt volumetric spikes characteristic of Distributed Denial of Service (DDoS).
- **Algorithm**: Autoregressive Integrated Moving Average (ARIMA).
- **Features**: Daily traffic trends, 7-day cyclical variations, and sudden synthetic attack bursts.
- **Cybersecurity Role**: Enables automated threshold alert triggers before network infrastructure becomes unresponsive.

### Experiment 3: Multi-Class Cyber-Attack Classification
- **Objective**: Classify diverse intrusion attempts into five distinct categories: DoS, Port Probe, Remote-to-Local (R2L), User-to-Root (U2R), and Normal traffic.
- **Algorithm**: Random Forest Classifier with feature importance evaluation.
- **Features**: Duration, protocol type, service, source bytes, destination bytes, error rates.
- **Cybersecurity Role**: Core logic for signature-and-anomaly Network Intrusion Detection Systems (NIDS).

### Experiment 4: Malicious & Phishing URL Classification
- **Objective**: Detect phishing and malicious domains based on character-level n-gram structures without relying on static blocklists.
- **Algorithm**: `TfidfVectorizer(analyzer="char", ngram_range=(3, 5))` + Logistic Regression.
- **Features**: Subdomain structures, typosquatting patterns, and character distribution.
- **Cybersecurity Role**: Protects endpoints and email gateways by screening outbound web requests in real-time.

### Experiment 5: Phishing URL Detection via Lexical Features
- **Objective**: Evaluate structural numerical features of URLs using maximum-margin hyperplanes.
- **Algorithm**: Linear Support Vector Classifier (`LinearSVC`).
- **Features**: URL length, domain length, number of subdomains, digit counts, special character counts, IP address usage, HTTPS status.
- **Cybersecurity Role**: High-speed, low-latency URL scoring suitable for high-throughput perimeter firewalls.

### Experiment 6: Unsupervised URL Anomaly Detection
- **Objective**: Flag suspicious, obfuscated, or zero-day URLs without requiring labeled malicious training data.
- **Algorithm**: Isolation Forest (`IsolationForest` with contamination factor).
- **Features**: Digit-to-letter ratios, special-character ratios, redirect indicators, obfuscation flags.
- **Cybersecurity Role**: Defense against novel attack campaigns where attacker domains have not yet been blacklisted.

### Experiment 7: Handwritten Digit Recognition & Visual Security
- **Objective**: Classify optical character patterns from pixel intensity data.
- **Algorithm**: Random Forest Classifier on 8x8 pixel grids (`sklearn.datasets.load_digits`).
- **Features**: 64 normalized grayscale pixel intensity values.
- **Cybersecurity Role**: Models computer vision mechanisms used in CAPTCHA challenge solving and identity document verification.

### Experiment 8: Email Spoofing & Phishing Header Analysis
- **Objective**: Detect domain spoofing and forged sender addresses by evaluating security authentication headers.
- **Algorithm**: Random Forest Classifier.
- **Features**: Sender domain alignment, SPF pass flag, DKIM cryptographic signature verification, DMARC policy pass flag.
- **Cybersecurity Role**: Essential layer in Secure Email Gateways (SEGs) combating Business Email Compromise (BEC) and executive impersonation.

### Experiment 9: Spam & Phishing Email Text Classification
- **Objective**: Identify social engineering, urgent calls to action, and phishing lures inside raw message bodies.
- **Algorithm**: Pipeline of `TfidfVectorizer` + `MultinomialNB` (Multinomial Naive Bayes).
- **Features**: Word frequency distributions and TF-IDF term weighting.
- **Cybersecurity Role**: Content filtering engine for mail transfer agents (MTAs) and user mailboxes.

### Experiment 10: Network Intrusion & Event Clustering
- **Objective**: Partition unlabelled network flow events into normal operations and anomalous behaviors.
- **Algorithm**: K-Means Clustering with `StandardScaler` feature normalization.
- **Features**: Source bytes, destination bytes, connection counts, service flags.
- **Cybersecurity Role**: Unsupervised network profiling and discovery of unknown intrusion vectors.

### Experiment 11: SQL Injection (SQLi) Query Detection
- **Objective**: Inspect database query strings to flag SQL injection exploits targeting backend databases.
- **Algorithm**: Decision Tree Classifier (`DecisionTreeClassifier`).
- **Features**: Query length, count of quotes/dashes/special symbols, presence of SQL keywords (`UNION`, `OR`, `SELECT`), failed query attempts.
- **Cybersecurity Role**: Web Application Firewall (WAF) rule augmentation and database auditing.

### Experiment 12: Wireless Sensor Network (WSN) Attack Detection
- **Objective**: Identify malicious sensor nodes performing routing attacks, blackhole attacks, or battery exhaustion attacks in IoT environments.
- **Algorithm**: Random Forest Classifier.
- **Features**: Node ID, packet delivery ratio, received signal strength (RSSI), energy depletion rate, transmission frequency.
- **Cybersecurity Role**: Securing mission-critical IoT, smart grid, and industrial control sensor topologies.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** and `pip` installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/harshith1432/MachineLearning-lab-experiments.git
cd MachineLearning-lab-experiments
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run the Experiments

Launch Jupyter Notebook or Jupyter Lab from the project root:
```bash
jupyter notebook
```
Navigate to any experiment notebook (e.g., `exp1.ipynb` through `exp12.ipynb`) and select **Cell -> Run All** or run individual cells with `Shift + Enter`.

You can also run the notebooks in:
- **VS Code**: Install the Jupyter extension and open any `exp*.ipynb` file.
- **Google Colab**: Upload the notebook and accompanying sample CSV files (`urlset.csv`, `PhiUSIIL_Phishing_URL_Dataset.csv`).

---

## 📁 Repository Structure

```text
MachineLearning-lab-experiments/
│
├── exp1.ipynb                         # Exp 1: Network Traffic Decomposition
├── exp2.ipynb                         # Exp 2: DDoS Anomaly Forecasting
├── exp3.ipynb                         # Exp 3: Cyber-Attack Classification
├── exp4.ipynb                         # Exp 4: Phishing URL Logistic Regression
├── exp5.ipynb                         # Exp 5: Phishing URL Support Vector Machine
├── exp6.ipynb                         # Exp 6: Unsupervised Anomaly Detection
├── exp7.ipynb                         # Exp 7: Handwritten Digit CAPTCHA Classifier
├── exp8.ipynb                         # Exp 8: Email Spoofing & Header Analysis
├── exp9.ipynb                         # Exp 9: Spam & Phishing Email Naive Bayes
├── exp10.ipynb                        # Exp 10: Network Event K-Means Clustering
├── exp11.ipynb                        # Exp 11: SQL Injection Detection
├── exp12.ipynb                        # Exp 12: Wireless Sensor Network Defense
│
├── urlset.csv                         # Sample dataset for Exp 4
├── PhiUSIIL_Phishing_URL_Dataset.csv   # Sample dataset for Exp 5 & 6
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Standard Python & Jupyter ignore rules
└── README.md                          # Comprehensive documentation & manual
```

---

## 🛡️ License & Acknowledgements

- **License**: Released under the [MIT License](LICENSE).
- **Source Reference**: Adapted and structured from [himani-2105/Machine-Learning-for-cybersecurity](https://github.com/himani-2105/Machine-Learning-for-cybersecurity).
- **Curated By**: [Harshith D](https://github.com/harshith1432)

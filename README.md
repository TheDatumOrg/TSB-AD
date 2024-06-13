<p align="center">
<img width="300" src="doc/fig/readme_title.png"/>
</p>

<h1 align="center">TSB-AD</h1>
<h2 align="center">The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark</h2>

## 📄 Contents
1. [Overview](#overview)
2. [Get Started](#start)
3. [Dive into TSB-AD](#tsb)


<h2 id="overview"> 1. Overview </h2>

Time-series anomaly detection is a fundamental task across scientific fields and industries. However, the field has long faced the "elephant in the room:" critical issues including flawed datasets, biased evaluation metrics, and inconsistent benchmarking practices that have remained largely ignored and unaddressed.  We introduce the TSB-AD to systematically tackle these issues in the following three aspects: (i) Dataset Integrity: with 1020 high-quality time series refined from an initial collection of 4k spanning 33 diverse domains, we provide the first large-scale, heterogeneous, meticulously curated dataset that combines the effort of human perception and model interpretation; (ii) Metric Reliability: by revealing bias in evaluation metrics, we perform ranking aggregation on a set of reliable evaluation metrics for comprehensive capturing of model performance and robustness to address concerns from the community; (iii) Comprehensive Benchmarking: with a broad spectrum of 35 detection algorithms, from statistical methods to the latest foundation models, we perform systematic hyperparameter tuning for a fair and complete comparison. Our findings challenge the conventional wisdom regarding the superiority of advanced neural network architectures, revealing that simpler architectures and statistical methods often yield better performance. While foundation models demonstrate promise, we need to proceed with caution in terms of data contamination.

<h2 id="start"> 2. Get Started </h2>

### 2.1 Dataset Download
Due to limitations in the upload size on GitHub, we host the datasets at [Link](https://drive.google.com/file/d/1jC_DLhFk8ytinRbucEPleH4Rdy2EF0t9/view?usp=sharing).

### 2.2 Detection Algorithm Implementation

To install TSB-AD from source, you will need the following tools:
- `git`
- `conda` (anaconda or miniconda)

**Step 1:** Clone this repository using `git` and change into its root directory.

```bash
git clone https://github.com/TheDatumOrg/TSB-AD.git
```

**Step 2:** Create and activate a `conda` environment named `TSBAD`.

```bash
conda env create --file environment.yml
conda activate TSBAD
```

**Step 3:** Install the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

If you want to use Chronos, please install the follwing
```bash
git clone https://github.com/autogluon/autogluon
cd autogluon && pip install -e timeseries/[TimeSeriesDataFrame,TimeSeriesPredictor]
```

#### 🧑‍💻 Basic Usage

See Example in `TSB_AD/main.py`

```bash
AD_Name = 'IForest'
data_direc = '../Datasets/TSB-AD-U/001_NAB_data_Traffic_4_624_2087.csv'

# Loading Data
df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float)
label = df['Label'].astype(int).to_numpy()

# Applying Anomaly Detector
output = run_Unsupervise_AD(AD_Name, data)

# Evaluation
evaluation_result = get_metrics(output, label)
```

<h2 id="tsb"> 3. Dive into TSB-AD </h2>

### Dataset Overview 

<p align="center">
<img width="500" src="doc/fig/tsb_overview.png"/>
</p>

> Example time series from TSB-AD, with anomalies highlighted in red. TSB-AD features high-quality labeled time series from a variety of domains, characterized by high variability in length and types of anomalies. Only one channel in a multivariate time series is visualized for brevity.

### Detection Algorithm

[Statistical Method]

[Neural Network-based Method]

[Foundation Model-based Method]


```bash
## WIP
```
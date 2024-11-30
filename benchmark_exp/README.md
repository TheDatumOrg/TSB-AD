### Scripts for running experiments/develop new methods in TSB-AD

* Hper-parameter Tuning: HP_Tuning_U/M.py

* Benchmark Evaluation: Run_Detector_U/M.py

* `benchmark_eval_results/`: Evaluation results of anomaly detectors across different time series in TSB-AD
    * All time series are normalized by z-score by default

* Develop your own algorithm: Run_Custom_Detector.py
    * Step 1: Implement `Custom_AD` class
    * Step 2: Implement model wrapper function `run_Custom_AD_Unsupervised` or `run_Custom_AD_Semisupervised`
    * Step 3: Specify `Custom_AD_HP` hyperparameter dict
    * Step 4: Run the custom algorithm either `run_Custom_AD_Unsupervised` or `run_Custom_AD_Semisupervised`
    * Step 5: Apply threshold to the anomaly score (if any)

🪧 How to commit your own algorithm to TSB-AD: you can send us the Run_Custom_Detector.py (replace Custom_Detector with the model name) to us via (i) [email](liu.11085@osu.edu) or (ii) open a pull request and add the file to `benchmark_exp` folder in `TSB-AD-algo` branch. We will test and evaluate the algorithm and include it in our [leaderboard](https://thedatumorg.github.io/TSB-AD/).
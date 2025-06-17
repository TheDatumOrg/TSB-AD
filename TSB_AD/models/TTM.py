"""
This function is adapted from [TTMs] by [Ekambaram et al.]
Original source: [https://github.com/ibm-granite/granite-tsfm]
"""

import os
import tempfile
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

from TSB_AD.models.base import BaseDetector
from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions


class TTM(BaseDetector):
    def __init__(self,
                 model_path="ibm-granite/granite-timeseries-ttm-r2",
                 context_length=512,
                 prediction_length=96,
                 num_epochs=50,
                 batch_size=4):
        super().__init__(contamination=0.1)
        self.model_name = 'TTM'
        self.model_path = model_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.tsp = None
        self.column_specifiers = {}
        self.split_config = {}

    def fit(self, data):
        try:
            print("[TTM] Step 0 Reconstructing DataFrame")

            num_features = data.shape[1]
            feature_names = [f"feature_{i}" for i in range(num_features)]
            df = pd.DataFrame(data, columns=feature_names)

            if not self.column_specifiers:
                self.column_specifiers = {
                    "timestamp_column": None,
                    "id_columns": [],
                    "target_columns": feature_names,
                    "control_columns": [],
                }

            if not self.split_config:
                num_rows = len(df)
                self.split_config = {
                    "train": [0, int(0.8 * num_rows)],
                    "valid": [int(0.8 * num_rows), int(0.9 * num_rows)],
                    "test": [int(0.9 * num_rows), num_rows],
                }

            print("[TTM] Step 1 Initialize preprocessor")
            self.tsp = TimeSeriesPreprocessor(
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                scaling=True,
                encode_categorical=False,
                scaler_type="standard",
                column_specifiers=self.column_specifiers
            )

            print("[TTM] Step 2 Load model")
            self.model = get_model(
                self.model_path,
                context_length=self.context_length,
                prediction_length=self.prediction_length
            )

            print("[TTM] Step 3 Creating datasets")
            dset_train, dset_val, dset_test = get_datasets(
                self.tsp,
                df,
                self.split_config,
                use_frequency_token=False
            )

            self.test_size_ = len(dset_test)

            print("[TTM] Step 4 Training")
            temp_dir = tempfile.mkdtemp()
            training_args = TrainingArguments(
                output_dir=temp_dir,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.num_epochs,
                evaluation_strategy="epoch",
                save_strategy="no",
                logging_strategy="epoch",
                report_to="none",
                seed=7
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dset_train,
                eval_dataset=dset_val
            )
            trainer.train()

            print("[TTM] Step 5 Predicting")
            predictions = trainer.predict(dset_test)
            preds = predictions.predictions[0]

            print("[TTM] Step 6 Extracting targets")
            first_sample = dset_test[0]
            if isinstance(first_sample, dict):
                targets = torch.stack([sample["future_values"] for sample in dset_test])
            elif isinstance(first_sample, (tuple, list)) and len(first_sample) > 1:
                targets = torch.stack([sample[1] for sample in dset_test])
            else:
                raise ValueError("[TTM] Unexpected dataset sample structure")

            if hasattr(preds, 'numpy'):
                preds = preds.numpy()
            if hasattr(targets, 'numpy'):
                targets = targets.numpy()

            print(f"[TTM] preds shape: {preds.shape}, targets shape: {targets.shape}")
            raw_scores = np.mean(np.abs(preds - targets), axis=(1, 2))

            padded_scores = np.zeros(len(data))
            padding = len(data) - len(raw_scores)

            if padding < 0:
                raise ValueError(f"[TTM] More scores ({len(raw_scores)}) than data rows ({len(data)}).")

            padded_scores[padding:] = raw_scores
            self.decision_scores_ = padded_scores

        except Exception as e:
            print(f"[TTM] Error in fit(): {e}")
            self.decision_scores_ = np.array([-1])
        return self

    def decision_function(self, X):
        if not hasattr(self, 'decision_scores_') or self.decision_scores_ is None:
            raise RuntimeError("Model has not been fitted correctly.")
        return self.decision_scores_

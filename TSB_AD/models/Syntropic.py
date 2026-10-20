"""
Syntropic: weighted mean of three robust-z-normalised anomaly scores.

1. WSK, multiscale window-statistics kNN against the training prefix. Each
   window is described by [std, range, max, min, mean]. These features are
   standardised with the prefix's statistics, and each window's score is its
   mean distance to the k nearest prefix windows. Scales are p/2, p, 2p, 16, 64
   and 256 (p = find_length_rank), capped at len(prefix) / 4. Each scale is
   robust-z normalised, and the point score is the max over scales. Adapted
   from the statistical component of CHARM_kNN (TSB-AD PR #74).
2. DiffZ, the robust z-score of absolute first differences against the prefix,
   max-filtered over one period.
3. Time-RCD zero-shot (thu-sail-lab/Time-RCD, pretrained on synthetic data
   only), pinned to a fixed Hugging Face revision.

Every score is robust-z normalised (median / IQR over the series, no labels).
The three are combined as a weighted mean, WSK : DiffZ : Time-RCD = 2 : 1 : 0.5.
The detector uses only the series and the training-prefix length; it never sees
the filename or the labels.

One global configuration. The weights were chosen on TSB-AD-U-Tuning only.

Authors: Spiro El Khoury (spiro@syntropiclabs.ai) & John Kane (john@syntropiclabs.ai), Syntropic Labs
License: Apache-2.0
"""
import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.neighbors import NearestNeighbors

from ..utils.slidingWindows import find_length_rank

EPS = 1e-8
TIME_RCD_REPO = "thu-sail-lab/Time-RCD"
TIME_RCD_REVISION = "0880420070a524efe14d7e163b34dcca90d2be1b"  # 2026-08-06
TIME_RCD_FILE = "best_model/pretrain_checkpoint_best_uni.pth"


def _rolling_mean_std(x, w):
    shift = x.mean()  # centring keeps the running sums from losing precision
    x = x - shift
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    mean = s1 / w
    var = np.maximum(s2 / w - mean ** 2, 0.0)
    return mean + shift, np.sqrt(var)


def _sliding_max(x, w):
    """max of x[s : s + w] for every window start s."""
    return maximum_filter1d(x, w, mode="nearest")[w // 2: w // 2 + len(x) - w + 1]


def window_stats(x, w):
    """[std, range, max, min, mean] for every window start, in O(n) per statistic."""
    mean, std = _rolling_mean_std(x, w)
    mx = _sliding_max(x, w)
    mn = -_sliding_max(-x, w)
    return np.stack([std, mx - mn, mx, mn, mean], axis=1)


def windows_to_points(win_score, w, n):
    """Point score = max over all windows that cover the point."""
    padded = np.concatenate([np.full(w - 1, -np.inf), win_score, np.full(w - 1, -np.inf)])
    return maximum_filter1d(padded, w)[w // 2: w // 2 + n]


def robust_z(s):
    med = np.median(s)
    iqr = np.subtract(*np.percentile(s, [75, 25]))
    return (s - med) / (iqr + EPS)


def scales_for(x, train_len):
    p = find_length_rank(x.reshape(-1, 1), rank=1)
    cands = {max(8, p // 2), p, 2 * p, 16, 64, 256}
    cap = max(8, train_len // 4)
    return sorted(w for w in cands if 4 <= w <= cap)


def wsk(x, train_len, k=3, max_train_windows=20000):
    n = len(x)
    per_scale = []
    for w in scales_for(x, train_len):
        feats = window_stats(x, w)
        n_train = max(train_len - w + 1, 1)
        tr = feats[:n_train]
        mu, sd = tr.mean(0), tr.std(0) + EPS
        feats = (feats - mu) / sd
        tr = feats[:n_train]
        if len(tr) > max_train_windows:
            tr = tr[np.linspace(0, len(tr) - 1, max_train_windows).astype(int)]
        nn = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(tr)
        dist, _ = nn.kneighbors(feats)
        per_scale.append(robust_z(windows_to_points(dist.mean(1), w, n)))
    return np.max(per_scale, axis=0)


def diffz(x, train_len):
    d = np.abs(np.diff(x, prepend=x[0]))
    tr = d[:train_len]
    med = np.median(tr)
    mad = np.median(np.abs(tr - med)) * 1.4826
    z = (d - med) / (mad + EPS)
    p = find_length_rank(x.reshape(-1, 1), rank=1)
    return maximum_filter1d(z, max(p, 1))


def time_rcd(data, win_size=15000, batch_size=64, device="cpu", checkpoint=None):
    from huggingface_hub import hf_hub_download
    from .Time_RCD import Time_RCD
    if checkpoint is None:
        checkpoint = hf_hub_download(TIME_RCD_REPO, TIME_RCD_FILE, revision=TIME_RCD_REVISION)
    clf = Time_RCD(win_size=win_size, input_c=data.shape[1], batch_size=batch_size,
                   device=device, checkpoint=checkpoint, model_id=TIME_RCD_REPO)
    clf.zero_shot(data)
    return np.asarray(clf.decision_scores_).ravel()


def _normalise(s):
    s = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return robust_z(s)


def run_Syntropic_Semisupervised(data_train, data_test, weights=(2.0, 1.0, 0.5), k=3,
                                 max_train_windows=20000, time_rcd_win_size=15000,
                                 time_rcd_batch_size=64, device="cpu", time_rcd_checkpoint=None):
    """data_train is the official training prefix and data_test the whole series."""
    data_test = np.asarray(data_test, dtype=float)
    x = data_test.ravel()
    train_len = len(data_train)
    parts = [
        wsk(x, train_len, k=k, max_train_windows=max_train_windows),
        diffz(x, train_len),
        time_rcd(data_test.reshape(len(x), -1), win_size=time_rcd_win_size,
                 batch_size=time_rcd_batch_size, device=device, checkpoint=time_rcd_checkpoint),
    ]
    return np.average([_normalise(p) for p in parts], axis=0, weights=np.asarray(weights, dtype=float))

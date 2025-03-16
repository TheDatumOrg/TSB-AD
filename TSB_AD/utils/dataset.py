import torch
import torch.utils.data
import numpy as np
epsilon = 1e-8

class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.pred_len = pred_len
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size - pred_len) // stride + 1, 0)

        # Generate samples efficiently
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        """ Normalize data using mean and standard deviation. """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        """ Generate windowed samples efficiently using vectorized slicing. """
        data = torch.tensor(self.data, dtype=torch.float32)

        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self.window_size] for i in indices])
        Y = torch.stack([data[i + self.window_size : i + self.window_size + self.pred_len] for i in indices])

        return X, Y  # Inputs & targets

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class TSDataset(torch.utils.data.Dataset):

    def __init__(self, X, y=None, mean=None, std=None):
        super(TSDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class ReconstructDataset_Moment(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        self.samples = self._generate_samples()
        self.input_mask = np.ones(self.window_size, dtype=np.float32)  # Fixed input mask

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.input_mask
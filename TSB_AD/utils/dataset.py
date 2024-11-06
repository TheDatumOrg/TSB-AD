import torch
import torch.utils.data
import numpy as np
epsilon = 1e-8

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


class ReconstructDataset(torch.utils.data.Dataset):

    def __init__(self, data, window_size, step=1, normalize=True):
        super().__init__()
        self.normalize = normalize

        if self.normalize:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std = np.where(data_std == 0, epsilon, data_std)
            self.data = (data - data_mean) / data_std
        else:
            self.data = data

        self.window_size = window_size
        self.step = step
        
        if data.shape[1] == 1:
            data = data.squeeze()
            self.len, = data.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i * step : i * step + self.window_size])
                
            self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(X, -1)
        else:
            self.sample_num = data.shape[0]

            X = torch.zeros((data.shape[0], data.shape[1]))            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i,:])

            self.samples, self.targets = X, X            

    def __len__(self):
        if self.data.shape[1] == 1:
            return self.sample_num
        else:
            return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.data.shape[1] == 1:
            return self.samples[index, :, :], self.targets[index, :, :]
        else:
            if index < self.data.shape[0] - self.window_size:
                return self.samples[index:index+self.window_size, :], self.targets[index:index+self.window_size, :]
            else:
                return self.samples[-self.window_size:, :], self.targets[-self.window_size:, :]


class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, normalize=True):
        super().__init__()
        self.normalize = normalize

        if self.normalize:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std = np.where(data_std == 0, epsilon, data_std)
            self.data = (data - data_mean) / data_std
        else:
            self.data = data

        self.window_size = window_size
        
        if data.shape[1] == 1:
            data = data.squeeze()
            self.len, = data.shape
            self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, pred_len))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(np.array(
                    data[i + self.window_size: i + self.window_size + pred_len]
                ))
            
            self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(Y, -1)
    
        else:
            self.len = self.data.shape[0]
            self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)

            X = torch.zeros((self.sample_num, self.window_size, self.data.shape[1]))
            Y = torch.zeros((self.sample_num, pred_len, self.data.shape[1]))

            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i : i + self.window_size, :])
                Y[i, :] = torch.from_numpy(data[i + self.window_size: i + self.window_size + pred_len, :])
            
            self.samples, self.targets = X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index, :, :], self.targets[index, :, :]


class ReconstructDataset_Moment(torch.utils.data.Dataset):

    def __init__(self, data, window_size, step=1, normalize=True):
        super().__init__()
        self.normalize = normalize

        if self.normalize:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            data_std = np.where(data_std == 0, epsilon, data_std)
            self.data = (data - data_mean) / data_std
        else:
            self.data = data
        self.window_size = window_size
        self.step = step
        
        if data.shape[1] == 1:
            data = data.squeeze()
            self.len, = data.shape
            self.sample_num = max(0, (self.len - self.window_size) // self.step + 1)
            
            X = torch.zeros((self.sample_num, self.window_size))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i * step : i * step + self.window_size])
                
            self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(X, -1)
        else:
            self.sample_num = data.shape[0]

            X = torch.zeros((data.shape[0], data.shape[1]))            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(data[i,:])

            self.samples, self.targets = X, X            

    def __len__(self):
        if self.data.shape[1] == 1:
            return self.sample_num
        else:
            return self.data.shape[0]
    
    def __getitem__(self, index):
        input_mask = np.ones(self.window_size)

        if self.data.shape[1] == 1:
            return self.samples[index, :, :], input_mask
        else:
            if index < self.data.shape[0] - self.window_size:
                return self.samples[index:index+self.window_size, :], input_mask
            else:
                return self.samples[-self.window_size:, :], input_mask
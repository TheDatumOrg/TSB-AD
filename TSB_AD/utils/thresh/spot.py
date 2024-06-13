#  This function is adapted from [peak-over-threshold] by [cbhua]
#  Original source: [https://github.com/cbhua/peak-over-threshold]

import numpy as np 

from .pot import pot
from .grimshaw import grimshaw


def spot(data:np.array, num_init:int, risk:float):
    ''' Streaming Peak over Threshold

    Reference: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        risk: detection level

    Returns:
        logs: 't' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'t': [], 'a': []}

    init_data = data[:num_init]
    rest_data = data[num_init:]

    z, t = pot(init_data)
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['t'] = [z] * num_init

    for index, x in enumerate(rest_data):
        if x > z:
            logs['a'].append(index + num_init)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            k = k + 1

        logs['t'].append(z)
    
    return logs
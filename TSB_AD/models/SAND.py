
import time
import os
import math
import pickle
import sys
from tqdm import tqdm
import pandas as pd


import numpy as np
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft

from tslearn.clustering import KShape
# from tslearn.cycc import cdist_normalized_cc, y_shifted_sbd_vec
from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec
from tslearn.utils import to_time_series_dataset,to_time_series

import stumpy



class SAND():
    

    """
    Online and offline method that use a set of weighted subsequences (Theta) to identify anomalies. 
    The anomalies are identified by computing the distance of a given subsequence (the targeted 
    subsequence to analyze) to Theta
    ----------
    subsequence_length : int : subsequence length to analyze
    pattern_length : int (greater than pattern length): length of the subsequences in Theta
    k : int (greater than 1) : number of subsequences in Theta

    online : Boolean, Compute the analysis online or offline
    - Online: run per batch the model update and the computation of the score
    (requires the set alpha, init_length, and batch_size)
    - Offline: run the model for one unique batch

    alpha : float ([0,1]) : update rate (used in Online mode only)
    init_length : int (greater than subsequence_length) : length of the initial batch (used in Online mode only)
    batch_size : int (greater than subsequence_length) : length of the batches (used in Online mode only)
    """
    def __init__(self,pattern_length,subsequence_length,k=6):
        
        # Configuration parameter
        self.current_time = 0
        self.mean = -1
        self.std = -1

        # algorithm parameter
        self.k = k
        self.subsequence_length = subsequence_length
        self.pattern_length = pattern_length
        
        # real time evolving storage
        self.clusters = []
        self.new_clusters_dist = []
        self.nm_current_weight = []
        self.S = []
        self.clusters_subseqs = []
    
    
    """
    Build the model and compute the anoamly score
    ----------
    X : np.array or List, the time series to analyse
    
    online : Boolean, Compute the analysis online or offline
    - Online: run per batch the model update and the computation of the score
    (requires the set alpha, init_length, and batch_size)
    - Offline: run the model for one unique batch

    alpha : float ([0,1]) : update rate (used in Online mode only)
    init_length : int (greater than subsequence_length) : length of the initial batch (used in Online mode only)
    batch_size : int (greater than subsequence_length) : length of the batches (used in Online mode only)
    overlapping rate (smaller than len(X)//2 and batch_size//2) : Number points seperating subsequences in the time series.
    """
    def fit(self,X, y=None, online=False, alpha=None, init_length=None, batch_size=None, overlaping_rate=10, verbose=True):
        # Take subsequence every 'overlaping_rate' points
        # Change it to 1 for completely overlapping subsequences 
        # Change it to 'subsequence_length' for non-overlapping subsequences 
        # Change it to 'subsequence_length//4' for non-trivial matching subsequences 
        self.overlaping_rate = overlaping_rate
        self.ts = list(X)
        self.decision_scores_ = []

        if online:
            if (alpha is None) or (init_length is None) or (batch_size is None):
                print("You must specify a value for alpha, init_length, and batch_size")
                return None
            
            self.alpha = alpha
            self.init_length = init_length
            self.batch_size = batch_size

            if verbose:
                print(self.current_time,end='-->')
            
            self._initialize()
            self._set_normal_model()
            self.decision_scores_ = self._run(self.ts[:min(len(self.ts),self.current_time)])

            while self.current_time < len(self.ts)-self.subsequence_length:

                if verbose:
                    print(self.current_time,end='-->')

                self._run_next_batch()
                self._set_normal_model()
                if self.current_time < len(self.ts)-self.subsequence_length:
                    self.decision_scores_ += self._run(self.ts[self.current_time-self.batch_size:min(len(self.ts),self.current_time)])
                else:
                    self.decision_scores_ += self._run(self.ts[self.current_time-self.batch_size:])

            if verbose:
                print("[STOP]: score length {}".format(len(self.decision_scores_)))


        else:
            self.init_length = len(X)
            self.alpha = 0.5
            self.batch_size = 0
            
            self._initialize()
            self._set_normal_model()
            self.decision_scores_ = self._run(self.ts)

        self.decision_scores_ = np.array(self.decision_scores_)


    # Computation of the anomaly score
    def _run(self,ts):
        all_join = []
        
        all_activated_weighted = []
        if len(self.nm_current_weight) != len(self.weights):
            self.nm_current_weight = self.nm_current_weight + self.weights[len(self.nm_current_weight):]

        for index_name in range(len(self.clusters)):
            if self.nm_current_weight[index_name]> 0:
                join = stumpy.stump(ts,self.pattern_length,self.clusters[index_name][0],ignore_trivial = False)[:,0]
                join = np.array(join)
                join = np.nan_to_num(join)
                all_join.append(join)

        join = [0]*len(all_join[0])
        
        for sub_join,scores_sub_join,scores_sub_join_old,t_decay in zip(all_join,self.weights,self.nm_current_weight,self.time_decay):
            new_w = float(scores_sub_join)/float(1+max(0,t_decay-self.batch_size))
            update_w = float(1-self.alpha)*float(scores_sub_join_old) + float(self.alpha)*float(new_w)

            join = [float(j) + float(sub_j)*update_w for j,sub_j in zip(list(join),list(sub_join))]
            all_activated_weighted.append(update_w)
        
        join = join + [join[-1]]*(self.pattern_length-1)
        join = np.array(join)/np.sum(all_activated_weighted)
        join = self._running_mean(join,self.pattern_length)
        join = [join[0]]*(self.pattern_length-1) + list(join)
        
        self.nm_current_weight = all_activated_weighted
        if self.mean == -1:
            self.mean = np.mean(join)
            self.std = np.std(join)
        else:
            self.mean = (1-self.alpha)*self.mean + self.alpha*np.mean(join)
            self.std = (1-self.alpha)*self.std + self.alpha*np.std(join)

        join = (np.array(join) - self.mean)/self.std
        


        return list(np.nan_to_num(join))


    # MAIN METHODS:
    # - Initialization
    # - Theta update for next batch
    # - Score computaiton

    # Initialization of the model
    def _initialize(self):
        
        cluster_subseqs,clusters = self._kshape_subsequence(initialization=True)

        all_mean_dist = []
        for i,(cluster,cluster_subseq) in enumerate(zip(clusters,cluster_subseqs)):
            self._set_initial_S(cluster_subseq,i,cluster[0])
            all_mean_dist.append(self._compute_mean_dist(cluster[0],cluster[1]))

        self.clusters = clusters
        self.new_clusters_dist = all_mean_dist
        self.current_time = self.init_length




    # Model update for next batch
    def _run_next_batch(self):
        
        # Run K-Shape algorithm on the subsequences of the current batch
        cluster_subseqs,clusters = self._kshape_subsequence(initialization=False)

        #self.new_clusters_subseqs = cluster_subseqs
        self.new_clusters_to_merge = clusters

        to_add = [[] for i in range(len(self.clusters))]
        new_c = []
        
        # Finding the clusters that match exisiting clusters
        # - Storing in to_add all the clusters that have to be merged with the existing clusters
        # - Storing in new_c tyhe new clusters to be added.
        for cluster,cluster_subseq in zip(clusters,cluster_subseqs):
            min_dist = np.Inf
            tmp_index = -1
            for index_o,origin_cluster in enumerate(self.clusters):
                new_dist = self._sbd(origin_cluster[0],cluster[0])[0]
                if min_dist > new_dist:
                    min_dist = new_dist
                    tmp_index = index_o
            if tmp_index != -1: 
                if min_dist < self.new_clusters_dist[tmp_index]:
                    to_add[tmp_index].append((cluster,cluster_subseq))
                else:
                    new_c.append((cluster,cluster_subseq))
        
        self.to_add = to_add
        self.new_c = new_c
        
        new_clusters = []
        all_mean_dist = []
        # Merging existing clusters with new clusters
        for i,(cur_c,t_a) in enumerate(zip(self.clusters,to_add)): 
            # Check if new subsequences to add
            if len(t_a) > 0:
                all_index = cur_c[1]
                all_sub_to_add = []
                for t_a_s in t_a:
                    all_index += t_a_s[0][1]
                    all_sub_to_add +=  t_a_s[1]

                # Updating the centroid shape
                new_centroid,_ = self._extract_shape_stream(all_sub_to_add,i,cur_c[0],initial=False)
                new_clusters.append((self._clean_cluster_tslearn(new_centroid),all_index))

                # Updating the intra cluster distance
                dist_to_add = self._compute_mean_dist(cur_c[0],all_index)
                ratio = float(len(cur_c[1]))/float(len(cur_c[1]) + len(all_index))
                all_mean_dist.append( (ratio) * self.new_clusters_dist[i] + (1.0 - ratio) * dist_to_add )

            # If no new subsequences to add, copy the old cluster
            else:
                new_clusters.append(cur_c)
                all_mean_dist.append(self.new_clusters_dist[i])
        
        # Adding new clusters
        for i,t_a in enumerate(new_c):
            self._set_initial_S(t_a[1],len(self.clusters) + i,t_a[0][0])
            new_clusters.append((t_a[0][0],t_a[0][1]))
            all_mean_dist.append(self._compute_mean_dist(t_a[0][0],t_a[0][1]))
        

        self.clusters = new_clusters
        self.new_clusters_dist = all_mean_dist
        self.current_time = self.current_time + self.batch_size

    
    # SBD distance
    def _sbd(self,x, y):
        ncc = self._ncc_c(x, y)
        idx = ncc.argmax()
        dist = 1 - ncc[idx]
        return dist, None

    # Core clustering computation unit
    def _kshape_subsequence(self,initialization=True):
        all_subsequences = []
        idxs = []
        
        if initialization:
            nb_subsequence = self.init_length
        else:
            nb_subsequence = self.batch_size

        for i in range(self.current_time,min(self.current_time + nb_subsequence,len(self.ts)-self.subsequence_length),self.overlaping_rate):
            all_subsequences.append(self.ts[i:i+self.subsequence_length])
            idxs.append(i)
        
        ks = KShape(n_clusters=self.k,verbose=False)
        list_label = ks.fit_predict(np.array(all_subsequences))
        

        cluster_subseq = [[] for i in range(self.k)]
        cluster_idx = [[] for i in range(self.k)]
        for lbl,idx in zip(list_label,idxs):
            cluster_idx[lbl].append(idx)
            cluster_subseq[lbl].append(self.ts[idx:idx+self.subsequence_length])
        
        # safety check
        new_cluster_subseq = []
        clusters = []

        for i in range(self.k):
            if len(cluster_subseq[i]) > 0:
                new_cluster_subseq.append(cluster_subseq[i])
                clusters.append((self._clean_cluster_tslearn(ks.cluster_centers_[i]),cluster_idx[i]))
        return new_cluster_subseq,clusters




    # Model elements update
    def _set_normal_model(self):
        Frequency = []
        Centrality = []
        Time_decay = []
        for i,nm in enumerate(self.clusters):
            Frequency.append(float(len(nm[1])))
            Time_decay.append(float(self.current_time)-float(nm[1][-1]))
            dist_nms = 0
            for j,nm_t in enumerate(self.clusters):
                if j != i:
                    dist_nms += self._sbd(nm[0],nm_t[0])[0]
            Centrality.append(dist_nms)
            
        Frequency = list((np.array(Frequency) - min(Frequency))/(max(Frequency) - min(Frequency)+0.0000001)+1)
        Centrality = list((np.array(Centrality) - min(Centrality))/(max(Centrality) - min(Centrality)+0.0000001)+1)
        
        weights = []
        for f,c,t in zip(Frequency,Centrality,Time_decay):
            weights.append(float(f)**2/float(c))
        
        self.weights = weights
        self.time_decay = Time_decay
        
    # Setting in memory the matrix S
    def _set_initial_S(self,X,idx,cluster_centers):
        X = to_time_series_dataset(X)
        cluster_centers = to_time_series(cluster_centers)
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(cluster_centers, X,
                        norm_ref=-1,
                        norms_dataset=np.linalg.norm(X, axis=(1, 2)))
        S = np.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        self.S.append(S)

    # Computation of the updated centroid
    def _extract_shape_stream(self,X,idx,cluster_centers,initial=True):
        X = to_time_series_dataset(X)
        cluster_centers = to_time_series(cluster_centers)
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(cluster_centers, X,
                        norm_ref=-1,
                        norms_dataset=np.linalg.norm(X, axis=(1, 2)))
        S = np.dot(Xp[:, :, 0].T, Xp[:, :, 0])

        if not initial:    
            S = S + self.S[idx]
        self.S[idx] = S
        Q = np.eye(sz) - np.ones((sz, sz)) / sz
        M = np.dot(Q.T, np.dot(S, Q))
        _, vec = np.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))
        dist_plus_mu = np.sum(np.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = np.sum(np.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return self._zscore(mu_k, ddof=1),S

    # Reset value of a cluster
    def _clean_cluster_tslearn(self,cluster):
        return np.array([val[0] for val in cluster])
    
    # Compute mean distance of a element in a cluster
    def _compute_mean_dist(self,cluster,all_index):
        dist_all = []
        for i in all_index:
            dist_all.append(self._sbd(self.ts[i:i+self.subsequence_length],cluster)[0])
        return np.mean(dist_all)

    def _running_mean(self,x,N):
        return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N

    def _ncc_c(self,x, y):
        den = np.array(norm(x) * norm(y))
        den[den == 0] = np.Inf

        x_len = len(x)
        fft_size = 1 << (2*x_len-1).bit_length()
        cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
        cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
        return np.real(cc) / den

    def _zscore(self,a, axis=0, ddof=0):
        a = np.asanyarray(a)
        mns = a.mean(axis=axis)
        sstd = a.std(axis=axis, ddof=ddof)
        if axis and mns.ndim < a.ndim:
            res = ((a - np.expand_dims(mns, axis=axis)) /
                   np.expand_dims(sstd, axis=axis))
        else:
            res = (a - mns) / sstd
        return np.nan_to_num(res)
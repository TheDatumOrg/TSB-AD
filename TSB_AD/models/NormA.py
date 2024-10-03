# Authors: Paul Boniol, Themis Palpanas
# Date: 08/07/2020
# copyright retained by the authors
# algorithms protected by patent application FR2003946
# code provided as is, and can be used only for research purposes
# the authors and their institution have no liability and bear no responsibility for any damages, or losses, of any kind resulting by the use of this code
#
# Reference using:
#
# Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas: Automated Anomaly Detection in Large Sequences. ICDE 2020: 1834-1837
#
# Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed Meftah. Emmanuel Remy. Unsupervised and Scalable Subsequence Anomaly Detection in Large Data Series. International Journal on Very Large Data Bases (VLDBJ), 2021
#



import numpy as np
import pandas as pd
from scipy import signal
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
from tslearn.clustering import KShape
import stumpy
import math
import random


class NORMA():
    def __init__(self, pattern_length, nm_size, clustering='hierarchical',percentage_sel=0.4,overlapping_factor=1, sampling_division=10,number_of_cluster=10):
        self.pattern_length = pattern_length
        self.nm_size=nm_size
        self.clustering=clustering
        self.percentage_sel=percentage_sel
        self.overlapping_factor=overlapping_factor
        self.sampling_division=sampling_division
        self.number_of_cluster=number_of_cluster
        self.model_name = 'NORMA'

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = X.squeeze()
        score,scores_nms,nms = self.run(X)
        # print('score: ', score.shape)
        # print('scores_nms: ', scores_nms)
        # print('nms: ', len(nms))
        self.nms = nms
        self.scores_nms=scores_nms
        self.decision_scores_ = score
        return self
    
    
    def run(self, ts):
		
		#Build Normal Model
        if self.clustering == "hierarchical":
            recurrent_sequence,sequence_rec = extract_recurrent_sequences_random(
				ts, 
				self.nm_size,
				percentage_sel=self.percentage_sel,
				overlapping_factor=self.overlapping_factor,
				sampling_division=self.sampling_division)
            print("Normal Model Length/number of subsequences selected: ",np.shape(recurrent_sequence))
            if len(sequence_rec) > 1:
                listcluster,dendogram = clustering_method(recurrent_sequence)
                new_nms,scores_nms= choose_normalmodel(listcluster,recurrent_sequence, sequence_rec)
            else:
                new_nms = [list(recurrent_sequence['0'].values)]
                scores_nms  = [1]

            nms = []
            for nm in new_nms:
                to_add = []
                for val in nm:
                    to_add.append(val)
                nms.append(to_add)

        elif self.clustering == "kshape":
            recurrent_sequence,sequence_rec = extract_recurrent_sequences_random_kshape(
				ts, 
				self.nm_size,
				percentage_sel=self.percentage_sel,
				overlapping_factor=self.overlapping_factor,
				sampling_division=self.sampling_division)
            print("Normal Model Length/number of subsequences selected: ",np.shape(recurrent_sequence))
            if len(sequence_rec) > 1:
                to_cluster = [recurrent_sequence[col] for col in recurrent_sequence.columns]
                ks = KShape(n_clusters=self.number_of_cluster, n_init=1, random_state=0).fit(np.array(to_cluster))
                listcluster = list(ks.labels_)
                listcluster = [cl +1 for cl in listcluster]
                new_nms,scores_nms= choose_normalmodel_kshape(ks,listcluster,recurrent_sequence, sequence_rec)
            else:
                new_nms = [list(recurrent_sequence['0'].values)]
                scores_nms  = [1]

            nms = []
            for nm in new_nms:
                to_add = []
                for val in nm:
                    try:
                        to_add.append(val[0])
                    except:
                        to_add.append(val)
                nms.append(to_add)

        self.normalmodel = [nms,scores_nms]
		
		# Compute score
        all_join = []
        for index_name in range(len(nms)):            
            join = stumpy.stump(ts,self.pattern_length,nms[index_name],ignore_trivial = False)[:,0]     # (ts_len-pattern_length+1, )
 			#join,_ = mp.join(nm_name + '/' + str(index_name),ts_name,len(nms[index_name]),len(ts), self.pattern_length)
            join = np.array(join)
 			#join = (join - min(join))/(max(join) - min(join))
            all_join.append(join)
            print('nms[index_name]: ', np.array(nms[index_name]).shape)

        join = [0]*len(all_join[0])
        for sub_join,scores_sub_join in zip(all_join,scores_nms):
            join = [float(j) + float(sub_j)*float(scores_sub_join) for j,sub_j in zip(list(join),list(sub_join))]

        join = np.array(join)
        join_n = running_mean(join,self.pattern_length)
		#reshifting the score time series
        join_n = np.array([join_n[0]]*(self.pattern_length//2) + list(join_n) + [join_n[-1]]*(self.pattern_length//2))
        return join_n,scores_nms,nms
    
    
###################################################################################################
################################# EXTRATION SEQUENCE FUNCTION ####################################
###################################################################################################

####### Align the recurrent sequences #######
def _unshift_series(ts, sequence_rec,normalmodel_size):
	result = []
	ref = ts[sequence_rec[0][0]:sequence_rec[0][1]]
	for seq in sequence_rec:
		shift = (np.argmax(signal.correlate(ref, ts[seq[0]:seq[1]])) - len(ts[seq[0]:seq[1]]))
		if (len(ts[seq[0]-int(shift):seq[1]-int(shift)]) == normalmodel_size):
			result.append([seq[0]-int(shift),seq[1]-int(shift)])
	return result

def extract_recurrent_sequences_random(ts, normalmodel_size, 
                                       percentage_sel = 0.2, overlapping_factor=1, sampling_division=10):
    """
    		INPUT
   			ts: a list representing the time series
   			normalmodel_size: an integer representing the size of the normalmodel
   							  you want to generate
   			percentage_sel: the percentage of the dataset to sample respect to
   							all the possible sequences to select (depends also on the overlapping_factor).
   							Values range between [0-1]
   			overlapping_factor: the overlapping factor to exclude:
    								1 is no overlapping, 0 is total overlapping allowed.
    								Values range between [0-1]
   			sampling_division: the number of chunk in which we divide our time series during sampling.
    							   Values range between [0-1]
    		OUTPUT
   			tuple(recurrent_sequence, sequence_rec)
    
   			recurrent_sequence: a panda dataframe containg all the recurrent sequences, one per column
   			sequence_rec: a list of couple(start,end) of each recurrent sequence in the original time series
    """
    if overlapping_factor == 0:
        recurrent_seq_num = (len(ts) - normalmodel_size)
    else:
        recurrent_seq_num = ((len(ts) - normalmodel_size) // (normalmodel_size*overlapping_factor))
        
    if len(ts) // sampling_division <= normalmodel_size:   # sampling_division is too large
        sampling_division = max(1,int(len(ts)/(1.5*normalmodel_size)))
        
    recurrent_seq_num = int(recurrent_seq_num * percentage_sel)
    recurrent_seq_4chunk = max(1, recurrent_seq_num // sampling_division)
    len_chunk = len(ts) // sampling_division
    sequence_rec = []

    for i in range(0,sampling_division):
        possible_idx = list(range(i*len_chunk,
                                  min(i*len_chunk + len_chunk, len(ts)-1) - normalmodel_size,
                                  max(1,int(normalmodel_size*overlapping_factor))))
    
        selected_idx = random.sample(possible_idx,min(recurrent_seq_4chunk,len(possible_idx)))
        for idx in selected_idx:
            sequence_rec.append((idx,idx+normalmodel_size))
    
		
	####### try to align the recurrent sequences #######
    sequence_rec = _unshift_series(ts,sequence_rec,normalmodel_size)

    recurrent_sequence = pd.DataFrame()
    for i,sr in enumerate(sequence_rec):
        recurrent_sequence[str(i)] = ts[(sr[0]):(sr[1])]

    return recurrent_sequence, sequence_rec


def extract_recurrent_sequences_random_kshape(ts, normalmodel_size,
									   percentage_sel = 0.2, overlapping_factor=1, sampling_division=10):
    '''
		INPUT
 			ts: a list representing the time series
 			normalmodel_size: an integer representing the size of the normalmodel
 							  you want to generate
 			percentage_sel: the percentage of the dataset to sample respect to
 							all the possible sequences to select (depends also on the overlapping_factor).
 							Values range between [0-1]
 			overlapping_factor: the overlapping factor to exclude:
								1 is no overlapping, 0 is total overlapping allowed.
								Values range between [0-1]
 			sampling_division: the number of chunk in which we divide our time series during sampling.
							   Values range between [0-1]
		OUTPUT
 			tuple(recurrent_sequence, sequence_rec)

 			recurrent_sequence: a panda dataframe containg all the recurrent sequences, one per column
 			sequence_rec: a list of couple(start,end) of each recurrent sequence in the original time series

    '''
    if overlapping_factor == 0: 
        recurrent_seq_num = (len(ts) - normalmodel_size)
    else:
        recurrent_seq_num = ((len(ts) - normalmodel_size)//(normalmodel_size*overlapping_factor))
        
    if len(ts) // sampling_division <= normalmodel_size:   # sampling_division is too large
        sampling_division = max(1,int(len(ts)/(1.5*normalmodel_size)))
        
    recurrent_seq_num = int(recurrent_seq_num * percentage_sel)
    recurrent_seq_4chunk = max(1, recurrent_seq_num // sampling_division)
    len_chunk = len(ts) // sampling_division
    sequence_rec = []

    for i in range(0,sampling_division):
        possible_idx = list(range(i*len_chunk,
								min(i*len_chunk + len_chunk, len(ts)-1) - normalmodel_size,
                                max(1,int(normalmodel_size*overlapping_factor))))
        selected_idx = random.sample(possible_idx,min(recurrent_seq_4chunk,len(possible_idx)))
        for idx in selected_idx:
            sequence_rec.append((idx,idx+normalmodel_size))
		
    recurrent_sequence = pd.DataFrame()
    for i,sr in enumerate(sequence_rec):
        recurrent_sequence[str(i)] = ts[(sr[0]):(sr[1])]

    return recurrent_sequence, sequence_rec



#######################################################################
#####				     CLUSTERING FUNCTIONS				     ######
#######################################################################

##### EXPENSIVE OPERATIONS #####
def generate_dendrogram(recurrent_sequence, corr_method="pearson",linkage_method="complete",metric='euclidean'):
    """
        INPUT
            recurrent_sequence: A dataframe containg the list of time series in the columns to cluster
            corr_method: The correlation method to use in order to create the dendogram used to cluster
                            opt('pearson', 'kendall', 'spearman')
            linkage_method: The linkage method to use in order to create the dendogram used to cluster
                            check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            metric: the metric used to crete the dendogram
                    check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

        OUTPUT
            tuple(correlation_matrix, dendo, distances)
            correlation_matrix: the correlation_matrix of the recurrent sequences
            dendo: a Dendrogram representation
            distances: all the pairwise distance of the matrix profile
    """
    if len(recurrent_sequence.columns) == 1:
        return None, None, None

    correlation_matrix = recurrent_sequence.corr(method = corr_method)
    correlation_matrix = correlation_matrix.fillna(0)
    dendo = hac.linkage(correlation_matrix, linkage_method, metric=metric)
    distances = hac.distance.pdist(correlation_matrix, metric=metric)
    return correlation_matrix, dendo, distances


def cutting_method(recurrent_sequence, correlation_matrix, dendo, distances,
                   cut_method="max", cluster_level=0.33):
    """
        INPUT
            recurrent_sequence: A dataframe containg the list of time series in the columns to cluster
            correlation_matrix: A dataframe matrix representing the correlation_matrix of the recurrent sequences
            dendo: the dendrogram of the recurrent sequences
            distances: the pairwise distances of the correlation_matrix
            cut_method: The method used to cut the dendogram and generate the clusters
                            opt('max','minmax','auto')
            cluster_level: A constant variable used to tune the cluster cutting
        OUTPUT
            listcluster: is a list containg all the clusters generated
    """
    if cut_method not in ['max','minmax','auto']:
        raise ValueError("cut_method must be in ['max','minmax','auto']")


    if cut_method == "max":
        listcluster = fcluster(dendo,cluster_level*distances.max(),'distance')

    elif cut_method == "minmax":
        listcluster = fcluster(dendo,(cluster_level * (distances.max()-distances.min()))+ distances.min(),'distance')

    elif cut_method == "auto":
		###### MDL VERSION TOP DOWN ######
        DicBitSavedForCluster = None
        totalBitSaved = None
        start = max(0,distances.max()-0.0001) # starting point (top cut)
        end =  max(0,distances.min()-0.0001) # end point top cut
        listclusterUpperMost= fcluster(dendo, start, 'distance')
        nunmberOfCluster = len(set(listclusterUpperMost))
        setDisttemp = set(distances) - set([distances.min()])
        step = np.min(list(setDisttemp)) - distances.min()

        #time saver 
        step = max(step,0.0001)
        
        listClusterReturn = listclusterUpperMost
        DdlClusters, centerWithMinDl, chosenCluster, sumDl, DicBitSavedForCluster, totalBitSaved = returnClustersMDL_AndSumMDL(recurrent_sequence, listClusterReturn)
        lastNumberCluster = nunmberOfCluster
        # lastSumDL = sumDl
        # bestCenter = centerWithMinDl
        # bestChosenClusterNumber = chosenCluster
        level = start-step

        while (level>=end):
            listclusterActual= fcluster(dendo, level, 'distance')
            nunmberOfCluster = len(set(listclusterActual))
            if(lastNumberCluster<nunmberOfCluster):
                DdlClusters, centerWithMinDl, chosenCluster, sumDl, DicBitSavedForClusterAc,totalBitSavedAc = returnClustersMDL_AndSumMDL(recurrent_sequence,listclusterActual)
                lastNumberCluster = nunmberOfCluster
                if(totalBitSaved<totalBitSavedAc):
                    totalBitSaved = totalBitSavedAc
                    # lastSumDL=sumDl
                    # bestCenter = centerWithMinDl
                    # bestChosenClusterNumber = chosenCluster
                    listClusterReturn = listclusterActual
                else:
                    break
            level = level - step
        listcluster = listClusterReturn
    return listcluster

def clustering_method(recurrent_sequence, corr_method="pearson",linkage_method="complete", cut_method="max",
                     cluster_level=0.33, metric='euclidean'): #this metric is the default scipy metric for the used functions
    """
        INPUT
            recurrent_sequence: A dataframe containg the list of time series in the columns to cluster
            corr_method: The correlation method to use in order to create the dendogram used to cluster
                            opt('pearson', 'kendall', 'spearman')
            linkage_method: The linkage method to use in order to create the dendogram used to cluster
                            check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            cut_method: The method used to cut the dendogram and generate the clusters
                            opt('max','minmax','auto')
            cluster_level: A constant variable used to tune the cluster cutting
            metric: the metric used to crete the dendogram
                    check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

        OUTPUT
            tuple(listcluster, dendo)
            listcluster: is a list containg all the clusters generated
            dendo: a Dendrogram representation
    """
    correlation_matrix, dendo, distances = generate_dendrogram(recurrent_sequence, corr_method=corr_method,
                                                                linkage_method=linkage_method, metric=metric)
    listcluster = cutting_method(recurrent_sequence, correlation_matrix, dendo, distances,
                                    cut_method=cut_method, cluster_level=cluster_level)
    return listcluster, dendo





def choose_normalmodel(listcluster,recurrent_sequence, sequence_rec):
    """
        INPUT
            listcluster: a list representing all the candidate to score
            recurrent_sequence:  dataframe of all the recurrent sequences
            sequence_rec:  a list of couple(start,end) of each recurrent sequence in the original time series
			score_funtion: the scoring function used to evaluate the clusters
						   opt('standard','extended')
        OUTPUT
            tuple(normalmodel,scores,min_max_index,cluster_mean)

            normalmodel: a list containg the normal model
            scores: a dictionary containg the following scores
                - score_time
                - score_mean
                - score_weight
            min_max_index: a list containg all the min max indexes for all the candidates
			cluster_mean: a list of all the centroids of the cluster
    """
    cluster_mean, min_max_index, score_time, score_mean, score_weight, all_index_p  = [], [], [], [], [], []
    # getting all the scoring variable

    for k in range(len(set(listcluster))):
        mean = pd.DataFrame()
        count = 0
        index_seq = []
        for i in range(len(listcluster)):
            if listcluster[i] == k+1 :
                index_seq.append(sequence_rec[i])
                mean_t = np.mean(recurrent_sequence[str(i)].values)
                # std_t = np.std(recurrent_sequence[str(i)].values)

                data_to_join = [(float(j) - mean_t)/1 for j in recurrent_sequence[str(i)].values]
                mean[str(count)] = data_to_join
                count += 1

        cluster_mean.append(mean.mean(axis=1).values)
        score_weight.append(count)

        i_seq_m = [ (i_s[1] + i_s[0])/2 for i_s in index_seq]
        all_index_p.append(i_seq_m)
        score_time.append(np.mean(i_seq_m))
        min_max_index.append([np.min(i_seq_m), np.max(i_seq_m)])
        # diff_off = np.diff(np.sort([i_s[0] for i_s in index_seq]))
        #score_distribution.append(np.sum([np.abs((d_f/(min_max_index[-1][1] - min_max_index[-1][0])) -
        #                            ((count-1)/(min_max_index[-1][1] - min_max_index[-1][0]))) for d_f in diff_off]))
    ########################################

    ###### pre process data to normalize later #####
    cluster_mean_diff = []
    min_max_diff = [np.diff(min_max)[0] for min_max in min_max_index]
    for c_mean in cluster_mean:
        cluster_mean_diff.append(np.sum([np.linalg.norm((c_mean - x), ord=1) for x in cluster_mean]))

    ###### compute scores ########
    for weight,min_max,c_mean in zip(score_weight,min_max_diff,cluster_mean_diff):
        
        weight_n = (float(weight - np.min(score_weight))/float(np.max(score_weight)-np.min(score_weight)+1))+1.0
        min_max_n = (float(min_max - np.min(min_max_diff))/float(np.max(min_max_diff)-np.min(min_max_diff)+1))+1.0
        #dist_n = (float(dist - np.min(score_distribution))/float(np.max(score_distribution)-np.min(score_distribution)+1))+1.0
        c_mean_n = (float(c_mean - np.min(cluster_mean_diff))/float(np.max(cluster_mean_diff)-np.min(cluster_mean_diff)+1))+1.0

        
        score_mean.append((weight_n*weight_n * min_max_n) / c_mean_n)
        

    return cluster_mean,score_mean


def choose_normalmodel_kshape(ks,listcluster,recurrent_sequence, sequence_rec):
   
    all_index_p, cluster_mean, min_max_index,score_time, score_mean, score_weight = [], [], [], [], [], []
    
    ########################################
    # getting all the scoring variable

    for k,idx_c in enumerate(set(listcluster)):
        mean = pd.DataFrame()
        count = 0
        index_seq = []
        for i in range(len(listcluster)):
            if listcluster[i] == idx_c :
                index_seq.append(sequence_rec[i])
                mean_t = np.mean(recurrent_sequence[str(i)].values)
                std_t = np.std(recurrent_sequence[str(i)].values)

                data_to_join = [(float(j) - mean_t)/(std_t) for j in recurrent_sequence[str(i)].values]
                mean[str(count)] = data_to_join
                count += 1

        cluster_mean.append(ks.cluster_centers_[k])
        score_weight.append(count)

        i_seq_m = [ (i_s[1] + i_s[0])/2 for i_s in index_seq]
        all_index_p.append(i_seq_m)
        score_time.append(np.mean(i_seq_m))
        min_max_index.append([np.min(i_seq_m), np.max(i_seq_m)])
        # diff_off = np.diff(np.sort([i_s[0] for i_s in index_seq]))
        #score_distribution.append(np.sum([np.abs((d_f/(min_max_index[-1][1] - min_max_index[-1][0])) -
        #                            ((count-1)/(min_max_index[-1][1] - min_max_index[-1][0]))) for d_f in diff_off]))
    ########################################

    ###### pre process data to normalize later #####
    cluster_mean_diff = []
    min_max_diff = [np.diff(min_max)[0] for min_max in min_max_index]
    for c_mean in cluster_mean:
        cluster_mean_diff.append(np.sum([np.linalg.norm((c_mean - x), ord=1) for x in cluster_mean]))

    ###### compute scores ########
    for weight,min_max,c_mean in zip(score_weight,min_max_diff,cluster_mean_diff):
        #normalize all the scores between (1,2)
        weight_n = (float(weight - np.min(score_weight))/float(np.max(score_weight)-np.min(score_weight)+1))+1.0
        min_max_n = (float(min_max - np.min(min_max_diff))/float(np.max(min_max_diff)-np.min(min_max_diff)+1))+1.0
        #dist_n = (float(dist - np.min(score_distribution))/float(np.max(score_distribution)-np.min(score_distribution)+1))+1.0
        c_mean_n = (float(c_mean - np.min(cluster_mean_diff))/float(np.max(cluster_mean_diff)-np.min(cluster_mean_diff)+1))+1.0

        score_mean.append((weight_n*weight_n * min_max_n) / c_mean_n)
        
    return cluster_mean,score_mean

###########################################
####### 	AGGREGATE FUNCTIONS		#######
###########################################

def running_mean(x,N):
	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N


###########################################
####### 	  GET FUNCTIONS			#######
###########################################


# =============================================================================
# def get_sequence_under_threshold(list_y,T,length):
#     result = []
#     list_y = np.array(list_y)
#     if len(list_y) == 0:
#         return result
#     ### Get value under threshold and order it in a ascendent way
#     idx_uT = np.where(list_y<T)[0]
#     idx_uT_ord = idx_uT[np.argsort(list_y[idx_uT])]
#     idx_uT_ord = idx_uT_ord[::-1]
# 
# 
#     if len(idx_uT_ord) == 0:
#         return result
# 
#     ## Remove overlapping sequences given priority to the lowest selfJoin values
#     match_mask = [0] * (len(list_y)) #create the mask
#     for off in idx_uT_ord:
#         is_overlapping_match = False
#         for mask in match_mask[max(0,(off-length)):min(len(match_mask),(off+length))]:
#             if(mask == 1):
#                 is_overlapping_match = True
#                 break
#         if(not is_overlapping_match):
#             result.append([off,off+length])
#             match_mask[off] = 1
#     return result
# =============================================================================



#pynorma/entropy_MDL.py



def findValuesSax(real):
    saxBP = [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53]
    # global saxBP
    pos=0
    for x in saxBP:
        if real>x:
            pos+=1
        else:
            break
    return pos


def computeEntropy(T):
    D={}
    for p in T:
        x = findValuesSax(p)
        if x in D:
            D[x] = D[x] +1
        else:
            D[x] =  1

    entropy = 0
    # nKeys = len(D.keys())
    for k in D.keys():
    #probability of key
        prob = float(D[k])/float(len(T))
        logProb = math.log(prob,2)
        entropy = entropy + prob*logProb

    entropy = -1 * entropy
    return entropy

def computeDescriptionLength(T):
    return (len(T) * computeEntropy(T))



def computeCondDescLength(A,B):
    diff = np.subtract(A,B)
    return computeDescriptionLength(diff)

def returnClustersMDL(dataFrameSubsequences, clusterNumbers):
    Cluster ={}
    ClustersCenters = {}
    numberElementsCluster = {}

    for i in range(len(clusterNumbers)):
        clNunmb = clusterNumbers[i]
        if clNunmb in ClustersCenters:
            listA = dataFrameSubsequences[str(i)].values
            listB = ClustersCenters[clNunmb]
            ClustersCenters[clNunmb] = np.add(listA, listB)
            Cluster[clNunmb].append(dataFrameSubsequences[str(i)].values)
            numberElementsCluster[clNunmb] +=1
        else:
            ClustersCenters[clNunmb]= dataFrameSubsequences[str(i)].values
            Cluster[clNunmb] = [dataFrameSubsequences[str(i)].values]
            numberElementsCluster[clNunmb] = 1

    MdlClusetrs ={}
    minDl = 0
    maxDl = 0
    minDlCenter = None
    bFirst=True
    numberClusterBest = 0
    for k in Cluster.keys():
        maxDlCl = 0
        sumDlCl = 0
        listSeq = Cluster[k]
        center = [ (i/numberElementsCluster[k]) for i in ClustersCenters[k]]
        desL = computeDescriptionLength(center)
        for seq in listSeq:
            dlCond = computeCondDescLength(seq,center)
            sumDlCl = sumDlCl+dlCond
            maxDlCl= max(maxDlCl,dlCond)

        dlc = desL - maxDlCl + sumDlCl
        MdlClusetrs[k] = dlc
        if(bFirst):
            bFirst=False
            minDl = dlc
            maxDl = dlc
            numberClusterBest = k
        else:
            minDl = min(dlc,minDl)
            maxDl = max(dlc,maxDl)
            if(minDl==dlc):
                minDlCenter = center
            numberClusterBest = k

    return MdlClusetrs, minDlCenter, numberClusterBest, minDl

def returnClustersMDL_AndSumMDL(dataFrameSubsequences, clusterNumbers):
    Cluster ={}
    ClustersCenters = {}
    numberElementsCluster = {}

    for i in range(len(clusterNumbers)):
        clNunmb = clusterNumbers[i]
        if clNunmb in ClustersCenters:
            listA = dataFrameSubsequences[str(i)].values
            listB = ClustersCenters[clNunmb]
            ClustersCenters[clNunmb] = np.add(listA, listB)
            Cluster[clNunmb].append(dataFrameSubsequences[str(i)].values)
            numberElementsCluster[clNunmb] +=1
        else:
            ClustersCenters[clNunmb]= dataFrameSubsequences[str(i)].values
            Cluster[clNunmb] = [dataFrameSubsequences[str(i)].values]
            numberElementsCluster[clNunmb] = 1

    MdlClusetrs ={}
    BitSaveCluster = {}
    minDl = 0
    maxDl = 0
    minDlCenter = None
    bFirst=True
    numberClusterBest = 0
    sumUptotalMDL = 0
    totalBitSavedClusters = 0
    for k in Cluster.keys():
        maxDlCl = 0
        sumDlCl = 0
        listSeq = Cluster[k]
        center = [ (i/numberElementsCluster[k]) for i in ClustersCenters[k]]
        desL = computeDescriptionLength(center)
        totalBitSeq = 0
        for seq in listSeq:
            dlCond = computeCondDescLength(seq,center)
            sumDlCl = sumDlCl+dlCond
            maxDlCl= max(maxDlCl,dlCond)
            totalBitSeq+=computeDescriptionLength(seq)

        dlc = desL - maxDlCl + sumDlCl
        MdlClusetrs[k] = dlc
        totalBitSave = totalBitSeq-dlc
        BitSaveCluster[k] = totalBitSave
        totalBitSavedClusters+=totalBitSave
        if(bFirst):
            bFirst=False
            minDl = dlc
            maxDl = dlc
            numberClusterBest = k
            minDlCenter = center
        else:
            minDl = min(dlc,minDl)
            maxDl = max(dlc,maxDl)
            if(minDl==dlc):
                minDlCenter = center
                numberClusterBest = k
        sumUptotalMDL+=dlc

    return MdlClusetrs, minDlCenter, numberClusterBest, sumUptotalMDL, BitSaveCluster, totalBitSavedClusters
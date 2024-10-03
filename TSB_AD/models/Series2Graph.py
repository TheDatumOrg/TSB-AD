# Authors: Paul Boniol, Themis Palpanas
# Date: 08/07/2020
# copyright retained by the authors
# algorithms protected by patent application FR2005261
# code provided as is, and can be used only for research purposes
#
# Reference using:
#
# P. Boniol and T. Palpanas, Series2Graph: Graph-based Subsequence Anomaly Detection in Time Series, PVLDB (2020)
#
# P. Boniol and T. Palpanas and M. Meftah and E. Remy, GraphAn: Graph-based Subsequence Anomaly Detection, demo PVLDB (2020)
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import math
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from tqdm import tqdm

class Series2Graph():

    """
    Series2Graph, a graph-based subsequence time series anomaly detection method,
    which is based on a graph representation of a novel low-dimensionality embedding of subsequences. 
    Series2Graph does znot need labeled instances (like supervised techniques), 
    nor anomaly-free data (like zero-positive learning techniques), and identifies anomalies of varying lengths.
    ----------
    latent: 
    pattern_length : int (smaller than anomaly-query length): length of the subsequence for the graph construction
    rate: 
    graph: the constructed graph structure
    """

    def __init__(self,pattern_length,latent=None,rate=30,pruning=True):
        self.pattern_length = pattern_length
        self.rate = rate
        if latent is not None:
            self.latent = latent
        else:
            self.latent = self.pattern_length//3

        self.graph = {}

        if pruning:
            self.downsample = max(1,self.latent//100)
        else:
            self.downsample = 1

    def fit(self,ts):
        """
        To construct the graph for the given time series.
        ----------
        ts: time series
        """

        min_df_r = ts.min()
        max_df_r = ts.max()

        df_ref = []
        for i in np.arange(min_df_r, max_df_r,(max_df_r-min_df_r)/100):
            tmp = []
            T = [i]*self.pattern_length
            for j in range(self.pattern_length - self.latent):
                tmp.append(sum(x for x in T[j:j+self.latent]))
            df_ref.append(tmp[::self.downsample])
        df_ref = pd.DataFrame(df_ref)
        
        phase_space_1 = self.__build_phase_space(ts)

        pca_1 = PCA(n_components=3)
        pca_1.fit(phase_space_1)
        reduced = pd.DataFrame(pca_1.transform(phase_space_1),columns=[str(i) for i in range(3)])
        reduced_ref = pd.DataFrame(pca_1.transform(df_ref),columns=[str(i) for i in range(3)])

        v_1 = reduced_ref.values[0]

        R = self.__get_rotation_matrix(v_1,[0.0, 0.0, 1.0])
        A = np.dot(R,reduced.T)
        A_ref = np.dot(R,reduced_ref.T)
        A = pd.DataFrame(A.T,columns=['0','1','2'])
        A_ref = pd.DataFrame(A_ref.T,columns=['0','1','2'])
    
        res_dist = self.__get_intersection_from_radius(A,'0','1',rate=self.rate)
        nodes_set,_ = self.__nodes_extraction(A,'0','1',res_dist,self.rate)
        list_edge,time_evo,dict_edge,dict_node = self.__edges_extraction(A,'0','1',nodes_set,rate=self.rate)
    
        G = nx.DiGraph(list_edge)
            
        result = {
            "Graph": G,					# networkx graph
            "list_edge": list_edge,		# list of edges crossed by the time seires ordered on time
            "edge_in_time": time_evo,	# the corresponding time series timestamp for each edge in list_edge
            "edge_weigth": dict_edge,	# Dictionary of weights for each edge
            "node_weigth":dict_node,	# Dictionary of weights for each node
            # TEST MODE:
            "pca_proj": pca_1,			# PCA transformation (test mode)
            "rotation_matrix": R,		# Rotation matrix (test mode)
            "node_set": nodes_set,		# Position of nodes in each radius (test mode)
            }

        self.graph = result

    

    def computeDegree(self,query_length):
        """
        To compute the degree for the given query length.
        ----------
        query_length: the length of the query
        """
        all_score_node_degree = []
        degree = nx.degree(self.graph["Graph"])
        for i in range(0,len(self.graph["edge_in_time"])-int(query_length)-1):
            P_edge = self.graph["list_edge"][self.graph["edge_in_time"][i]:self.graph["edge_in_time"][i+int(query_length)]]
            if len(P_edge) == 0:
                all_score_node_degree.append(degree[self.graph["list_edge"][max(0,self.graph["edge_in_time"][i]-1)][1]])
            else:
                all_score_node_degree.append(np.mean([degree[edge[1]] for edge in P_edge]))
            
        self.all_score_node_degree = all_score_node_degree
    

    def computeNodeWeight(self,query_length):
        all_score_node_weight = []
        for i in range(0,len(self.graph["edge_in_time"])-int(query_length)-1):
            P_edge = self.graph["list_edge"][self.graph["edge_in_time"][i]:self.graph["edge_in_time"][i+int(query_length)]]
            if len(P_edge) == 0:
                all_score_node_weight.append(self.graph["node_weigth"][self.graph["list_edge"][max(0,self.graph["edge_in_time"][i]-1)][1]])
            else:
                all_score_node_weight.append(np.mean([self.graph["node_weigth"][edge[1]] for edge in P_edge]))
        
        self.all_score_node_weight = all_score_node_weight


    def computeEdgeWeight(self,query_length):
        all_score_edge_weight = []
        for i in range(0,len(self.graph["edge_in_time"])-int(query_length)-1):
            P_edge = self.graph["list_edge"][self.graph["edge_in_time"][i]:self.graph["edge_in_time"][i+int(query_length)]]
            if len(P_edge) == 0:
                all_score_edge_weight.append(self.graph["edge_weigth"][str(self.graph["list_edge"][max(0,self.graph["edge_in_time"][i]-1)])])
            else:
                all_score_edge_weight.append(np.mean([self.graph["edge_weigth"][str(edge)] for edge in P_edge]))
        
        self.all_score_edge_weight = all_score_edge_weight



    def score(self,query_length, dataset):
        all_score = [np.nan]
        degree = nx.degree(self.graph["Graph"])
        diff_pq = int(query_length) - int(self.pattern_length)

        for i in range(len(self.graph["edge_in_time"])-diff_pq):
            P_edge = self.graph["list_edge"][self.graph["edge_in_time"][i]:self.graph["edge_in_time"][i+diff_pq]]
            score,len_score = self.__score_P_degree(self.graph["edge_weigth"],P_edge,degree)
            if len_score == 0:
                all_score.append(all_score[-1])
            else:
                all_score.append(score)

        all_score = self.__pandas_fill(np.array(all_score))
        all_score = [-score for score in all_score[1:]]
        all_score = np.array(all_score)
        self.decision_scores_ = all_score
        
    def plot_graph(self):
        edge_size = []
        for edge in self.graph["Graph"].edges():
            edge_size.append(self.graph["list_edge"].count([edge[0],edge[1]]))
        edge_size_b = [float(1+(e - min(edge_size)))/float(1+max(edge_size) - min(edge_size)) for e in edge_size]
        edge_size = [min(e*50,30) for e in edge_size_b]
        pos = nx.nx_agraph.graphviz_layout(nx.Graph(self.graph["list_edge"]),prog="fdp")
        
        plt.figure(figsize=(20,20))
        dict_node = []
        for node in self.graph["Graph"].nodes():
            dict_node.append(self.graph["node_weigth"][node]/5)
        nx.draw(self.graph["Graph"],pos, node_size=dict_node,with_labels=True,width=edge_size)

#####################################################################
####################### Tools ############################
#####################################################################

    def __build_phase_space(self,T,rate=1):
        tmp_glob = []
        current_seq = [0]*self.pattern_length
        first = True
        for i in range(int((len(T) - self.pattern_length)/rate)):
            tmp = []
            it_rate = i*rate
            if first:
                first = False
                for j in range(self.pattern_length - self.latent):
                    tmp.append(sum(x for x in T[it_rate+j:it_rate+j+self.latent]))
                tmp_glob.append(tmp[::self.downsample])
                current_seq = tmp
            else:
                tmp = current_seq[1:]
                tmp.append(sum(x for x in T[it_rate+self.pattern_length-self.latent:it_rate+self.pattern_length]))
                tmp_glob.append(tmp[::self.downsample])
                current_seq = tmp
                
        return pd.DataFrame(tmp_glob)

    def __get_rotation_matrix(self,i_v, unit):
        curve_vec_1 = i_v
        curve_vec_2 = unit
        a,b = (curve_vec_1/ np.linalg.norm(curve_vec_1)).reshape(3), (curve_vec_2/ np.linalg.norm(curve_vec_2)).reshape(3)
        v = np.cross(a,b)
        c = np.dot(a,b)
        s = np.linalg.norm(v)
        I = np.identity(3)
        vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
        k = np.matrix(vXStr)
        r = I + k + k@k * ((1 -c)/(s**2))
        return r

    #####################################################################
    ####################### NODES EXTRACTION ############################
    #####################################################################

    def __distance(self,a,b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def __det(self,a, b):
        return a[0] * b[1] - a[1] * b[0]

    def __line_intersection(self,line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        div = self.__det(xdiff, ydiff)
        if div == 0:
            return None,None

        max_x_1 = max(line1[0][0],line1[1][0])
        max_x_2 = max(line2[0][0],line2[1][0])
        max_y_1 = max(line1[0][1],line1[1][1])
        max_y_2 = max(line2[0][1],line2[1][1])
        
        min_x_1 = min(line1[0][0],line1[1][0])
        min_x_2 = min(line2[0][0],line2[1][0])
        min_y_1 = min(line1[0][1],line1[1][1])
        min_y_2 = min(line2[0][1],line2[1][1])
        
        d = (self.__det(*line1), self.__det(*line2))
        x = self.__det(d, xdiff) / div
        y = self.__det(d, ydiff) / div
        if not(((x <= max_x_1) and (x >= min_x_1)) and ((x <= max_x_2) and (x >= min_x_2))):
            return None,None
        if not(((y <= max_y_1) and (y >= min_y_1)) and ((y <= max_y_2) and (y >= min_y_2))):
            return None,None
        return [x, y], self.__distance(line1[0],[x,y])

    def __find_tuple_interseted(self,proj,line):
        result = []
        dist_l = []
        for i in range(len(proj)-1):
            intersect,dist = self.__line_intersection(line, [proj[i],proj[i+1]])
            if intersect is not None:
                result.append(intersect)
                dist_l.append(dist)
        return [result,dist_l]

    def __PointsInCircum(self,r,n=500):
        return [(math.cos(2*np.pi/n*x)*r,math.sin(2*np.pi/n*x)*r) for x in range(0,n)]


    def __find_closest_node(self,list_maxima_ind,point):
        result_list = [np.abs(maxi - point) for maxi in list_maxima_ind]
        result_list_sorted = sorted(result_list)
        return result_list.index(result_list_sorted[0])
        

    def __find_theta_to_check(self,proj,k,rate):
        k_0 = proj[k][0]
        k_1 = proj[k][1]
        k_1_0 = proj[k+1][0]
        k_1_1 = proj[k+1][1]
        dist_to_0 = np.sqrt(k_0**2 + k_1**2)
        dist_to_1 = np.sqrt(k_1_0**2 + k_1_1**2)
        theta_point = np.arctan2([k_1/dist_to_0],[k_0/dist_to_0])[0]
        theta_point_1 = np.arctan2([k_1_1/dist_to_1],[k_1_0/dist_to_1])[0]
        if theta_point < 0:
            theta_point += 2*np.pi    
        if theta_point_1 < 0:
            theta_point_1 += 2*np.pi    
        theta_point = int(theta_point/(2.0*np.pi) * (rate)) 
        theta_point_1 = int(theta_point_1/(2.0*np.pi) * (rate))
        diff_theta = abs(theta_point - theta_point_1)
        if diff_theta > rate//2:
            if theta_point_1 > rate//2:
                diff_theta = abs(theta_point - (-rate + theta_point_1))
            elif theta_point > rate//2:
                diff_theta = abs((-rate + theta_point) - theta_point_1)
        diff_theta = min(diff_theta,rate//2)
        theta_to_check = [(theta_point + lag) % rate for lag in range(-diff_theta-1,diff_theta+1)]
        return theta_to_check
            

    def __get_intersection_from_radius(self,A,col1,col2,rate=100):
        
        max_1 = max(max(A[col1].values),abs(min(A[col1].values)))
        max_2 = max(max(A[col2].values),abs(min(A[col2].values)))
        set_point = self.__PointsInCircum(np.sqrt(max_1**2 + max_2**2),n=rate)
        previous_node = "not_defined"

        result_dist = [[] for i in range(len(set_point))]

        proj = A[[col1,col2]].values
        for k in range(0,len(A)-1):	
            theta_to_check = self.__find_theta_to_check(proj,k,rate)
            was_found = False
            for i in theta_to_check:
                intersect,dist = self.__line_intersection(
                    [[0,0],set_point[i]],
                    [proj[k],proj[k+1]])
                if intersect is not None:
                    was_found = True
                    result_dist[i].append(dist)
                elif (was_found == True) and intersect is None:
                    break 
        return result_dist


    def __kde_scipy(self,x, x_grid):
        kde = gaussian_kde(x, bw_method='scott')
        return list(kde.evaluate(x_grid))


    def __nodes_extraction(self,A,col1,col2,res_dist,rate=100):
        max_all = max(max(max(A[col1].values),max(A[col2].values)),max(-min(A[col1].values),-min(A[col2].values)))
        max_all = max_all*1.2
        range_val_distrib = np.arange(0, max_all, max_all/250.0)
        list_maxima = []
        list_maxima_val = []
        for segment in range(rate):
            # No interesection on this radius
            if len(res_dist[segment]) == 0:
                list_maxima.append([0])
                list_maxima_val.append([0])
            # Only one point found
            elif len(res_dist[segment]) == 1:
                list_maxima.append([res_dist[segment][0]])
                list_maxima_val.append([0])
            # Multiple points with same values
            elif len(set(res_dist[segment])) == 1:
                list_maxima.append([res_dist[segment][0]])
                list_maxima_val.append([0])
            # Kernel density estimation
            else:
                dist_on_segment = self.__kde_scipy(res_dist[segment], 
                                            range_val_distrib) 
                dist_on_segment = (dist_on_segment - min(dist_on_segment))/(max(dist_on_segment) - min(dist_on_segment))
                maxima = argrelextrema(np.array(dist_on_segment), np.greater)[0]
                if len(maxima) == 0:
                    maxima = np.array([0])
                maxima_ind = [range_val_distrib[val] for val in list(maxima)]
                maxima_val = [dist_on_segment[val] for val in list(maxima)]
                list_maxima.append(maxima_ind)
                list_maxima_val.append(maxima_val)
        return list_maxima,list_maxima_val

    #####################################################################
    ####################### EDGES EXTRACTION ############################
    #####################################################################
        
    def __edges_extraction(self,A,col1,col2,set_nodes,rate=100):
        
        list_edge = []
        new_node_list = []
        edge_in_time = []
    
        dict_edge = {}
        dict_node = {}

        max_1 = max(max(A[col1].values),abs(min(A[col1].values)))
        max_2 = max(max(A[col2].values),abs(min(A[col2].values)))
        set_point = self.__PointsInCircum(np.sqrt(max_1**2 + max_2**2),n=rate)
        previous_node = "not_defined"

        proj = A[[col1,col2]].values
        for k in range(0,len(A)-2):
            
            theta_to_check = self.__find_theta_to_check(proj,k,rate)
            was_found = False
            for i in theta_to_check:
                to_add = self.__find_tuple_interseted(proj[k:k+2],[[0,0],set_point[i]])[1]
                if to_add == [] and not was_found:
                    continue
                elif to_add == [] and was_found:
                    break
                else:
                    was_found = True
                    node_in = self.__find_closest_node(set_nodes[i],to_add[0])
                    
                    if previous_node == "not_defined":
                        previous_node = "{}_{}".format(i,node_in)
                        dict_node[previous_node] = 1
                    else:
                        list_edge.append([previous_node,"{}_{}".format(i,node_in)])
                        
                        if "{}_{}".format(i,node_in) not in dict_node.keys():
                            dict_node["{}_{}".format(i,node_in)] = 1
                        else:
                            dict_node["{}_{}".format(i,node_in)] += 1

                        if str(list_edge[-1]) in dict_edge.keys():
                            dict_edge[str(list_edge[-1])] += 1
                        else:
                            dict_edge[str(list_edge[-1])] = 1
                        previous_node = "{}_{}".format(i,node_in)
                    
            edge_in_time.append(len(list_edge))		

        # fill the missing 2 elements
        edge_in_time += [len(list_edge),len(list_edge)]

        return list_edge,edge_in_time,dict_edge,dict_node
        
    def __pandas_fill(self,arr):
        df = pd.DataFrame(arr)
        df = df.fillna(method='bfill')
        out = df.to_numpy()
        return out

    def __get_nodes_from_P(self,G,node_set,P,latent,length_pattern,pca_1,R,skip=1,rate=100):
        P_space = self.__build_phase_space(P,latent,length_pattern,skip)
        reduced_P = pd.DataFrame(pca_1.transform(P_space),columns=[str(i) for i in range(3)])
        A = np.dot(R,reduced_P.T)
        A = pd.DataFrame(A.T,columns=['0','1','2'])
        list_edge,_,_ = self.__edges_extraction(A,'0','1',node_set,rate)
        return list_edge,A

    def __score_P_degree(self,dict_edge,list_edge_P,node_degree):

        score = np.sum(dict_edge[str(edge)]*(node_degree[edge[0]]-1) for edge in list_edge_P)/float(0.00000001+len(list_edge_P))
        return score,len(list_edge_P)
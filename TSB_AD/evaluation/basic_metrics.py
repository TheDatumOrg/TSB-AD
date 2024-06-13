from sklearn import metrics
import numpy as np
import math

def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

from .cal_vus import RangeAUC_volume
def generate_curve_numba(label,score,slidingWindow):
    # tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)
    
    ## Numba version
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

def generate_curve(label,score,slidingWindow):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = basic_metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)
    
    ## Numba version
    # tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d

class basic_metricor:
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias 
    
    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score 
        else:
            scoreX = score
            
        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L

        
    def labels_conv(self, preds):
        '''return indices of predicted anomaly
        '''

        index = np.where(preds >= 0.5)
        return index[0]
    
    def labels_conv_binary(self, preds):
        '''return predicted label
        '''
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p 


    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0 
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1 
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score
        
    def b(self, i, length):
        bias = self.bias 
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1


    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)
    
    
    def metric_new(self, label, score, preds, plot_ROC=False, alpha=0.2):
        '''input:
               Real labels and anomaly score in prediction
            
           output:
               AUC, 
               Precision, 
               Recall, 
               F-score, 
               Range-precision, 
               Range-recall, 
               Range-Fscore, 
               Precison@k, 
             
            k is chosen to be # of outliers in real labels
        '''
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None
        
        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None
        
        #area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()            
            
        #precision, recall, F
        if preds is None:
            preds = score > (np.mean(score)+3*np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #point-adjust
        adjust_preds = adjust_predicts(score, label, pred=preds)
        PointF1PA = metrics.f1_score(label, adjust_preds)

        #range anomaly 
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]
        
        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))
        
        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k
        
        L = [auc, precision, recall, f, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_PR(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        # plt.figure()
        # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()
        AP = metrics.average_precision_score(label, score)
        return precision, recall, AP

    def metric_best_F(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 0.00001)
        best_f1 = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]
        return best_f1, best_threshold

    def metric_best_RF(self, label, score):

        thresholds = np.linspace(score.min(), score.max(), 100)
        Rf1_scores = []

        for threshold in thresholds:
            preds = (score > threshold).astype(int)

            Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
            Rprecision = self.range_recall_new(preds, label, 0)[0]
            if Rprecision + Rrecall==0:
                Rf=0
            else:
                Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)            
            
            Rf1_scores.append(Rf)

        Best_RF1_Threshold = thresholds[np.argmax(Rf1_scores)]
        Best_RF1 = max(Rf1_scores)
        return Best_RF1, Best_RF1_Threshold


    def range_recall_new(self, labels, preds, alpha):   
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)  
        range_label = self.range_convers_new(labels)
        
        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, p)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        L = []
        i = 0
        j = 0 
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i+=1
                if i >= len(label):
                    break
            j = i+1
            # print('j'+str(j))
            if j >= len(label):
                if j==len(label):
                    L.append((i,j-1))
    
                break
            while label[j] != 0:
                j+=1
                if j >= len(label):
                    L.append((i,j-1))
                    break
            if j >= len(label):
                break
            L.append((i, j-1))
            i = j
        return L
        
    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair 
        preds predicted data
        '''

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score
    
    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count
    
    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            
            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            l0 = int((e-s+1)*percentage)
            
            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
            
            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred
        
        TP = np.sum(product)
        
        # recall = min(TP/P,1)
        P_new = (P+np.sum(labels))/2      # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP/P_new,1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))
        
        
        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1]+1)])>0:
                existence += 1
                
        existence_ratio = existence/len(L)
        # print(existence_ratio)
        
        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall*existence_ratio
        
        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))
        
        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP/N_new
        
        Precision_RangeAUC = TP/np.sum(pred)
        
        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC
    
    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)
        
        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type=='window':
            labels = self.extend_postive_range(labels, window=window)
        else:   
            labels = self.extend_postive_range_individual(labels, percentage=percentage)
        
        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]
        
        for i in np.linspace(0, len(score)-1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score>= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
            
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)
            
        TPR_list.append(1)
        FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
        
        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)
        
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1])/2
        AUC_range = np.sum(width*height)
        
        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = (prec[1:] + prec[:-1])/2
        AP_range = np.sum(width_PR*height_PR)
        
        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec
        
        return AUC_range
        

    # TPR_FPR_window
    def RangeAUC_volume(self, labels_original, score, windowSize):
        score_sorted = -np.sort(-score)
        
        tpr_3d=[]
        fpr_3d=[]
        prec_3d=[]
        
        auc_3d=[]
        ap_3d=[]
        
        window_3d = np.arange(0, windowSize+1, 1)
        P = np.sum(labels_original)
       
        for window in window_3d:
            labels = self.extend_postive_range(labels_original, window)
            
            # print(np.sum(labels))
            L = self.range_convers_new(labels)
            TPR_list = [0]
            FPR_list = [0]
            Precision_list = [1]
            
            for i in np.linspace(0, len(score)-1, 250).astype(int):
                threshold = score_sorted[i]
                # print('thre='+str(threshold))
                pred = score>= threshold
                TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
                
                TPR_list.append(TPR)
                FPR_list.append(FPR)
                Precision_list.append(Precision)
                
            TPR_list.append(1)
            FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
            
            
            tpr = np.array(TPR_list)
            fpr = np.array(FPR_list)
            prec = np.array(Precision_list)
            
            tpr_3d.append(tpr)
            fpr_3d.append(fpr)
            prec_3d.append(prec)
            
            width = fpr[1:] - fpr[:-1]
            height = (tpr[1:] + tpr[:-1])/2
            AUC_range = np.sum(width*height)
            auc_3d.append(AUC_range)
            
            width_PR = tpr[1:-1] - tpr[:-2]
            height_PR = (prec[1:] + prec[:-1])/2
            AP_range = np.sum(width_PR*height_PR)
            ap_3d.append(AP_range)

        
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d)/len(window_3d), sum(ap_3d)/len(window_3d)
        
        
class EventF1PA_metricor():
    def __init__(self, mode="log", base=3) -> None:
        """
        Using the Event-based point-adjustment F1 score to evaluate the models.
        
        Parameters:
            mode (str): Defines the scale at which the anomaly segment is processed. \n
                One of:\n
                    - 'squeeze': View an anomaly event lasting t timestamps as one timepoint.
                    - 'log': View an anomaly event lasting t timestamps as log(t) timepoint.
                    - 'sqrt': View an anomaly event lasting t timestamps as sqrt(t) timepoint.
                    - 'raw': View an anomaly event lasting t timestamps as t timepoint.
                If using 'log', you can specify the param "base" to return the logarithm of x to the given base, 
                calculated as log(x) / log(base).
            base (int): Default is 3.
        """
        super().__init__()
        
        self.eps = 1e-15
        self.name = "event-based f1 under pa with mode %s"%(mode)
        if mode == "squeeze":
            self.func = lambda x: 1
        elif mode == "log":
            self.func = lambda x: math.floor(math.log(x+base, base))
        elif mode == "sqrt":
            self.func = lambda x: math.floor(math.sqrt(x))
        elif mode == "raw":
            self.func = lambda x: x
        else:
            raise ValueError("please select correct mode.")
        
    def calc(self, scores, labels):
        '''
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;\n
            threshold: the value of threshold when getting best f1.
        '''
        
        search_set = []
        tot_anomaly = 0
        ano_flag = 0
        ll = len(labels)
        for i in range(labels.shape[0]):
            if labels[i] > 0.5 and ano_flag == 0:
                ano_flag = 1
                start = i
            
            # alleviation
            elif labels[i] <= 0.5 and ano_flag == 1:
                ano_flag = 0
                end = i
                tot_anomaly += self.func(end - start)
                
            # marked anomaly at the end of the list
            if ano_flag == 1 and i == ll - 1:
                ano_flag = 0
                end = i + 1
                tot_anomaly += self.func(end - start)

        flag = 0
        cur_anomaly_len = 0
        cur_max_anomaly_score = 0
        for i in range(labels.shape[0]):
            if labels[i] > 0.5:
                # record the highest score in an anomaly segment
                if flag == 1:
                    cur_anomaly_len += 1
                    cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                else:
                    flag = 1
                    cur_anomaly_len = 1
                    cur_max_anomaly_score = scores[i]
            else:
                # reconstruct the score using the highest score
                if flag == 1:
                    flag = 0
                    search_set.append((cur_max_anomaly_score, self.func(cur_anomaly_len), True))
                    search_set.append((scores[i], 1, False))
                else:
                    search_set.append((scores[i], 1, False))
        if flag == 1:
            search_set.append((cur_max_anomaly_score, self.func(cur_anomaly_len), True))
            
        search_set.sort(key=lambda x: x[0], reverse=True)
        best_f1 = 0
        threshold = 0
        P = 0
        TP = 0
        best_P = 0
        best_TP = 0
        for i in range(len(search_set)):
            P += search_set[i][1]
            if search_set[i][2]:  # for an anomaly point
                TP += search_set[i][1]
            precision = TP / (P + self.eps)
            recall = TP / (tot_anomaly + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            if f1 > best_f1:
                best_f1 = f1
                threshold = search_set[i][0]
                best_P = P
                best_TP = TP

        precision = best_TP / (best_P + self.eps)
        recall = best_TP / (tot_anomaly + self.eps)

        return float(precision), float(recall), float(best_f1), float(threshold)


class PointF1PA_metricor():
    """
    Using Point-based point-adjustment F1 score to evaluate the models.
    """
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-15
        self.name = "best f1 under pa"
        
    def calc(self, scores, labels):
        '''
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;\n
            threshold: the value of threshold when getting best f1.
        '''
        search_set = []
        tot_anomaly = 0
        for i in range(labels.shape[0]):
            tot_anomaly += (labels[i] > 0.5)
        flag = 0
        cur_anomaly_len = 0
        cur_max_anomaly_score = 0
        for i in range(labels.shape[0]):
            if labels[i] > 0.5:
                # record the highest score in an anomaly segment
                if flag == 1:
                    cur_anomaly_len += 1
                    cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                else:
                    flag = 1
                    cur_anomaly_len = 1
                    cur_max_anomaly_score = scores[i]
            else:
                # reconstruct the score using the highest score
                if flag == 1:
                    flag = 0
                    search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                    search_set.append((scores[i], 1, False))
                else:
                    search_set.append((scores[i], 1, False))
        if flag == 1:
            search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
            
        search_set.sort(key=lambda x: x[0], reverse=True)
        best_f1 = 0
        threshold = 0
        P = 0
        TP = 0
        best_P = 0
        best_TP = 0
        for i in range(len(search_set)):
            P += search_set[i][1]
            if search_set[i][2]:  # for an anomaly point
                TP += search_set[i][1]
            precision = TP / (P + self.eps)
            recall = TP / (tot_anomaly + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            if f1 > best_f1:
                best_f1 = f1
                threshold = search_set[i][0]
                best_P = P
                best_TP = TP

        precision = best_TP / (best_P + self.eps)
        recall = best_TP / (tot_anomaly + self.eps)

        return float(precision), float(recall), float(best_f1), float(threshold)
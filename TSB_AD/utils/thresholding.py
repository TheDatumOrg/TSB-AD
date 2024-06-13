import pandas as pd
from operator import itemgetter
import numpy as np
import datetime
import statistics


def Moving2T(MAerror, factor, hscaleCount=1000):
    """
    Based on the historic anomaly score calculate the standard deviation and mean of scores to derive in new threshold.

    :param MAerror: all anomaly scores until now.
    :param factor: the multiplier factor for standard deviation to calculate threshold.
    :param hscaleCount: the number of historical values to take in to consideration in threshold calculation.
    :return: Boolean (is anomaly) and the threshold value
    """
    historyerrors = MAerror[max(0, len(MAerror) - hscaleCount):]
    if len(historyerrors) == 1:
        return False,historyerrors[-1]
    th = statistics.mean(historyerrors) + factor * statistics.stdev(historyerrors)
    secondpass=[ d for d in historyerrors if d<th]
    if len(secondpass) == 0:
        return False, historyerrors[-1]
    fianal_threshold= statistics.mean(secondpass) + factor * statistics.stdev(secondpass)
    return MAerror[-1]>fianal_threshold, fianal_threshold

def Moving2Texclude(MAerror,anomalies, factor, hscaleCount=1000):
    """
    This method excludes preciously found anomalies before apply Moving2T technique.

    :param MAerror: all anomaly scores until now.
    :param anomalies:  a boolean array (with size equal or smaller than MAerror) which indicate if an instance is anomaly or no.
    :param factor: the multiplier factor for standard deviation to calculate threshold.
    :param hscaleCount: the number of historical values to take in to consideration in threshold calculation.
    :return: Boolean (is anomaly) and the threshold value
    """
    withoutAnomalies = [error for error, isanomaly in zip(MAerror[:len(anomalies)], anomalies) if isanomaly == False]
    withoutAnomalies.extend(MAerror[len(anomalies):])
    return Moving2T(withoutAnomalies, factor, hscaleCount=hscaleCount)






def dynamicThresholding(MAerror, DesentThreshold=0.02, hscaleCount=1000,alldata=False):
    """
    Re-Implementation of dynamic thresholding from : Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding

    This method is used to calculate the threshold only for the last sample of Anomaly scores, based on the calculated anomaly scores so far.

    Note: To calculate the threshold for each anomaly score this method needs to used iteratively.

    :param MAerror: Anomaly Scores
    :param DesentThreshold: Parameter to prune anomalies when their percentage difference from threshold is lower than this.
    :param hscaleCount: Historic anomaly scores samples to consider (History window)
    :param alldata: True in case we want to consider as historic samples, all anomaly scores so far.
    :return: False in case it couldn't produce a threshold, else True and the threshold value
    """
    normalization_in_error = False
    # start_time = time.time()

    historyerrors=MAerror[max(0,len(MAerror)-hscaleCount):]
    if alldata==True:
        historyerrors=MAerror
    error = historyerrors[-1]
    if len(historyerrors) == 1:
        return False,historyerrors[-1]

    # =======================================
    # ======= define parameters of threshold calculation ===================
    z = [v / 6 for v in range(12, 30)]  # z vector for threshold calculation


    diviation = statistics.stdev(historyerrors)  # diviation of errors
    meso = statistics.mean(historyerrors) # mean of errors
    e = [meso + (element * diviation) for element in z]  # e: set of candidate thresholds


    maximazation_value = []
    maxvalue = -1
    thfinal = e[0]
    maxEA = []
    # ============ threshold calculation ========================
    for th in e:
        EA = []  # List of sequence of anomalous errors
        ea = [(i,distt) for i,distt in enumerate(historyerrors) if distt>th] # dataframe of anomaly errors

        # if ea equals to 0 that means no anomalies so the Δμ/μ and Δσ/σ also are equal to zero
        if len(ea) == 0:
            continue
        if len([element for element in historyerrors if element < th]) <= 1:
            continue
        # Δμ -> difference betwen mean of errors and mean of errors without anomalies
        dmes = meso - statistics.mean([element for element in historyerrors if element < th])
        # Δσ ->  difference betwen diviation of errors and diviation of errors without anomalies
        ddiv = diviation - statistics.stdev([element for element in historyerrors if element < th])


        # ========= group anomaly error in sequences================
        # ea= [ (position, dist/error) , ... , (position, dist/error)]
        posi = ea[0][0]
        while posi <= ea[-1][0]:
            sub = []

            tempea=[tupls for tupls in ea if tupls[0]>=posi]
            sub.append(tempea[0])
            # store all continues errors (in index) in same subssequence
            for row in tempea[1:]:
                # if index of error is the last index of subsequence plus 1 then error is part of this sequence
                if row[0] == sub[-1][0] + 1:
                    sub.append(row)
                    posi = row[0] + 1
                else:
                    posi = row[0]
                    break
            # add the subsequence in to the list
            EA.append(sub)
            if len(tempea[1:]) == 0:
                break



        # ================ persentage impact of the threshold =================
        argmaxError = (dmes / meso + ddiv / diviation) / (len(ea) + len(EA) * len(EA))  # calculate value of formula which we try to maximize
        if maxvalue < argmaxError:
            maxvalue = argmaxError
            thfinal = th
            maxEA = EA
        maximazation_value.append(argmaxError)
    if len(maxEA) == 0:
        return False,thfinal

    if error > thfinal:
        # ==================look for prunning===========================
        # if last value belongs to anomalies then i will be a part of last anomaly sequence
        notea = [err for err in historyerrors if err<=thfinal]
        normalmax = max(notea)

        #maxEA = maxEA[:-1]
        lastSeq = maxEA[-1]
        maxlastSeq = max(lastSeq, key=itemgetter(1))
        maxErrorEA = [max(seq, key=itemgetter(1)) for seq in maxEA]
        maxErrorEA.append((-1, normalmax))
        minhistory = 0
        if normalization_in_error == True:
            minhistory = min(historyerrors)

        
        maxlastSeq = (maxlastSeq[0], maxlastSeq[1] - (minhistory - minhistory / 100.0))

        sortedmax = sorted(maxErrorEA, key=lambda x: x[1], reverse=True)

        checkpoint = -1
        count = -1
        for tup1, tup2 in zip(sortedmax[:-1], sortedmax[1:]):
            count += 1
            diff = (tup1[1] - tup2[1]) / tup1[1]
            if diff > DesentThreshold:
                checkpoint = count
        if checkpoint != -1:
            realAnomalies = sortedmax[:checkpoint + 1]
            if maxlastSeq[0] in list(map(list, zip(*realAnomalies)))[0]:
                return True,thfinal
    return False,thfinal


def DynamicThresholdingExclude(MAerror,anomalies, DesentThreshold=0.01, hscaleCount=1000):
    """
    This method excludes preciously found anomalies before apply Dynamic Thresholding technique technique.

    :param MAerror: all anomaly scores until now.
    :param anomalies:  a boolean array (with size equal or smaller than MAerror) which indicate if an instance is anomaly or no.
    :param DesentThreshold: The per sent difference to consider an anomaly (how much anomaly score has to overcome the threshold in persentage)
    :param hscaleCount: the number of historical values to take in to consideration in threshold calculation.
    :return: Boolean (is anomaly) and the threshold value
    """
    withoutAnomalies=[error for error,isanomaly in zip(MAerror[:len(anomalies)],anomalies) if isanomaly==False]
    withoutAnomalies.extend(MAerror[len(anomalies):])
    return dynamicThresholding(withoutAnomalies, DesentThreshold, hscaleCount=hscaleCount)



def selfTuning(factor,anomaly_scores_in_normal):
    """
    This method calculates the mean and standard deviation of Anomaly scores in Normal reference data
    and use them to calculate a threshold as mean+(factor*standard_deviation).
    
    :param factor: multiplier of standard deviation
    :param anomaly_scores_in_normal: Anomaly scores produced from normal data
    :return: threshold value
    """
    if len(anomaly_scores_in_normal)==0:
        if len(anomaly_scores)<sizeOfReference:
            return False,max(anomaly_scores) # not enough data to calculate threshold using the parameters.
        anomaly_scores_in_normal=anomaly_scores[:sizeOfReference]
    th = statistics.mean(anomaly_scores_in_normal) + factor * statistics.stdev(anomaly_scores_in_normal)
    return th



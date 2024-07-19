import numpy as np
from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .basic_metrics import basic_metricor, generate_curve, generate_curve_numba, EventF1PA_metricor, PointF1PA_metricor

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    grader = basic_metricor()
    AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    _, _, AUC_PR = grader.metric_PR(labels, score)
    Best_PointF1, Best_PointF1_Threshold = grader.metric_best_F(labels, score)

    _, _, Best_PointF1PA, _ = PointF1PA_metricor().calc(score, labels)
    _, _, Best_EventF1PA, _ = EventF1PA_metricor().calc(score, labels)
    Best_RF1, Best_RF1_Threshold = grader.metric_best_RF(labels, score)

    discrete_score = np.array(score > 0.5, dtype=np.float32)
    events_pred = convert_vector_to_events(discrete_score)
    events_gt = convert_vector_to_events(labels)
    Trange = (0, len(discrete_score))
    affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

    Affiliation_Precision = affiliation_metrics['Affiliation_Precision']
    Affiliation_Recall = affiliation_metrics['Affiliation_Recall']
    Affiliation_F = 2*Affiliation_Precision*Affiliation_Recall / (Affiliation_Precision+Affiliation_Recall+0.00001)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)
    # _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve_numba(labels, score, 2*slidingWindow)


    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = Best_PointF1
    metrics['PA-F1'] = Best_PointF1PA
    metrics['Event-based-F1'] = Best_EventF1PA
    metrics['R-based-F1'] = Best_RF1
    metrics['Affiliation-F'] = Affiliation_F
    return metrics
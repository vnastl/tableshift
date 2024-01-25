import torch
import pandas as pd

def binary_stat_scores_format(target,prediction,threshold= 0.5):
    """Convert all input to label format.

    - If prediction tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If prediction tensor is floating point, thresholds afterwards
    - Mask all datapoints that should be ignored with negative values

    """
    if isinstance(target, pd.Series):
        target = target.to_numpy()
    target = torch.from_numpy(target)
    prediction = torch.from_numpy(prediction)
    if prediction.is_floating_point():
        if not torch.all((prediction >= 0) * (prediction <= 1)):
            # prediction is logits, convert with sigmoid
            prediction = prediction.sigmoid()
        prediction = (prediction > threshold)*1.0

    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    return target, prediction

def balanced_accuracy_score(target,prediction):
    """Calculates balanced accuracy.

    Args:
        target (np.array): Vector containing the target values for each sample.
        prediction (np.array): Vector containing the predictions for each sample.
        inference (bool, optional): If set to True, calculate standard errors of balanced accuracy. Defaults to False.

    Returns:
        list | float64: If inference is set to True, return list of balanced accuracy and standard error of balanced accuracy.
            Else return balanced accuracy.
    """
    target, prediction = binary_stat_scores_format(target,prediction)
    n = len(prediction)
    tp = torch.count_nonzero((target == 1) & (prediction == 1))
    fn = torch.count_nonzero((target == 1) & (prediction == 0))
    tn = torch.count_nonzero((target == 0) & (prediction == 0))
    fp = torch.count_nonzero((target == 0) & (prediction == 1))
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    balanced_acc = (sensitivity + specificity)/2
    sensitivity_se = torch.sqrt(sensitivity*(1-sensitivity)/n)
    specificity_se = torch.sqrt(specificity*(1-specificity)/n)
    balanced_acc_se = torch.sqrt(sensitivity_se**2/4 + specificity_se**2/4)
    return balanced_acc.item(), balanced_acc_se.item()
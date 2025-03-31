import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from scipy.stats import entropy

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, average_precision_score, confusion_matrix
)


def compute_auroc(y_confidence, is_known):
    """
    Compute AUROC for distinguishing known (in-distribution) and unknown (out-of-distribution) samples.

    Args:
        y_confidence (array-like): Confidence scores (e.g., max softmax probabilities) for all samples.
        is_known (array-like): Boolean array indicating if each sample is known (in-distribution).

    Returns:
        float: AUROC score.
    """
    return roc_auc_score(is_known, y_confidence) * 100

def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    print(f'OSCR: {OSCR}')

    return OSCR

def compute_closed_set_accuracy(y_pred, y_true, is_known):
    """
    Compute accuracy on known samples.

    Args:
        y_pred (array-like): Predicted labels for all samples.
        y_true (array-like): True labels for all samples.
        is_known (array-like): Boolean array indicating known samples.

    Returns:
        float: Closed set accuracy.
    """
    return accuracy_score(y_true[is_known], y_pred[is_known]) * 100


def compute_macro_f1(y_true_all, y_pred_all):
    """
    Compute macro F1-score including the unknown class.

    Args:
        y_true_all (array-like): True labels for all samples (including unknown).
        y_pred_all (array-like): Predicted labels for all samples (including unknown).

    Returns:
        float: Macro-averaged F1-score.
    """
    labels = np.unique(np.concatenate([y_true_all, y_pred_all]))
    return f1_score(y_true_all, y_pred_all, average='macro', labels=labels) * 100


def compute_auin(y_confidence, is_known):
    """
    Compute AUIN (Area Under In-distribution PR curve).

    Args:
        y_confidence (array-like): Confidence scores for all samples.
        is_known (array-like): Boolean array indicating known samples.

    Returns:
        float: AUIN score.
    """
    return average_precision_score(is_known, y_confidence) * 100


def compute_auout(y_confidence, is_known):
    """
    Compute AUOUT (Area Under Out-of-distribution PR curve).

    Args:
        y_confidence (array-like): Confidence scores for all samples.
        is_known (array-like): Boolean array indicating known samples.

    Returns:
        float: AUOUT score.
    """
    return average_precision_score(~is_known, 1 - np.asarray(y_confidence)) * 100


def compute_dtacc(y_confidence, is_known, y_true, y_pred):
    """
    Compute maximum detection accuracy over all thresholds.

    Args:
        y_confidence (array-like): Confidence scores for all samples.
        is_known (array-like): Boolean array indicating known samples.
        y_true (array-like): True labels for all samples.
        y_pred (array-like): Predicted labels for all samples.

    Returns:
        float: Maximum detection accuracy.
    """
    y_confidence = np.asarray(y_confidence)
    is_known = np.asarray(is_known)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    known_mask = is_known
    y_conf_known = y_confidence[known_mask]
    y_conf_unknown = y_confidence[~known_mask]
    y_true_known = y_true[known_mask]
    y_pred_known = y_pred[known_mask]

    n_known = len(y_true_known)
    n_unknown = len(y_conf_unknown)
    n_total = n_known + n_unknown

    if n_total == 0:
        return 0.0

    # Precompute correctness for known samples
    correct_known = (y_pred_known == y_true_known)
    # Sort all confidence scores descending
    thresholds = np.sort(y_confidence)[::-1]

    max_acc = 0.0
    for t in thresholds:
        # Correct accepted known
        accepted = y_conf_known >= t
        correct_acc = np.sum(correct_known[accepted])

        # Correct rejected unknown
        rejected_unk = np.sum(y_conf_unknown < t)

        acc = (correct_acc + rejected_unk) / n_total
        if acc > max_acc:
            max_acc = acc

    return max_acc * 100


def compute_filtered_accuracy(y_true, y_pred, src_class, novel_class_label=-1):
    """
    Compute classification accuracy after filtering out:
    1. Samples not in src_class (ground truth unknown samples).
    2. Samples predicted as novel class (out-of-distribution).

    Args:
        y_true (array-like): True labels for all samples.
        y_pred (array-like): Predicted labels for all samples.
        src_class (list): List of known classes (e.g., [0, 1]).
        novel_class_label (int): Label used to indicate predicted novel class (default: -1).

    Returns:
        float: Filtered classification accuracy.
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out samples not in src_class (ground truth unknown samples)
    known_mask = np.isin(y_true, src_class)

    # Filter out samples predicted as novel class
    predicted_known_mask = (y_pred != novel_class_label)

    # Combine both masks
    final_mask = known_mask & predicted_known_mask

    # If no samples remain, return 0.0
    if np.sum(final_mask) == 0:
        return 0.0

    # Compute accuracy on the remaining samples
    filtered_accuracy = np.mean(y_true[final_mask] == y_pred[final_mask])
    return filtered_accuracy * 100


# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
def inception_score(data, model, splits=1, labels=None, batch_size=32):
    """Computes the inception score of the generated data
    """
    N = len(data)

    if torch.is_tensor(data):
        dataset = torch.utils.data.TensorDataset(data, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                      drop_last=False)
        iterat = iter(data_loader)
    else:
        iterat = data
    outputs_accum = []
    while True:
        try:
            data, labels = next(iterat)
        except:
            break
        features, outputs = model(data)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs_accum.append(outputs.detach().cpu().numpy())
    outputs_accum = np.concatenate(outputs_accum)

    split_scores = []
    for k in range(splits):
        part = outputs_accum[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

    # scores = []
    # for i in range(splits):
    #   part = outputs_accum[(i * outputs_accum.shape[0] // splits):((i + 1) * outputs_accum.shape[0] // splits), :]
    #   kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    #   kl = np.mean(np.sum(kl, 1))
    #   scores.append(np.exp(kl))
    #
    # return np.mean(scores), np.std(scores)


def calculate_activation_statistics(feas):
    """Calculation of the statistics used by the FID.
    Params:
    -- feas       : feature maps

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(feas, axis=0)
    sigma = np.cov(feas, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_wrapped(data, model, args, labels=None):
    if torch.is_tensor(data):
        dataset = torch.utils.data.TensorDataset(data, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                      drop_last=False)
        iterat = iter(data_loader)
    else:
        iterat = data
    feas_accum = []
    while True:
        try:
            data, labels = next(iterat)
        except:
            break
        features, _ = model(data)
        feas_accum.append(features.detach().cpu().numpy())
    feas_accum = np.concatenate(feas_accum)
    mu, sigma = calculate_activation_statistics(feas_accum)
    return mu, sigma


# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
def fid_score(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


import numpy as np
import torch
from scipy import linalg
from scipy.stats import entropy

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, average_precision_score, roc_curve
)


def compute_auroc(y_confidence, labels, novel_class_label=-1):

    is_ood = (labels == novel_class_label)
    return roc_auc_score(is_ood, -y_confidence) * 100


def compute_oscr(y_confidence, labels, pred, novel_class_label=-1):
    is_ood = (labels == novel_class_label)
    fpr, tpr, _ = roc_curve(is_ood, -y_confidence)
    oscr_auc = np.trapz(tpr, fpr)
    return oscr_auc * 100


def compute_closed_set_accuracy(y_pred, y_true, novel_class_label=-1):
    id_mask = (y_true != novel_class_label)
    return accuracy_score(y_true[id_mask], y_pred[id_mask]) * 100


def compute_macro_f1(y_true, y_pred, novel_class_label=-1):
    return f1_score(y_true, y_pred, average='macro') * 100


def compute_auin(y_confidence, labels, novel_class_label=-1):
    is_ood = (labels == novel_class_label)
    return average_precision_score(is_ood, -y_confidence) * 100


def compute_auout(y_confidence, labels, novel_class_label=-1):
    is_id = (labels != novel_class_label)
    return average_precision_score(is_id, y_confidence) * 100


def compute_dtacc(y_confidence, labels, novel_class_label=-1):
    is_ood = (labels == novel_class_label)
    thresholds = np.sort(np.unique(y_confidence))[::-1]

    max_acc = 0
    for t in thresholds:
        pred_ood = (y_confidence <= t)
        acc = np.mean(pred_ood == is_ood)
        if acc > max_acc:
            max_acc = acc
    return max_acc * 100


def compute_fpr95(y_confidence, labels, novel_class_label=-1):
    is_ood = (labels == novel_class_label)
    fpr, tpr, _ = roc_curve(is_ood, -y_confidence)

    # Find first threshold where TPR >= 95%
    tpr_95_idx = np.where(tpr >= 0.95)[0]
    if len(tpr_95_idx) == 0:
        return 1.0  # Worst case
    return fpr[tpr_95_idx[0]] * 100


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


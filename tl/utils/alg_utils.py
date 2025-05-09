# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : alg_utils.py
# Euclidean Alignment
# Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power


# numpy implementation, if error try EA_SPDsafe function
def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


# arithmetic mean only, SPD-safe
def EA_SPDsafe(x, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    n = len(x)
    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar = C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])

    eigvals, eigvecs = np.linalg.eigh(R_bar)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    ref = eigvecs @ D_inv_sqrt @ eigvecs.T

    XEA = ref @ x

    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)
    sample_num: int
        previous number of samples used to calculate R

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA



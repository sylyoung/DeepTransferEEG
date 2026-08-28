import numpy as np
from scipy.linalg import fractional_matrix_power


# Euclidean Alignment
# Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach
def EA(x, epsilon=1e-6):
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
    try:
        sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    except:
        to_add = np.eye(len(refEA)) * epsilon
        sqrtRefEA = fractional_matrix_power(refEA + to_add, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_offline(X, num_subjects):
    '''
    :param X: np array, EEG data (num_trials, num_channels, num_timesamples)
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    return X


def EA_online(X):
    '''
    :param X: np array, EEG data (num_trials, num_channels, num_timesamples)
    :return: np array, aligned EEG data
    '''
    # Online(Incremental) EA
    # Much proper way to do EA for target subject considering online BCIs
    Xt_aligned = []
    R = 0
    num_samples = 0
    for ind in range(len(X)):
        curr = X[ind]
        cov = np.cov(curr)
        # Note that the following line is an update of the mean covariance matrix (R), instead of a full recalculation. It is much faster computation in this way.
        # Note also that the covariance matrix calculation should take in all visible samples(trials) for this domain(subject)
        R = (R * num_samples + cov) / (num_samples + 1)
        num_samples += 1
        sqrtRefEA = fractional_matrix_power(R, -0.5)
        # transform the original trial. All latter algorithms only use the transformed data as input
        curr_aligned = np.dot(sqrtRefEA, curr)
        Xt_aligned.append(curr_aligned)
    Xt_aligned = np.array(Xt_aligned)
    # EA done

    return Xt_aligned

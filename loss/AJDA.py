import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import inv, det

# 损失函数定义
def rbf_metric(X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = rbf_kernel(X, Y, sigmas, wts)
    return _metric(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

def rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = torch.matmul(X, X.transpose(-1, -2))
    XY = torch.matmul(X, Y.transpose(-1, -2))
    YY = torch.matmul(Y, Y.transpose(-1, -2))

    X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
    Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

    r = lambda x: x.unsqueeze(0)
    c = lambda x: x.unsqueeze(1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms))).sigmoid()
        K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms))).sigmoid()
        K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms))).sigmoid()
    return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

def _metric(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    ns = float(K_XX.size(0))
    nt = float(K_YY.size(0))

    C_K_XX = torch.pow(K_XX, 2)
    C_K_YY = torch.pow(K_YY, 2)
    C_K_XY = torch.pow(K_XY, 2)

    if biased:
        metric = (torch.sum(C_K_XX) / (ns * ns) +
                  torch.sum(C_K_YY) / (nt * nt) -
                  2 * torch.sum(C_K_XY) / (ns * nt))
    else:
        if const_diagonal is not False:
            trace_X = ns * const_diagonal
            trace_Y = nt * const_diagonal
        else:
            trace_X = torch.trace(C_K_XX)
            trace_Y = torch.trace(C_K_YY)

        metric = ((torch.sum(C_K_XX) - trace_X) / ((ns - 1) * ns) +
                  (torch.sum(C_K_YY) - trace_Y) / ((nt - 1) * nt) -
                  2 * torch.sum(C_K_XY) / (ns * nt))
    return metric

def HoSDR(X1, X2, m=2, bandwidths=[5]):
    kernel_loss = rbf_metric(10**(2*m) * torch.pow(X1, 2*m), 
                             10**(2*m) * torch.pow(X2, 2*m), 
                             sigmas=bandwidths)
    return 100 * kernel_loss

def DOA(mu1, sigma1, mu2, sigma2):
    mu1, sigma1, mu2, sigma2 = mu1.detach().cpu().numpy(), sigma1.detach().cpu().numpy(), \
                               mu2.detach().cpu().numpy(), sigma2.detach().cpu().numpy()
    sigma = (sigma1 + sigma2) / 2.
    inv_sigma = inv(sigma)
    diff = mu1 - mu2
    mahalanobis = diff.T @ inv_sigma @ diff
    det_sigma = det(sigma)
    det_sigma1 = det(sigma1)
    det_sigma2 = det(sigma2)
    dof = np.exp(-0.125 * mahalanobis) * np.sqrt(det_sigma / np.sqrt(det_sigma1 * det_sigma2))
    return dof

def tf_cov(x):
    mean_x = torch.mean(x, dim=0)
    cov_xx = torch.matmul((x - mean_x).transpose(-1, -2), x - mean_x) / x.size(0)
    return mean_x, cov_xx

def classification_division(data, label):
    label = torch.argmax(label, dim=-1)
    
    # 使用布尔索引进行分类
    mask_0 = (label == 0)
    mask_1 = (label == 1)
    mask_2 = (label == 2)
    mask_3 = (label == 3)

    
    a = data[mask_0]
    b = data[mask_1]
    c = data[mask_2]
    d = data[mask_3]

    
    return a, b, c, d

def Class_loss(index, s, t_set, thre, mult):
    loss = 0
    doa_set = []
    if s.size(0) != 0 and t_set[index].size(0) != 0 and t_set[index].size(0) > (s.size(0) / mult):
        mu_s, sigma_s = tf_cov(s)
        mu_ind, sigma_ind = tf_cov(t_set[index])
        doa_ind = DOA(mu_ind, sigma_ind, mu_s, sigma_s)

        for ti in t_set:
            if s.size(0) != 0 and ti.size(0) != 0 and ti.size(0) > (s.size(0) / mult):
                mu_s, sigma_s = tf_cov(s)
                mu_ti, sigma_ti = tf_cov(ti)
                doa = DOA(mu_ti, sigma_ti, mu_s, sigma_s)
                doa_set.append(doa)

        if doa_ind > thre and doa_ind == max(doa_set):
            loss = doa_ind * HoSDR(s, t_set[index])
    return loss

def AJDA(data1, label1, data2, label2, thre, mult):
    s1, s2, s3, s4 = classification_division(data1, label1)
    t1, t2, t3, t4 = classification_division(data2, label2)

    MDA_loss = 10 * HoSDR(data1, data2)

    CDA_1 = Class_loss(0, s1, [t1, t2, t3, t4], thre, mult)
    CDA_2 = Class_loss(1, s2, [t1, t2, t3, t4], thre, mult)
    CDA_3 = Class_loss(2, s3, [t1, t2, t3, t4], thre, mult)
    CDA_4 = Class_loss(3, s4, [t1, t2, t3, t4], thre, mult)


    CDA_loss = CDA_1 + CDA_2 + CDA_3 + CDA_4
    AJDA_loss = MDA_loss + CDA_loss

    return MDA_loss, CDA_loss, AJDA_loss
import torch
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
from matplotlib.patches import Circle
import math
import cv2
from sklearn.preprocessing import normalize
from scipy.spatial import Delaunay
import ipdb
from torch.utils.data import Dataset, DataLoader


def batch_kron(a, b):
    """
    A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    return res.reshape(siz0 + siz1)


def kronecker(matrix1, matrix2):
    """
    - Kronecker Product

    Arguments:
    ----------
        - matrix1:  matrix1
        - matrix2:  matrix2

    Returns:
    --------
        - Kronecker product between matrix1 and matrix2

    """
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute(
        [0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


def batch_diagonal(mat):
    """
    - Batch-wise Diagonalization

    Arguments:
    ----------
        - mat: mat matrix (batch-wise) with entries that should be placed on the diagonals (Dimension: batch x N)

    Returns:
    --------
        - output: stack of diagonal matrices (Dimension: batch x N x N)

    """
    dims = [mat.size(i) for i in torch.arange(mat.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims).to(mat.device)

    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(mat.dim() - 1)]
    strides.append(output.size(-1) + 1)

    # stride and copy the imput to the diagonal
    output.as_strided(mat.size(), strides).copy_(mat)

    return output


def build_graph_structure(A):
    """
    BUILDS NODE-EDGE INCIDENCE MATRICES G AND H FROM GIVEN ADJACENCY MATRIX

    Arguments:
    ----------
        - A: node-to-node adjaceny matrix

    Returns:
    --------
        - G and H: node-edge incidence matrices such that: A = G*H^T

    """

    # Get number of nodes
    n = A.shape[0]

    #  Count number of ones in the adj. matrix to get number of edges
    nr_edges = torch.sum(A).to(torch.int32).item()

    #  Init G and H
    G = torch.zeros(n, nr_edges).to(A.device)
    H = torch.zeros(n, nr_edges).to(A.device)

    #  Get all non-zero entries and build G and H
    entries = (A != 0).nonzero()
    for count, (i, j) in enumerate(entries, start=0):
        G[i, count] = 1
        H[j, count] = 1

    return G, H


def build_graph_structure_QG(A):

    nn = A.shape[0]
    #  Count number of ones in the adj. matrix to get number of edges
    ne = torch.sum(A).to(torch.int32).item()//2
    #  Init G and H
    G = torch.zeros(nn, ne).to(A.device)
    H = torch.zeros(nn, ne).to(A.device)

    #  Get all non-zero entries and build G and H
    entries = torch.triu(A, 1).nonzero()
    for count, (i, j) in enumerate(entries, start=0):
        G[i, count] = 1
        H[j, count] = 1

    return G, H


def batch_build_graph_structure(A):
    """
    BUILDS NODE-EDGE INCIDENCE MATRICES G AND H FROM GIVEN ADJACENCY MATRIX IN BATCH

    Arguments:
    ----------
        - A: node-to-node adjaceny matrix stacked by batch

    Returns:
    --------
        - G and H: node-edge incidence matrices such that: A = G*H^T

    """

    # Get number of nodes
    nbatch= A.shape[0]
    n = A.shape[1]

    #  Count number of ones in the adj. matrix to get number of edges
    nr_edges = torch.sum(torch.sum(A, dim=1), dim=1).view(nbatch, -1).unsqueeze(-1)

    #  Init G and H
    G = torch.zeros(nbatch, n, nr_edges)
    H = torch.zeros(nbatch, n, nr_edges)

    #  Get all non-zero entries and build G and H
    entries = (A != 0).nonzero()
    for count, (i, j) in enumerate(entries, start=0):
        G[i, count] = 1
        H[j, count] = 1

    return G, H


def show_features(img1, P, img2=torch.Tensor(), P2=torch.Tensor(), pred_P=torch.Tensor(), gt_P=torch.Tensor(), radius=6):
    if torch.is_tensor(img1):
        img1 = img1.numpy()
    if torch.is_tensor(img2):
        img2 = img2.numpy()
    # norm = np.linalg.norm(img1)
    # img1 /= norm
    fig = plt.figure()
    color = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
             '#911eb4', '#46f0f0', '#f032e6',
             '#000000', '#000000', '#000000', '#000000', '#000000',
             '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
             '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#ffffff', '#000000']
    n_color = len(color)

    x = P[:, 0]
    y = P[:, 1]
    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Source Image')
    ax1.imshow(img1)
    # plt.axes(ax[0])
    # cv2.imshow('img1', img1)
    for i, (xx, yy) in enumerate(zip(x, y)):
        circle = Circle((xx, yy), radius, color=color[i % n_color])
        ax1.add_patch(circle)

    if P2.dim() == 2:
        x = P2[:, 0]
        y = P2[:, 1]
        ax2 = fig.add_subplot(222)
        ax2.title.set_text('Target Image')
        ax2.imshow(img2)
        # plt.axes(ax[1])
        # cv2.imshow('img1', img1)
        for i, (xx, yy) in enumerate(zip(x, y)):
            circle = Circle((xx, yy), radius, color=color[i % n_color])
            ax2.add_patch(circle)

    if pred_P.dim() == 2:
        x = pred_P[:, 0]
        y = pred_P[:, 1]
        ax3 = fig.add_subplot(223)
        ax3.title.set_text('Predicted Locations')
        ax3.imshow(img2)
        # plt.axes(ax[1])
        # cv2.imshow('img1', img1)
        for i, (xx, yy) in enumerate(zip(x, y)):
            circle = Circle((xx, yy), radius, color=color[i % n_color])
            ax3.add_patch(circle)

    if gt_P.dim() == 2:

        # norm = np.linalg.norm(img2)
        # img2 /= norm
        x = gt_P[:, 0]
        y = gt_P[:, 1]
        ax4 = fig.add_subplot(224)
        ax4.title.set_text('Ground Truth')
        ax4.imshow(img2)
        # plt.axes(ax[2])
        # cv2.imshow('img2', img2)
        for i, (xx, yy) in enumerate(zip(x, y)):
            circle = Circle((xx, yy), radius, color=color[i % n_color])
            ax4.add_patch(circle)

    plt.show()


def show_features_single(img1, P, radius=6):
    if torch.is_tensor(img1):
        img1 = img1.numpy()
    # norm = np.linalg.norm(img1)
    # img1 /= norm
    fig, ax = plt.subplots(1)
    color = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
             '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
             '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
             '#ffffff', '#000000']
    n_color = len(color)
    if P.dim() == 2:
        x = P[:, 0]
        y = P[:, 1]
        ax.imshow(img1)
        # plt.axes(ax)
        # cv2.imshow('img1', img1)
        for i, (xx, yy) in enumerate(zip(x, y)):
            circle = Circle((xx, yy), radius, color=color[i % n_color])
            ax.add_patch(circle)
    plt.show()


def batch_calc_pck(img, pred_P, gt_P, alpha=0.1):
    h, w = img.shape[-2], img.shape[-1]
    thres = alpha * math.sqrt(h*h + w*w)
    n = gt_P.shape[-2]
    dist = torch.norm(pred_P[:, :n, :] - gt_P, dim=-1)
    pck = torch.mean((dist <= thres).to(torch.float)) * 100.
    return pck.item()


def delaunay(P):
    # return adjacency matrix for the Delauney triangulation structure of P
    # P: 2d coordinates of points
    if torch.is_tensor(P):
        P = P.cpu().numpy()
    n_p, d = P.shape
    A = np.zeros((n_p, n_p))
    tri = Delaunay(P)
    n_tri = tri.simplices.shape[0]
    for itri in range(n_tri):
        A[tri.simplices[itri][0], tri.simplices[itri][1]] = 1
        A[tri.simplices[itri][1], tri.simplices[itri][0]] = 1
        A[tri.simplices[itri][1], tri.simplices[itri][2]] = 1
        A[tri.simplices[itri][2], tri.simplices[itri][1]] = 1
        A[tri.simplices[itri][0], tri.simplices[itri][2]] = 1
        A[tri.simplices[itri][2], tri.simplices[itri][0]] = 1
    return A


def delaunay_batch(P):
    # return adjacency matrix for the Delauney triangulation structure of P
    # P: 2d coordinates of points, in batch
    if torch.is_tensor(P):
        P = P.cpu().numpy()
    nbatch, n_p, d = P.shape
    A = np.zeros((nbatch, n_p, n_p))
    for ib in range(nbatch):
        tri = Delaunay(P[ib, :, :])
        n_tri = tri.simplices.shape[0]
        for itri in range(n_tri):
            A[ib, tri.simplices[itri][0], tri.simplices[itri][1]] = 1
            A[ib, tri.simplices[itri][1], tri.simplices[itri][0]] = 1
            A[ib, tri.simplices[itri][1], tri.simplices[itri][2]] = 1
            A[ib, tri.simplices[itri][2], tri.simplices[itri][1]] = 1
            A[ib, tri.simplices[itri][0], tri.simplices[itri][2]] = 1
            A[ib, tri.simplices[itri][2], tri.simplices[itri][0]] = 1
    return A


def hook_print(grad):
    if torch.sum(torch.isnan(grad)) > 0:
        print('nan-grad size: ', grad.shape)
        # print(grad)
        ipdb.set_trace()


def check_nan(var):
    if torch.sum(torch.isnan(var)) > 0:
        print('nan-var size: ', var.shape)
        # print(grad)
        ipdb.set_trace()


def check_data(dataset, nbatch=8, itr=5):
    for i in range(itr):
        data_loader = DataLoader(dataset, batch_size=nbatch)
        for batch_idx, (img1, img2, P1, gt_P1, img_path1, img_path2, P2) in enumerate(data_loader):
            check_nan(img1)
            check_nan(img2)
            check_nan(P1)
            check_nan(gt_P1)
            check_nan(P2)
            print('batch: %d(%d)' % (batch_idx, len(data_loader)))


def pad_edge(A, n):
    # A is the adjacency matrix
    # n is the desired number of edges
    # return a new adjacency matrix having n edges
    # this function assume that the number of edges inferred by A is less than n
    existing_edges = np.sum(A)/2
    add_edges = n - existing_edges
    npt = A.shape[0]
    added = 0
    while added < add_edges:
        r = np.random.randint(npt)
        c = np.random.randint(npt)
        if r != c and A[r, c] == 0:
            A[r, c] = 1
            A[c, r] = 1
            added += 1
    return A





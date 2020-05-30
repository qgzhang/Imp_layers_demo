import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import numpy as np
from torch.autograd import Function
from utils.utils import *
from scipy.optimize import lsq_linear, minimize, LinearConstraint, Bounds
import sys
sys.path.append('../ImpFuncToolbox/')
from ImpFunction import *
from time import time
import ipdb




def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def affinityMatrix_forward(F1, F2, U1, U2, A1, A2, lam1, lam2):
    """
    AFFINITY MATRIX LAYER

    Arguments:
    ----------
        - F1, F2: edge features of input image 1 and 2 (of complete batch)
        - U1, U2: node features of input image 1 and 2 (of complete batch)
        - G1, H1: node-edge incidence matrices of image 1
        - G2, H2: node-edge incidence matrices of image 2

    Returns:
    ----------
        - M: global affinity matrix (of complete batch)

    """
    #  (a) Get node start and end indices of edges

    G1, H1 = build_graph_structure_QG(A1)
    G2, H2 = build_graph_structure_QG(A2)

    idx1_start = G1.nonzero()[:, 0]
    idx2_start = G2.nonzero()[:, 0]
    idx1_end = H1.t().nonzero()[:, 1]
    idx2_end = H2.t().nonzero()[:, 1]

    #  (b) Build X and Y
    X = torch.cat((F1[:, idx1_start, :], F1[:, idx1_end, :]), -1)
    Y = torch.cat((F2[:, idx2_start, :], F2[:, idx2_end, :]), -1)

    # (c) Calculate M_e = X * \lambda * Y^T
    lam = F.relu(torch.cat((torch.cat((lam1, lam2), dim=1), torch.cat((lam2, lam1), dim=1))))
    lam.register_hook(hook_print)
    M_e = torch.bmm(torch.bmm(X, lam.expand(X.shape[0], -1, -1)), Y.permute(0, 2, 1))
    # M_e.register_hook(hook_print)
    #  (d) Calculate M_p = U1 * U2^T
    M_p = torch.bmm(U1, U2.permute(0, 2, 1))
    # M_p.register_hook(hook_print)
    #  (e) Calculate node-to-node and edge-to-edge similarity matrices
    # the reason for performing permute() on M_p and M_e is to make the vectorization column-major
    diagM_p = batch_diagonal(M_p.permute(0, 2, 1).reshape(M_p.shape[0], -1))
    diagM_e = batch_diagonal(M_e.permute(0, 2, 1).reshape(M_e.shape[0], -1))

    # (f) Calculate M = [vec(M_p)] + (G_2 \kronecker G_1)[vec(M_e)](H_2 \kronecker H_1)^T
    M = diagM_p + torch.bmm(torch.bmm(kronecker(G2, G1).expand(M_p.shape[0], -1, -1), diagM_e),
                            kronecker(H2, H1).expand(M_e.shape[0], -1, -1).permute(0, 2, 1))

    return M


def affinityMatrix_forward_FGM(F1, F2, U1, U2, A1, A2, lam1, lam2, lam3, lam4):
    # build M with FGM's formula

    G1, H1 = build_graph_structure_QG(A1)
    G2, H2 = build_graph_structure_QG(A2)

    idx1_start = G1.nonzero()[:, 0]
    idx2_start = G2.nonzero()[:, 0]
    idx1_end = H1.t().nonzero()[:, 1]
    idx2_end = H2.t().nonzero()[:, 1]

    #  (b) Build X and Y
    X = torch.cat((F1[:, idx1_start, :], F1[:, idx1_end, :]), -1)
    Y = torch.cat((F2[:, idx2_start, :], F2[:, idx2_end, :]), -1)

    # (c) Calculate M_e = X * \lambda * Y^T
    lam_F = F.relu(torch.cat((torch.cat((lam1, lam2), dim=1), torch.cat((lam2, lam1), dim=1))))
    lam_U = F.relu(torch.cat((torch.cat((lam3, lam4), dim=1), torch.cat((lam4, lam3), dim=1))))
    M_e = torch.bmm(torch.bmm(X, lam_F.expand(X.shape[0], -1, -1)), Y.permute(0, 2, 1))

    #  (d) Calculate M_p = U1 * U2^T
    # M_p = torch.bmm(U1, U2.permute(0, 2, 1))
    M_p = torch.bmm(torch.bmm(U1, lam_U.expand(U1.shape[0], -1, -1)), U2.permute(0, 2, 1))

    G1_FGM = G1 + H1
    G2_FGM = G2 + H2
    H1_FGM = torch.cat((G1_FGM, torch.eye(G1.shape[-2], dtype=torch.float).to(G1.device)), dim=-1)
    H2_FGM = torch.cat((G2_FGM, torch.eye(G2.shape[-2], dtype=torch.float).to(G2.device)), dim=-1)
    L11 = M_e  # Kq
    L12 = - torch.bmm(M_e, G2_FGM.expand(M_e.shape[0], -1, -1).permute(0, 2, 1))
    L21 = - torch.bmm(G1_FGM.expand(M_e.shape[0], -1, -1), M_e)
    L22 = torch.bmm(G1_FGM.expand(M_e.shape[0], -1, -1),
                    torch.bmm(M_e, G2_FGM.expand(M_e.shape[0], -1, -1).permute(0, 2, 1))) + M_p
    L = torch.cat((torch.cat((L11, L12), dim=-1), torch.cat((L21, L22), dim=-1)), dim=-2)

    H2_Kron_H1 = kronecker(H2_FGM, H1_FGM).expand(M_e.shape[0], -1, -1)
    diag_L = batch_diagonal(L.permute(0, 2, 1).reshape(M_e.shape[0], -1))
    M = torch.bmm(H2_Kron_H1, torch.bmm(diag_L, H2_Kron_H1.permute(0, 2, 1)))

    return M


def affinityMatrix_forward_FGM_2(F1, F2, U1, U2, A1, A2):
    # build M with FGM's formula
    # build Me with formula:  Me_ijab = abs(||Fi - Fj|| - ||Fa - Fb||)
    G1, H1 = build_graph_structure_QG(A1)
    G2, H2 = build_graph_structure_QG(A2)

    idx1_start = G1.nonzero()[:, 0]
    idx2_start = G2.nonzero()[:, 0]
    idx1_end = H1.t().nonzero()[:, 1]
    idx2_end = H2.t().nonzero()[:, 1]

    Fij = torch.sum(F1[:, idx1_start, :] * F1[:, idx1_end, :], dim=-1, keepdim=True)  # [nbatch, nedge1, 1]
    Fab = torch.sum(F2[:, idx2_start, :] * F2[:, idx2_end, :], dim=-1, keepdim=True)  # [nbatch, nedge2, 1]
    M_e = torch.abs(Fij - Fab.permute(0, 2, 1))  # [nbatch, nedge1, nedge2]

    #  (d) Calculate M_p = U1 * U2^T
    M_p = torch.bmm(U1, U2.permute(0, 2, 1))

    G1_FGM = G1 + H1
    G2_FGM = G2 + H2
    H1_FGM = torch.cat((G1_FGM, torch.eye(G1.shape[-2], dtype=torch.float).to(G1.device)), dim=-1)
    H2_FGM = torch.cat((G2_FGM, torch.eye(G2.shape[-2], dtype=torch.float).to(G2.device)), dim=-1)
    L11 = M_e  # Kq
    L12 = - torch.bmm(M_e, G2_FGM.expand(M_e.shape[0], -1, -1).permute(0, 2, 1))
    L21 = - torch.bmm(G1_FGM.expand(M_e.shape[0], -1, -1), M_e)
    L22 = torch.bmm(G1_FGM.expand(M_e.shape[0], -1, -1),
                    torch.bmm(M_e, G2_FGM.expand(M_e.shape[0], -1, -1).permute(0, 2, 1))) + M_p
    L = torch.cat((torch.cat((L11, L12), dim=-1), torch.cat((L21, L22), dim=-1)), dim=-2)

    H2_Kron_H1 = kronecker(H2_FGM, H1_FGM).expand(M_e.shape[0], -1, -1)
    diag_L = batch_diagonal(L.permute(0, 2, 1).reshape(M_e.shape[0], -1))
    M = torch.bmm(H2_Kron_H1, torch.bmm(diag_L, H2_Kron_H1.permute(0, 2, 1)))

    return M


def powerIteration_forward(M, N=50):
    """
    POWER ITERATION LAYER

    Arguments:
    ----------
        - M: affinity matrix (of complete batch)

    Returns:
    --------
        - v*: optimal assignment vector (of every sample in the batch)
    """

    # Init starting assignment-vector
    v = torch.rand(M.shape[0], M.shape[2], 1).to(M.device)

    # Perform N power iterations:
    # v_k+1 = M*v_k / (||M*v_k||_2)
    I = torch.eye(M.shape[1]).to(M.device) * 1e-10
    M = M + I.expand(M.shape[0], -1, -1)
    for i in range(N):
        v = F.normalize(torch.bmm(M, v), dim=1)

    return v


def biStochastic_forward(v, n, m, N=5):
    # Reshape the assignment vector to matrix form
    # S = v.view(v.shape[0], n, m)
    S = v.view(v.shape[0], m, n).permute(0, 2, 1)  # now it is reshaped as column major

    #  Perform N iterations: S_k+1 = ...., S_k+2 = ...
    for i in range(N):
        # S = torch.bmm(S, torch.bmm(torch.ones(v.shape[0], 1, n), S).inverse())
        # S = torch.bmm(torch.bmm(S, torch.ones(v.shape[0], m, 1)).inverse(), S)
        S = torch.div(S, torch.bmm(torch.ones(v.shape[0], 1, n).to(v.device), S))
        S = torch.div(S, torch.bmm(S, torch.ones(v.shape[0], m, 1).to(v.device)))

    return S


def voting_flow_forward(v, alpha=1., th=10):
    """
    VOTING LAYER BASED ON ASSIGNMENT VECTOR

    Arguments:
    ----------
        v: optimal assignment vector (of complete batch)
        alpha: scale value in softmax
        th: threshold value

    Returns:
    --------
        d: displacement vector

    """

    n = int(np.sqrt(v.shape[1]))
    n_ = int(np.sqrt(n))

    #  Calculate coordinate arrays
    i_coords, j_coords = np.meshgrid(range(n_), range(n_), indexing='ij')
    [P_y, P_x] = torch.from_numpy(np.array([i_coords, j_coords]))
    P_x = P_x.view(1, n, -1).expand(v.shape[0], -1, -1).to(torch.float32)
    P_y = P_y.view(1, n, -1).expand(v.shape[0], -1, -1).to(torch.float32)

    #  Perform displacement calculation
    S = alpha * v.view(v.shape[0], n, -1)
    S_ = F.softmax(S, dim=-1)
    P_x_ = torch.bmm(S_, P_x)
    P_y_ = torch.bmm(S_, P_y)
    d_x = P_x_ - P_x
    d_y = P_y_ - P_y
    d = torch.cat((d_x, d_y), dim=2)

    return d


def voting_forward(S, P, alpha=200., th=10):
    """
    VOTING LAYER BASED ON BISTOCHASTIC MATRIX

    Arguments:
    ----------
        S - confidence map obtained from bi-stochastic layer (of complete batch)
        P - Position matrix (batch x m x 2)
        alpha - scaling factor
        th - number of pixels to be set as threshold beyond which confidence levels are set to zero.
    Returns:
    --------
        - d: displacement vector

    """
    d = torch.bmm(F.softmax(alpha * S, dim=-1), P.to(torch.float))
    return d


class VGG_backbone(torch.nn.Module):
    """
    VGG_BACKBONE
    """
    def __init__(self, args):
        super(VGG_backbone, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.score_U = nn.Conv2d(512, 64, 1)
        self.score_F = nn.Conv2d(512, 64, 1)

        # upsampling
        self.deconv1_u = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2_u = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3_u = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4_u = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1_u = nn.BatchNorm2d(64)
        # self.bn2_u = nn.BatchNorm2d(64)
        # self.bn3_u = nn.BatchNorm2d(64)
        # self.bn4_u = nn.BatchNorm2d(64)
        self.deconv1_f = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2_f = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3_f = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4_f = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1_f = nn.BatchNorm2d(64)
        # self.bn2_f = nn.BatchNorm2d(64)
        # self.bn3_f = nn.BatchNorm2d(64)
        # self.bn4_f = nn.BatchNorm2d(64)
        self.relu_up = nn.ReLU(inplace=True)
        # self.upscore32 = nn.ConvTranspose2d(64, 64, 64, stride=32, bias=False)
        # self.upscore16 = nn.ConvTranspose2d(64, 64, 32, stride=16, bias=False)


        # Trainable Lambda Parameters from Affinity Matrix Layer
        eps = 1e-8
        self.lam1 = nn.Parameter(torch.rand(64, 64) * 0.1 + eps)
        self.lam2 = nn.Parameter(torch.rand(64, 64) * 0.1 + eps)
        self.lam3 = nn.Parameter(torch.rand(32, 32) * 0.1 + eps)
        self.lam4 = nn.Parameter(torch.rand(32, 32) * 0.1 + eps)

        self._initialize_weights()
        self.copy_params_from_vgg16()
        self.args = args

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self):
        """
            RETRIEVES VGG PARAMETERS (pretrained or not pretrained network)
        """
        vgg16 = models.vgg16(pretrained = True)   # Enabling/Disabling Pretrained-Version of VGG
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1
                ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)

    def _vgg_forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        feat1 = self.relu4_2(self.conv4_2(h))
        f = self.score_F(feat1)
        f = self.relu_up(self.deconv2_f(f))
        f = self.relu_up(self.deconv3_f(f))
        f = self.relu_up(self.deconv4_f(f))

        h = self.relu4_3(self.conv4_3(feat1))
        h = self.pool4(h)
        h = self.relu5_1(self.conv5_1(h))
        u = self.score_U(h)
        u = self.relu_up(self.deconv1_u(u))
        u = self.relu_up(self.deconv2_u(u))
        u = self.relu_up(self.deconv3_u(u))
        u = self.relu_up(self.deconv4_u(u))

        return f, u

    def get_normalised_fu(self, img1, P1, img2, P2):

        F1, U1 = self._vgg_forward(img1)  # F1,U1 : [nbatch, d_fea=64, H, W]
        F2, U2 = self._vgg_forward(img2)

        nb = P1.shape[0]
        np1 = P1.shape[1]
        np2 = P2.shape[1]
        d_fea = F1.shape[1]
        newP1 = torch.cat(
            (torch.tensor(range(nb)).view(nb, -1).repeat_interleave(np1, dim=0).to(self.args.device), P1.view(-1, 2)), dim=1)
        newP2 = torch.cat(
            (torch.tensor(range(nb)).view(nb, -1).repeat_interleave(np2, dim=0).to(self.args.device), P2.view(-1, 2)), dim=1)

        # [nbatch, np<=15, d_fea=64]
        F1 = F1.permute(0, 2, 3, 1)[newP1[:, 0], newP1[:, 1], newP1[:, 2], :].view(nb, -1, d_fea)
        U1 = U1.permute(0, 2, 3, 1)[newP1[:, 0], newP1[:, 1], newP1[:, 2], :].view(nb, -1, d_fea)
        F2 = F2.permute(0, 2, 3, 1)[newP2[:, 0], newP2[:, 1], newP2[:, 2], :].view(nb, -1, d_fea)
        U2 = U2.permute(0, 2, 3, 1)[newP2[:, 0], newP2[:, 1], newP2[:, 2], :].view(nb, -1, d_fea)

        # F1 = F1[:, :, P1[:, :, 1].squeeze(), P1[:, :, 0].squeeze()].permute(0, 2, 1)  # [nbatch, np<=15, d_fea=64]
        # U1 = U1[:, :, P1[:, :, 1].squeeze(), P1[:, :, 0].squeeze()].permute(0, 2, 1)
        # F2 = F2[:, :, P2[:, :, 1].squeeze(), P2[:, :, 0].squeeze()].permute(0, 2, 1)  # [nbatch, np=grid_size, d_fea=64]
        # U2 = U2[:, :, P2[:, :, 1].squeeze(), P2[:, :, 0].squeeze()].permute(0, 2, 1)
        F1, U1, F2, U2 = F.normalize(F1, dim=-1), F.normalize(U1, dim=-1), F.normalize(F2, dim=-1), F.normalize(U2, dim=-1)

        return F1, U1, F2, U2


class GMN(VGG_backbone):
    """
    GRAPH MATCHING NETWORK (GMN)
    a non official implementation of the Graph Matching Network (GMN) from
    <Deep Learning of Graph Matching - CVPR18>
    """
    def __init__(self, args):
        super(GMN, self).__init__(args)

    def forward(self, img1, P1, img2, P2):
        """
        FORWARD PASS - IMPLEMENTATION
        Arguments:
        ----------
            - im_1: input images (batch)
            - im_2: input images (batch)
            - P1: 2d coordinates of parts(points) in image 1
            - P2: 2d coordinates of parts(points) in image 2
        Returns:
        --------
            - d: displacement vector of complete batch
        """
        # Get node features
        F1, U1, F2, U2 = self.get_normalised_fu(img1, P1, img2, P2)
        np1 = P1.shape[1]
        np2 = P2.shape[1]
        #######################################
        # Build Graph Structure based on given adjacency matrix.
        # Adjacency matrix could be arbitrary,
        # for example, fully connected, but the fully connected structure exceeds memory limit;
        # so we can choose the Delaunay triangulation structure

        # Delauney connection
        A1 = torch.from_numpy(delaunay(P1[0])).to(F1.device)
        # A2 = torch.from_numpy(delaunay(P2[0])).to(F2.device)

        # Fully Connected
        # A1 = (torch.ones((np1, np1)) - torch.eye(np1)).to(F1.device)
        A2 = (torch.ones((np2, np2)) - torch.eye(np2)).to(F2.device)

        # Compute Forward pass using building blocks from paper
        # M = affinityMatrix_forward(F1, F2, U1, U2, A1, A2, self.lam1, self.lam2)
        M = affinityMatrix_forward_FGM_2(F1, F2, U1, U2, A1, A2)
        M = F.relu(M)

        v = powerIteration_forward(M, N=20)

        S = biStochastic_forward(v, np1, np2, N=10)

        d = voting_forward(S, P2, alpha=200)

        # if F1.requires_grad:
        #     check_nan(F1)
        #     check_nan(U1)
        #     check_nan(F2)
        #     check_nan(U2)
        #     check_nan(M )
        #     check_nan(v )
        #     check_nan(S )
        #     check_nan(d )
        #     F1.register_hook(hook_print)
        #     U1.register_hook(hook_print)
        #     F2.register_hook(hook_print)
        #     U2.register_hook(hook_print)
        #     M.register_hook(hook_print)
        #     v.register_hook(hook_print)
        #     S.register_hook(hook_print)
        #     d.register_hook(hook_print)
        #     F1.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     U1.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     F2.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     U2.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     M.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     v.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     S.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     d.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     torch.save((M, v, S, d), 'vars.save')

        return d


class ImpGMN(VGG_backbone):

    def __init__(self, args):
        super(ImpGMN, self).__init__(args)

    def forward(self, img1, P1, img2, P2):
        F1, U1, F2, U2 = self.get_normalised_fu(img1, P1, img2, P2)
        np1 = P1.shape[1]
        np2 = P2.shape[1]

        # Delauney connection
        A1 = torch.from_numpy(delaunay(P1[0])).to(F1.device)
        # A2 = torch.from_numpy(delaunay(P2[0])).to(F2.device)

        # Fully Connected
        # A1 = (torch.ones((np1, np1)) - torch.eye(np1)).to(F1.device)
        A2 = (torch.ones((np2, np2)) - torch.eye(np2)).to(F2.device)

        # Compute Forward pass using building blocks from paper
        # M = affinityMatrix_forward(F1, F2, U1, U2, A1, A2, self.lam1, self.lam2)
        M = affinityMatrix_forward_FGM_2(F1, F2, U1, U2, A1, A2)
        M = F.relu(M)

        # 0: SM
        # 1: SMAC - Rayleigh Quotient
        # 2: SMAC - bi-directional eigen decomposition
        # 3: SMAC - forward eigen decompostion and backward KKT
        if self.args.SMAC == 0:
            v = Imp_SM_F.apply(M)
            v = v + 0.001  #######################################################################
            S = biStochastic_forward(v, np1, np2, N=1)
        else:
            if self.args.SMAC == 1:
                v = Imp_SMAC_F_RQ.apply(M, np1, np2)
            elif self.args.SMAC == 2:
                v = Imp_SMAC_F_biEigen.apply(M, np1, np2)
            elif self.args.SMAC == 3:
                v = Imp_SMAC_F.apply(M, np1, np2)
                # evals, evecs = torch.symeig(M, eigenvectors=True)
                # v = evecs[:, :, -1]
            v = v + 0.001  #######################################################################
            S = biStochastic_forward(v, np1, np2, N=1)


        d = voting_forward(S, P2, alpha=200)

        # if F1.requires_grad:
        #     check_nan(F1)
        #     check_nan(U1)
        #     check_nan(F2)
        #     check_nan(U2)
        #     check_nan(M )
        #     check_nan(v )
        #     check_nan(S )
        #     check_nan(d )
        #     F1.register_hook(hook_print)
        #     U1.register_hook(hook_print)
        #     F2.register_hook(hook_print)
        #     U2.register_hook(hook_print)
        #     M.register_hook(hook_print)
        #     v.register_hook(hook_print)
        #     S.register_hook(hook_print)
        #     d.register_hook(hook_print)
        #     F1.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     U1.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     F2.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     U2.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     M.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     v.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     S.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     d.register_hook(lambda grad: torch.clamp(grad, -self.args.clip, self.args.clip))
        #     torch.save((M, v, S, d), 'vars.save')

        return d


class Imp_SM_F(Function):
    """
    The implicit layer implementation of the Spectral Matching (SM) algorithm
    """
    @staticmethod
    def forward(ctx, *args):
        M = args[0]
        evals, evecs = torch.symeig(M, eigenvectors=True)
        evec = evecs[:, :, -1]
        eval = evals[:, -1]
        ctx.save_for_backward(M, eval, evec)
        return evec

    @staticmethod
    def backward(ctx, *grad_outputs):
        M, eval, evec = ctx.saved_tensors
        M0 = M.detach().cpu().numpy()
        y0 = evec.detach().cpu().numpy()
        l0 = -eval.detach().cpu().numpy()

        nbatch = M0.shape[0]
        ny = y0.shape[-1]
        nM = ny * ny
        nl = 1  # for the dual variable lambda
        nf = ny + nl

        J_F_y = torch.zeros((nbatch, nf, ny))
        J_F_M = torch.zeros((nbatch, nf, nM))
        J_F_l = torch.zeros((nbatch, nf, nl))

        y = torch.from_numpy(y0).unsqueeze(-1)
        M = torch.from_numpy(M0)
        l = torch.from_numpy(l0).unsqueeze(-1)

        # print('y.shape = ', y.shape)
        # print('M.shape = ', M.shape)
        # print('l.shape = ', l.shape)

        y.requires_grad = True
        M.requires_grad = True
        l.requires_grad = True

        def stationarity(j):
            I_row = torch.zeros(ny)
            I_row[j] = 1
            return torch.matmul(M[:, j, :].unsqueeze(1) + I_row.expand(nbatch, -1).unsqueeze(1) * l.unsqueeze(-1),
                                y[:, :, :]).squeeze(1)

        def primal_fea(j):
            return torch.matmul(y.view(nbatch, -1), y+1e-10)  # + 0 * M + 0 * l

        functions = {stationarity: (0, ny),
                     primal_fea: (ny, ny + nl)}

        for func in functions.keys():
            for j in range(functions[func][0], functions[func][1]):
                with torch.set_grad_enabled(True):
                    aux = func(j - functions[func][0])
                    aux.backward(torch.ones_like(aux))
                J_F_y[:, j, :] = y.grad.view(nbatch, -1)
                J_F_M[:, j, :] = M.grad.view(nbatch, -1)
                J_F_l[:, j, :] = l.grad.view(nbatch, -1)
                y.grad.zero_()
                M.grad.zero_()
                l.grad.zero_()

        J_F_var = torch.cat((J_F_y, J_F_l), dim=-1)
        J_var_M = - torch.matmul(torch.inverse(J_F_var), J_F_M)
        J_y_M = J_var_M[:, :ny, :]
        dL_dy = grad_outputs[0].detach().cpu()
        dL_dM = torch.matmul(torch.transpose(J_y_M, 1, 2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
        dL_dM = dL_dM.view(nbatch, -1, ny)

        grads = dL_dM.to(grad_outputs[0].device)
        return grads


class Imp_SMAC_F(Function):
    """
    The implicit layer implementation of the Spectral Matching with Affine Constraints (SMAC) algorithm
    Forward: Eigen decomposition
    Backward: KKT
    """
    @staticmethod
    def forward(ctx, *args):
        M, np1, np2 = args[0], args[1], args[2]
        nbatch = M.shape[0]

        C = torch.zeros([np1+np2, np1*np2]).to(M.device)
        for i in range(np1):
            C[i, i*np2:(i+1)*np2] = 1
        for i in range(np1, np1+np2):
            C[i, range(i-np1, np1*np2, np2)] = 1

        b = torch.ones([np1+np2, 1]).to(M.device)

        k = np1 + np2
        Ck = C[-1, :].view(-1, C.shape[-1])
        bk = b[-1, :]
        Ceq = torch.mm(torch.cat((torch.eye(k-1), torch.zeros([k-1, 1])), dim=-1).to(M.device), C - torch.mm(b/bk, Ck))
        Inn = torch.eye(np1*np2).to(M.device)
        PC = Inn - torch.mm(torch.mm(Ceq.t(), torch.inverse(torch.mm(Ceq, Ceq.t()))), Ceq)

        M2 = torch.bmm(torch.bmm(PC.expand(nbatch, -1, -1), M), PC.expand(nbatch, -1, -1))


        evals, evecs = torch.symeig(M2, eigenvectors=True)
        evec = evecs[:, :, -1]
        eval = evals[:, -1]
        ctx.save_for_backward(M, eval, evec, C, b, torch.tensor(np1), torch.tensor(np2))
        return evec


    @staticmethod
    def backward(ctx, *grad_outputs):
        M, eval, evec, C, b, np1, np2 = ctx.saved_tensors
        M0 = M.detach().cpu().numpy()
        y0 = evec.detach().cpu().numpy()
        C0 = C.cpu().numpy()
        np1, np2 = int(np1), int(np2)
        nbatch = M0.shape[0]
        # print(nbatch, np1, np2)
        l0 = np.zeros([nbatch, np1 + np2])

        ny = y0.shape[-1]
        nM = ny * ny
        nl = l0.shape[-1]  # for the dual variable lambda
        nf = ny + nl
        # print('eval = ', eval)
        # print('ny = ', ny)
        # solve for lambda by least square: min || C.t() * lambda - bb ||_2
        I0 = np.eye(ny)
        for ib in range(nbatch):
            yi = y0[ib, :]
            Mi = M0[ib, :, :]
            yty = yi.dot(yi)
            ytMy = yi.dot(Mi).dot(yi)
            # print('yi  = ', yi  )
            # print('Mi  = ', Mi  )
            # print('yty = ', yty )
            # print('ytMy= ', ytMy)
            bb = - 2*((Mi * yty - ytMy * I0) / yty**2).dot(yi)
            l0[ib, :] = lsq_linear(C0.transpose(), bb).x
            # print('C0 = ', C0)
            # print('bb = ', bb)
            # print('x = ', lsq_linear(C0.transpose(), bb).x)

        # print('l0 = ', l0)
        # return None
        J_F_y = torch.zeros((nbatch, nf, ny))
        J_F_M = torch.zeros((nbatch, nf, nM))
        J_F_l = torch.zeros((nbatch, nf, nl))

        y = torch.from_numpy(y0).unsqueeze(-1)
        M = torch.from_numpy(M0)
        l = torch.from_numpy(l0).unsqueeze(-1).to(torch.float)
        C = torch.from_numpy(C0)

        # print('y.shape = ', y.shape)
        # print('M.shape = ', M.shape)
        # print('l.shape = ', l.shape)

        y.requires_grad = True
        M.requires_grad = True
        l.requires_grad = True

        def stationarity(j):
            yty = torch.matmul(y.permute(0,2,1), y+1e-10).squeeze(-1)
            ytMy = torch.matmul(torch.matmul(y.permute(0,2,1), M), y+1e-10).squeeze()
            # print('yty = ', yty.shape)
            # print('ytMy = ', ytMy.shape)
            # print('M[:, j, :] = ', M[:, j, :].shape)
            # print('yty * M[:, j, :] = ', (yty * M[:, j, :]).shape)
            # print('batch_diagonal(ytMy)', batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :].shape)
            nor = (yty * M[:, j, :] - batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :])
            # print('nor = ', nor)
            left = 2 * torch.matmul(nor.unsqueeze(1), y[:, :, :]).squeeze(1) / yty*yty
            # print('left = ', left)
            # torch.set_printoptions(threshold=5000)
            # print('C=', C)
            right = torch.matmul(C.permute(1, 0)[j, :].expand(nbatch, -1).unsqueeze(1), l[:, :, :]).squeeze()
            # print('right = ', right)
            return left + right

        def primal_fea(j):
            return torch.bmm(C[j, :].expand(nbatch, -1).unsqueeze(1), y[:,:,:]).squeeze() - 1  # b === 1

        functions = {stationarity: (0, ny),
                     primal_fea: (ny, ny + nl)}

        for func in functions.keys():
            for j in range(functions[func][0], functions[func][1]):
                with torch.set_grad_enabled(True):
                    aux = func(j - functions[func][0])
                    aux.backward(torch.ones_like(aux))
                J_F_y[:, j, :] = y.grad.view(nbatch, -1)
                J_F_M[:, j, :] = M.grad.view(nbatch, -1)
                J_F_l[:, j, :] = l.grad.view(nbatch, -1)
                y.grad.zero_()
                M.grad.zero_()
                l.grad.zero_()

        J_F_var = torch.cat((J_F_y, J_F_l), dim=-1)
        # print(J_F_var)
        J_var_M = - torch.matmul(torch.inverse(J_F_var), J_F_M)
        J_y_M = J_var_M[:, :ny, :]
        dL_dy = grad_outputs[0].detach().cpu()
        dL_dM = torch.matmul(torch.transpose(J_y_M, 1, 2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
        dL_dM = dL_dM.view(nbatch, -1, ny)

        grads = (dL_dM.to(grad_outputs[0].device), None, None)
        return grads


class Imp_SMAC_F_RQ(ImpFunction):
    """
    The implicit layer implementation of the Spectral Matching with Affine Constraints (SMAC) algorithm
    Solve:  the Rayleigh Quotient formulation
    Forward: scipy minimize
    Backward: KKT
    """

    @staticmethod
    def forward(ctx, *args):

        ############################################################
        #  The forward solver
        ############################################################
        def f_rayleign_quotient(M):
            def f_rq(x):
                return x.dot(-M).dot(x) / x.dot(x)  # it is -M because we are maximizing the Rayleigh Quotient
            return f_rq

        M, np1, np2 = args[0], args[1], args[2]
        nbatch = M.shape[0]

        C = np.zeros([np1+np2, np1*np2])
        for i in range(np1):
            C[i, i*np2:(i+1)*np2] = 1
        for i in range(np1, np1+np2):
            C[i, range(i-np1, np1*np2, np2)] = 1

        b = np.ones(np1+np2)
        linear_constraint = LinearConstraint(C, b, b)
        bounds = Bounds(np.zeros(np1*np2), np.ones(np1*np2)+np.inf)

        #########################  use eigen decomposition to initialize y  ####################
        # k = np1 + np2
        # Ct = torch.from_numpy(C).cuda().to(torch.float)
        # bt = torch.from_numpy(b).view(-1,1).cuda().to(torch.float)
        # Ck = Ct[-1, :].view(-1, Ct.shape[-1])
        # bk = bt[-1, :]
        # Ceq = torch.mm(torch.cat((torch.eye(k - 1), torch.zeros([k - 1, 1])), dim=-1).to(M.device),
        #                Ct - torch.mm(bt / bk, Ck))
        # Inn = torch.eye(np1 * np2).to(M.device)
        # PC = Inn - torch.mm(torch.mm(Ceq.t(), torch.inverse(torch.mm(Ceq, Ceq.t()))), Ceq)
        #
        # M2 = torch.bmm(torch.bmm(PC.expand(nbatch, -1, -1), M), PC.expand(nbatch, -1, -1))
        #
        # _, evecs = torch.symeig(M2, eigenvectors=True)
        # evec = evecs[:, :, -1]
        #########################  use eigen decomposition to initialize y  ####################

        # start = time()
        y = np.zeros([nbatch, np1*np2])
        lam = np.zeros([nbatch, np1*np2])  # dual variables for inequality constraints Gx <= h (-Ix<=0)
        nu = np.zeros([nbatch, np1+np2])   # dual variables for equality constraints Cx=b
        for ib in range(nbatch):
            Mi = M[ib, :, :].cpu().numpy()
            f_rq = f_rayleign_quotient(Mi)
            y0 = np.ones(np1*np2)
            # y0 = evec[ib, :].cpu().numpy()
            res = minimize(f_rq, x0=y0,
                           bounds=bounds, method='trust-constr', constraints=[linear_constraint])
            y[ib, :] = res.x
            lam[ib, :] = res.v[1]
            nu[ib, :] = res.v[0]
        # print('forward optimization time = ', time() - start)

        y = torch.from_numpy(y).to(M.device).to(torch.float)
        C = torch.from_numpy(C).to(M.device).to(torch.float)
        b = torch.from_numpy(b).to(M.device).to(torch.float)
        l = torch.from_numpy(lam).to(M.device).to(torch.float)
        n = torch.from_numpy(nu).to(M.device).to(torch.float)

        ny = y.shape[-1]
        nl = l.shape[-1]  # for the dual variable lambda
        nn = n.shape[-1]  # for the dual variable nu
        nf = ny + nl + nn

        ############################################################
        # Define implicit functions
        ############################################################
        def stationarity(j, argv):
            M, y, l, n, C, b = argv
            # print('M.shape = ', M.shape, M.device)
            # print('y.shape = ', y.shape, y.device)
            # print('l.shape = ', l.shape, l.device)
            # print('n.shape = ', n.shape, n.device)
            # print('C.shape = ', C.shape, C.device)
            # print('b.shape = ', b.shape, b.device)

            y = y.unsqueeze(-1)
            l = l.unsqueeze(-1).to(torch.float)
            n = n.unsqueeze(-1).to(torch.float)
            yty = torch.matmul(y.permute(0,2,1), y+1e-10).squeeze(-1)
            ytMy = torch.matmul(torch.matmul(y.permute(0,2,1), M), y+1e-10).squeeze()
            # print('yty = ', yty.shape)
            # print('ytMy = ', ytMy.shape)
            # print('M[:, j, :] = ', M[:, j, :].shape)
            # print('yty * M[:, j, :] = ', (yty * M[:, j, :]).shape)
            # print('batch_diagonal(ytMy)', batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :].shape)
            nor = (yty * M[:, j, :] - batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :])
            # print('nor = ', nor)
            left = 2 * torch.matmul(nor.unsqueeze(1), y[:, :, :]).squeeze(1) / yty*yty
            # print('left = ', left)
            # torch.set_printoptions(threshold=5000)
            # print('C=', C)
            I = torch.eye(ny).to(C.device).to(torch.float)
            CT_nu = torch.matmul(C.permute(1, 0)[j, :].expand(nbatch, -1).unsqueeze(1), n[:, :, :]).squeeze()
            IT_lam = torch.matmul(I[j, :].expand(nbatch, -1).unsqueeze(1), l[:, :, :]).squeeze()
            # print('right = ', right)
            return left + CT_nu - IT_lam

        def primal_fea(j, argv):  # equality constraints Cx=b
            _, y, _, _, C, _ = argv
            y = y.unsqueeze(-1)
            return torch.bmm(C[j, :].expand(nbatch, -1).unsqueeze(1), y[:,:,:]).squeeze() - 1  # b === 1

        def comp_slack(j, argv):  # inequality constraints -Ix <= 0
            _, y, l, _, _, _ = argv
            y = y.unsqueeze(-1)
            l = l.unsqueeze(-1).to(torch.float)
            # D(lam)( -I * y ) = 0
            Ij = torch.zeros(np1*np2).to(y.device)
            Ij[j] = -1
            Gy = torch.bmm(Ij.expand(nbatch, -1).unsqueeze(1), y[:,:,:]).squeeze()
            return l[:, j, :] * Gy  # h = 0

        f_dict = {stationarity: (0, ny),
                  primal_fea: (ny, ny + nn),
                  comp_slack: (ny + nn, nf)}

        ############################################################
        # Construct the imp_struct
        ############################################################
        imp_struct = ImpStruct()
        imp_struct.x = M
        imp_struct.params = []
        imp_struct.y = y
        imp_struct.duals = [l, n]
        imp_struct.other_inputs = [C, b]
        imp_struct.none_grad = 2
        imp_struct.f_dict = f_dict
        ctx.imp_struct = imp_struct

        return y


    # @staticmethod
    # def backward(ctx, *grad_outputs):
    #     M, y, C, b, l, n, np1, np2 = ctx.saved_tensors
    #     M0 = M.detach().cpu().numpy()
    #     y0 = y.detach().cpu().numpy()
    #     C0 = C.cpu().numpy()
    #     l0 = l.detach().cpu().numpy()
    #     n0 = n.detach().cpu().numpy()
    #     np1, np2 = int(np1), int(np2)
    #     nbatch = M0.shape[0]
    #     # print(nbatch, np1, np2)
    #     # l0 = np.zeros([nbatch, np1 + np2])
    #
    #     ny = y0.shape[-1]
    #     nM = ny * ny
    #     nl = l0.shape[-1]  # for the dual variable lambda
    #     nn = n0.shape[-1]  # for the dual variable nu
    #     nf = ny + nl + nn
    #     # print('eval = ', eval)
    #     # print('ny = ', ny)
    #     ######
    #     # # solve for lambda by least square: min || C.t() * lambda - bb ||_2
    #     # I0 = np.eye(ny)
    #     # x0 = np.ones(np1 + np2)
    #     # for ib in range(nbatch):
    #     #     yi = y0[ib, :]
    #     #     Mi = M0[ib, :, :]
    #     #     yty = yi.dot(yi)
    #     #     ytMy = yi.dot(Mi).dot(yi)
    #     #     # print('yi  = ', yi  )
    #     #     # print('Mi  = ', Mi  )
    #     #     # print('yty = ', yty )
    #     #     # print('ytMy= ', ytMy)
    #     #     bb = - 2*((Mi * yty - ytMy * I0) / yty**2).dot(yi)
    #     #
    #     #     def f(x):
    #     #         return np.linalg.norm(C0.transpose().dot(x) - bb)
    #     #
    #     #     l0[ib, :] = minimize(f, x0=x0, method='trust-constr').x
    #     #     # l0[ib, :] = lsq_linear(C0.transpose(), bb).x
    #     #     # print('C0 = ', C0)
    #     #     # print('bb = ', bb)
    #     #     # print('x = ', lsq_linear(C0.transpose(), bb).x)
    #     ######
    #     # print('l0 = ', l0)
    #     # return None
    #     J_F_y = torch.zeros((nbatch, nf, ny))
    #     J_F_M = torch.zeros((nbatch, nf, nM))
    #     J_F_l = torch.zeros((nbatch, nf, nl))
    #     J_F_n = torch.zeros((nbatch, nf, nn))
    #
    #     y = torch.from_numpy(y0).unsqueeze(-1)
    #     M = torch.from_numpy(M0)
    #     l = torch.from_numpy(l0).unsqueeze(-1).to(torch.float)
    #     n = torch.from_numpy(n0).unsqueeze(-1).to(torch.float)
    #     C = torch.from_numpy(C0)
    #
    #     # print('y.shape = ', y.shape)
    #     # print('M.shape = ', M.shape)
    #     # print('l.shape = ', l.shape)
    #
    #     y.requires_grad = True
    #     M.requires_grad = True
    #     l.requires_grad = True
    #     n.requires_grad = True
    #
    #     def stationarity(j):
    #         yty = torch.matmul(y.permute(0,2,1), y+1e-10).squeeze(-1)
    #         ytMy = torch.matmul(torch.matmul(y.permute(0,2,1), M), y+1e-10).squeeze()
    #         # print('yty = ', yty.shape)
    #         # print('ytMy = ', ytMy.shape)
    #         # print('M[:, j, :] = ', M[:, j, :].shape)
    #         # print('yty * M[:, j, :] = ', (yty * M[:, j, :]).shape)
    #         # print('batch_diagonal(ytMy)', batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :].shape)
    #         nor = (yty * M[:, j, :] - batch_diagonal(ytMy.view(nbatch, -1).repeat(1, ny))[:, j, :])
    #         # print('nor = ', nor)
    #         left = 2 * torch.matmul(nor.unsqueeze(1), y[:, :, :]).squeeze(1) / yty*yty
    #         # print('left = ', left)
    #         # torch.set_printoptions(threshold=5000)
    #         # print('C=', C)
    #         I = torch.eye(ny).to(C.device).to(torch.float)
    #         CT_nu = torch.matmul(C.permute(1, 0)[j, :].expand(nbatch, -1).unsqueeze(1), n[:, :, :]).squeeze()
    #         IT_lam = torch.matmul(I[j, :].expand(nbatch, -1).unsqueeze(1), l[:, :, :]).squeeze()
    #         # print('right = ', right)
    #         return left + CT_nu - IT_lam
    #
    #     def primal_fea(j):  # equality constraints Cx=b
    #         return torch.bmm(C[j, :].expand(nbatch, -1).unsqueeze(1), y[:,:,:]).squeeze() - 1  # b === 1
    #
    #     def comp_slack(j):  # inequality constraints -Ix <= 0
    #         # D(lam)( -I * y ) = 0
    #         Ij = torch.zeros(np1*np2)
    #         Ij[j] = -1
    #         Gy = torch.bmm(Ij.expand(nbatch, -1).unsqueeze(1), y[:,:,:]).squeeze()
    #         return l[:, j, :] * Gy  # h = 0
    #
    #
    #     functions = {stationarity: (0, ny),
    #                  primal_fea: (ny, ny + nn),
    #                  comp_slack: (ny + nn, nf)}
    #
    #     # start = time()
    #     for func in functions.keys():
    #         for j in range(functions[func][0], functions[func][1]):
    #             with torch.set_grad_enabled(True):
    #                 aux = func(j - functions[func][0])
    #                 aux.backward(torch.ones_like(aux))
    #             J_F_y[:, j, :] = y.grad.view(nbatch, -1)
    #             J_F_M[:, j, :] = M.grad.view(nbatch, -1)
    #             J_F_l[:, j, :] = l.grad.view(nbatch, -1)
    #             J_F_n[:, j, :] = n.grad.view(nbatch, -1)
    #             y.grad.zero_()
    #             M.grad.zero_()
    #             l.grad.zero_()
    #             n.grad.zero_()
    #
    #     # print('backward build Jacobians time = ', time() - start)
    #     # start = time()
    #     J_F_var = torch.cat((J_F_y, J_F_l, J_F_n), dim=-1)
    #     # print(J_F_var)
    #     J_var_M = - torch.matmul(torch.pinverse(J_F_var), J_F_M)
    #     J_y_M = J_var_M[:, :ny, :]
    #     dL_dy = grad_outputs[0].detach().cpu()
    #     dL_dM = torch.matmul(torch.transpose(J_y_M, 1, 2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
    #     dL_dM = dL_dM.view(nbatch, -1, ny)
    #     # print('backward inverse time = ', time() - start)
    #     grads = (dL_dM.to(grad_outputs[0].device), None, None)
    #     return grads


class Imp_SMAC_F_biEigen(Function):
    """
    The implicit layer implementation of the Spectral Matching with Affine Constraints (SMAC) algorithm
    Solve:  Eigen decomposition
    Forward: Eigen decomposition
    Backward: Eigen decomposition equations
    """

    @staticmethod
    def forward(ctx, *args):
        M, np1, np2 = args[0], args[1], args[2]
        nbatch = M.shape[0]

        C = torch.zeros([np1 + np2, np1 * np2]).to(M.device)
        for i in range(np1):
            C[i, i * np2:(i + 1) * np2] = 1
        for i in range(np1, np1 + np2):
            C[i, range(i - np1, np1 * np2, np2)] = 1

        b = torch.ones([np1 + np2, 1]).to(M.device)

        k = np1 + np2
        Ck = C[-1, :].view(-1, C.shape[-1])
        bk = b[-1, :]
        Ceq = torch.mm(torch.cat((torch.eye(k - 1), torch.zeros([k - 1, 1])), dim=-1).to(M.device),
                       C - torch.mm(b / bk, Ck))
        Inn = torch.eye(np1 * np2).to(M.device)
        PC = Inn - torch.mm(torch.mm(Ceq.t(), torch.inverse(torch.mm(Ceq, Ceq.t()))), Ceq)

        M2 = torch.bmm(torch.bmm(PC.expand(nbatch, -1, -1), M), PC.expand(nbatch, -1, -1))

        evals, evecs = torch.symeig(M2, eigenvectors=True)
        evec = evecs[:, :, -1]
        eval = evals[:, -1]
        ctx.save_for_backward(M2, eval, evec, C, b, torch.tensor(np1), torch.tensor(np2))
        return evec

    @staticmethod
    def backward(ctx, *grad_outputs):
        M, eval, evec, C, b, np1, np2 = ctx.saved_tensors
        M0 = M.detach().cpu().numpy()
        y0 = evec.detach().cpu().numpy()
        l0 = eval.detach().cpu().numpy()

        nbatch = M0.shape[0]
        ny = y0.shape[-1]
        nM = ny * ny
        nl = 1  # for the dual variable lambda
        nf = ny + nl

        J_F_y = torch.zeros((nbatch, nf, ny))
        J_F_M = torch.zeros((nbatch, nf, nM))
        J_F_l = torch.zeros((nbatch, nf, nl))

        y = torch.from_numpy(y0).unsqueeze(-1)
        M = torch.from_numpy(M0)
        l = torch.from_numpy(l0).unsqueeze(-1)

        # print('y.shape = ', y)
        # print('M.shape = ', M)
        # print('l.shape = ', l)

        y.requires_grad = True
        M.requires_grad = True
        l.requires_grad = True

        def stationarity(j):
            I_row = torch.zeros(ny)
            I_row[j] = 1
            return torch.matmul(M[:, j, :].unsqueeze(1) - I_row.expand(nbatch, -1).unsqueeze(1) * l.unsqueeze(-1),
                                y[:, :, :]).squeeze(1)

        def primal_fea(j):
            return torch.matmul(y.view(nbatch, -1), y + 1e-10)  # + 0 * M + 0 * l

        functions = {stationarity: (0, ny),
                     primal_fea: (ny, ny+1)}

        for func in functions.keys():
            for j in range(functions[func][0], functions[func][1]):
                with torch.set_grad_enabled(True):
                    aux = func(j - functions[func][0])
                    aux.backward(torch.ones_like(aux))
                J_F_y[:, j, :] = y.grad.view(nbatch, -1)
                J_F_M[:, j, :] = M.grad.view(nbatch, -1)
                J_F_l[:, j, :] = l.grad.view(nbatch, -1)
                y.grad.zero_()
                M.grad.zero_()
                l.grad.zero_()

        np.savez('J_F', J_F_M=J_F_M, J_F_y=J_F_y, J_F_l=J_F_l)
        J_F_var = torch.cat((J_F_y, J_F_l), dim=-1)
        J_var_M = - torch.matmul(torch.inverse(J_F_var), J_F_M)
        J_y_M = J_var_M[:, :ny, :]
        dL_dy = grad_outputs[0].detach().cpu()
        dL_dM = torch.matmul(torch.transpose(J_y_M, 1, 2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
        dL_dM = dL_dM.view(nbatch, -1, ny)

        grads = (dL_dM.to(grad_outputs[0].device), None, None)
        return grads


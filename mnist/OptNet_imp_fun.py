import torch
from torch.autograd import Function, Variable
from qpth.util import bger, expandParam, extract_nBatch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from qpth import solvers
from qpth.solvers.pdipm import batch as pdipm_b
from qpth.solvers.pdipm import spbatch as pdimp_spb
import numpy as np
import sys
sys.path.append('../ImpFuncToolbox/')
from ImpFunction import *
# from ... import ImpFuction


from enum import Enum


class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2


solver = QPSolvers.PDIPM_BATCHED

'''
z = argmin_z  0.5 * zT * Q * z + pT * z
s.t.    Gz <= h
        Az =  b
'''


class OptNet_imp_fun(Module):
    def __init__(self, in_features, out_features, neq, nineq, eps=1e-4, device=torch.device('cuda')):
        super(OptNet_imp_fun, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neq = neq
        self.nineq = nineq
        self.eps = eps
        self.device = device

        # M and L are to form a semi-definite positive Q
        self.M = Parameter(torch.tril(torch.ones(out_features, out_features)))
        self.L = Parameter(torch.tril(torch.rand(out_features, out_features)))
        self.p = Parameter(torch.Tensor(out_features))
        self.G = Parameter(torch.Tensor(nineq, out_features).uniform_(-1, 1))
        self.A = Parameter(torch.Tensor(neq, out_features))

        # instead of having parameter 'h' as in the inequality constraints,
        # introduce slack variables 's'
        # self.h = Parameter(torch.Tensor(nineq))
        self.s0 = Parameter(torch.ones(nineq))
        self.b = Parameter(torch.Tensor(neq))
        self.z0 = Parameter(torch.zeros(out_features))

        # self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        nBatch = x.size(0)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.out_features))
        Q = Q.unsqueeze(0).expand(nBatch, self.out_features, self.out_features)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.out_features)
        z0 = self.z0.unsqueeze(0).expand(nBatch, self.out_features)
        s0 = self.s0.unsqueeze(0).expand(nBatch, self.nineq)
        h = z0.mm(self.G.t()) + s0
        A = Variable(torch.Tensor())
        b = Variable(torch.Tensor())

        return OptNet_imp_fun_F_encap.apply(x, Q, G, h, A, b)


# class OptNet_imp_fun_F(Function):
#     @staticmethod
#     def forward(ctx, Q_, p_, G_, h_, A_, b_, f_dict):
#         eps = 1e-12
#         verbose = 0
#         notImprovedLim = 3
#         maxIter = 20
#         nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
#         Q, _ = expandParam(Q_, nBatch, 3)
#         p, _ = expandParam(p_, nBatch, 2)
#         G, _ = expandParam(G_, nBatch, 3)
#         h, _ = expandParam(h_, nBatch, 2)
#         A, _ = expandParam(A_, nBatch, 3)
#         b, _ = expandParam(b_, nBatch, 2)
#
#         _, nineq, nz = G.size()
#         neq = A.size(1) if A.nelement() > 0 else 0
#         assert (neq > 0 or nineq > 0)
#         ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz
#         ctx.f_dict = f_dict
#
#         if solver == QPSolvers.PDIPM_BATCHED:
#             ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
#             zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
#                 Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
#                 eps, verbose, notImprovedLim, maxIter)
#         elif solver == QPSolvers.CVXPY:
#             vals = torch.Tensor(nBatch).type_as(Q)
#             zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
#             lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
#             nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
#                 if ctx.neq > 0 else torch.Tensor()
#             slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
#             for i in range(nBatch):
#                 Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
#                 vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(
#                     *[x.cpu().numpy() if x is not None else None
#                       for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
#                 # if zhati[0] is None:
#                 #     import IPython, sys; IPython.embed(); sys.exit(-1)
#                 zhats[i] = torch.Tensor(zhati)
#                 lams[i] = torch.Tensor(lami)
#                 slacks[i] = torch.Tensor(si)
#                 if neq > 0:
#                     nus[i] = torch.Tensor(nui)
#
#             ctx.vals = vals
#             ctx.lams = lams
#             ctx.nus = nus
#             ctx.slacks = slacks
#         else:
#             assert False
#
#         # ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
#         ctx.output = (zhats, ctx.lams, ctx.nus)
#         ctx.input = (Q_, p_, G_, h_, A_, b_)
#
#         return zhats
#
#     @staticmethod
#     def backward(ctx, *dl_dzhat):
#         f_dict = ctx.f_dict
#         # print(f_dict)
#         zhats, lams, nus = ctx.output
#         Q, p, G, h, A, b = ctx.input
#         # zhats, Q, p, G, h, A, b = ctx.saved_tensors
#         neq, nineq = ctx.neq, ctx.nineq
#         grad_outputs = dl_dzhat[0].detach().numpy()
#         y0 = zhats.detach().numpy()
#         x0 = p.detach().numpy()
#         Q0 = Q.detach().numpy()
#         G0 = G.detach().numpy()
#         h0 = h.detach().numpy()
#         A0 = A.detach().numpy()
#         b0 = b.detach().numpy()
#         nbatch = y0.shape[0]
#
#         if nineq is not 0:
#             l0 = ctx.lams.detach().numpy()  # lambda
#         else:
#             l0 = np.zeros((0, 0))
#         if neq is not 0:
#             n0 = ctx.nus.detach().numpy()  # nu
#         else:
#             n0 = np.zeros((0, 0))
#
#         # print('Q0.shape = ', Q0.shape)
#         # print('A0.shape = ', A0.shape)
#         # print('G0.shape = ', G0.shape)
#         # print('y0.shape = ', y0.shape)
#         # print('x0.shape = ', x0.shape)
#         # print('l0.shape = ', l0.shape)
#         # print('n0.shape = ', n0.shape)
#         # print('h0.shape = ', h0.shape)
#         # print('b0.shape = ', b0.shape)
#
#         # print('Q = ', Q)
#
#         ny = y0.shape[-1]
#         nx = x0.shape[-1]
#         nl = l0.shape[-1]
#         nn = n0.shape[-1]
#         nQ = Q0.shape[-1] * Q0.shape[-2]
#         nG = G0.shape[-1] * G0.shape[-2]
#         nh = h0.shape[-1]
#         if neq is not 0:
#             nA = A0.shape[-1] * A0.shape[-2]
#         else:
#             nA = 0
#         nb = b0.size
#         nf = ny + nl + nn  # number of implicit functions = number of outputs
#
#         J_F_y = np.zeros((nbatch, nf, ny))
#         J_F_x = np.zeros((nbatch, nf, nx))
#         J_F_l = np.zeros((nbatch, nf, nl))
#         J_F_n = np.zeros((nbatch, nf, nn))
#         J_F_Q = np.zeros((nbatch, nf, nQ))
#         J_F_G = np.zeros((nbatch, nf, nG))
#         J_F_h = np.zeros((nbatch, nf, nh))
#         J_F_A = np.zeros((nbatch, nf, nA))
#         J_F_b = np.zeros((nbatch, nf, nb))
#
#         # print('J_F_y.shape = ', J_F_y.shape)
#         # print('J_F_x.shape = ', J_F_x.shape)
#         # print('J_F_l.shape = ', J_F_l.shape)
#         # print('J_F_n.shape = ', J_F_n.shape)
#         # print('J_F_Q.shape = ', J_F_Q.shape)
#         # print('J_F_G.shape = ', J_F_G.shape)
#         # print('J_F_h.shape = ', J_F_h.shape)
#         # print('J_F_A.shape = ', J_F_A.shape)
#         # print('J_F_b.shape = ', J_F_b.shape)
#
#         y = torch.from_numpy(y0).unsqueeze(-1)  # nbatch x ny x 1
#         x = torch.from_numpy(x0).unsqueeze(-1)  # nbatch x nx x 1
#         l = torch.from_numpy(l0).unsqueeze(-1)  # nbatch x nl x 1
#         n = torch.from_numpy(n0).unsqueeze(-1)  # nbatch x nn x 1
#         # Q = torch.from_numpy(np.tile(Q0, (nbatch, 1, 1)))     # nbatch x ny x ny
#         Q = torch.from_numpy(Q0)  # nbatch x ny x ny
#         # G = torch.from_numpy(np.tile(G0, (nbatch, 1, 1)))     # nbatch x nineq x ny
#         G = torch.from_numpy(G0)  # nbatch x nineq x ny
#         # h = torch.from_numpy(np.tile(h0.reshape((-1, 1)), (nbatch, 1, 1)))   # nbatch x nh x 1
#         h = torch.from_numpy(h0).unsqueeze(-1)  # nbatch x nh x 1
#         # A = torch.from_numpy(np.tile(A0, (nbatch, 1, 1)))     # nbatch x neq x 1
#         A = torch.from_numpy(A0)  # nbatch x neq x 1
#         b = torch.from_numpy(np.tile(b0.reshape((-1, 1)), (nbatch, 1, 1)))  # nbatch x nb x 1
#
#         y.requires_grad = True
#         x.requires_grad = True
#         l.requires_grad = True
#         n.requires_grad = True
#         Q.requires_grad = True
#         G.requires_grad = True
#         h.requires_grad = True
#         A.requires_grad = True
#         b.requires_grad = True
#
#         # print('Q = ', Q)
#
#         # Define implicit functions.  Generally, there are 3 groups of functions (by KKT):
#         # 1. stationarity
#         # 2. primal feasibility
#         # 3. complementary slackness
#
#         for func in f_dict.keys():
#             for j in range(f_dict[func][0], f_dict[func][1]):
#                 with torch.set_grad_enabled(True):
#                     aux = func(j - f_dict[func][0], Q, x, G, h, A, b, y, n, l)
#                     aux.backward(torch.ones_like(aux))
#
#                 J_F_x[:, j, :] = x.grad.numpy().squeeze()
#                 J_F_y[:, j, :] = y.grad.numpy().squeeze()
#                 # print('y.grad = ', y.grad)
#                 if neq is not 0:
#                     J_F_n[:, j, :] = n.grad.numpy().squeeze()
#                     J_F_A[:, j, :] = np.reshape(A.grad.numpy(), (nbatch, -1))
#                 J_F_l[:, j, :] = l.grad.numpy().squeeze()
#                 J_F_Q[:, j, :] = np.reshape(Q.grad.numpy(), (nbatch, -1))
#                 J_F_G[:, j, :] = np.reshape(G.grad.numpy(), (nbatch, -1))
#                 J_F_h[:, j, :] = np.reshape(h.grad.numpy(), (nbatch, -1))
#                 J_F_b[:, j, :] = np.reshape(b.grad.numpy(), (nbatch, -1))
#
#                 y.grad.zero_()
#                 x.grad.zero_()
#                 if neq is not 0:
#                     n.grad.zero_()
#                     A.grad.zero_()
#                 l.grad.zero_()
#                 Q.grad.zero_()
#                 G.grad.zero_()
#                 h.grad.zero_()
#                 b.grad.zero_()
#
#         # var is a concatenation of all values: y, lambda, and nu
#         J_F_var = np.concatenate((J_F_y, J_F_n, J_F_l), axis=-1)
#
#         # print('J_F_y = ', J_F_y.shape, J_F_y)
#         # print('J_F_var = ', J_F_var.shape, J_F_var)
#
#         J_F_var_inv = np.linalg.inv(J_F_var)
#         J_var_x = - np.matmul(J_F_var_inv, J_F_x)
#         J_y_x = J_var_x[:, :ny, :]
#         dL_dx = np.matmul(np.transpose(J_y_x, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze()
#
#         J_var_Q = - np.matmul(J_F_var_inv, J_F_Q)
#         J_var_G = - np.matmul(J_F_var_inv, J_F_G)
#         J_var_h = - np.matmul(J_F_var_inv, J_F_h)
#         J_var_A = - np.matmul(J_F_var_inv, J_F_A)
#         J_var_b = - np.matmul(J_F_var_inv, J_F_b)
#
#         J_y_Q = J_var_Q[:, :ny, :]
#         J_y_G = J_var_G[:, :ny, :]
#         J_y_h = J_var_h[:, :ny, :]
#         J_y_A = J_var_A[:, :ny, :]
#         J_y_b = J_var_b[:, :ny, :]
#
#         # print('grad_outputs = ', grad_outputs.shape)
#         # print('J_y_Q.shape = ', J_y_Q.shape)
#         # print('J_y_G.shape = ', J_y_G.shape)
#         # print('J_y_h.shape = ', J_y_h.shape)
#         # print('J_y_A.shape = ', J_y_A.shape)
#         # print('J_y_b.shape = ', J_y_b.shape)
#
#         # dL_dQ = np.sum(np.matmul(np.transpose(J_y_Q, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)), axis=0).squeeze()
#         # dL_dG = np.sum(np.matmul(np.transpose(J_y_G, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)), axis=0).squeeze()
#         # dL_dh = np.sum(np.matmul(np.transpose(J_y_h, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)), axis=0).squeeze()
#         # dL_dA = np.sum(np.matmul(np.transpose(J_y_A, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)), axis=0).squeeze()
#         # dL_db = np.sum(np.matmul(np.transpose(J_y_b, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)), axis=0).squeeze()
#
#         dL_dQ = np.matmul(np.transpose(J_y_Q, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze(-1)
#         dL_dG = np.matmul(np.transpose(J_y_G, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze(-1)
#         dL_dh = np.matmul(np.transpose(J_y_h, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze(-1)
#         dL_dA = np.matmul(np.transpose(J_y_A, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze(-1)
#         dL_db = np.matmul(np.transpose(J_y_b, [0, 2, 1]), np.expand_dims(grad_outputs, axis=2)).squeeze(-1)
#
#         dL_dQ = dL_dQ.reshape((nbatch, -1, ny))
#         dL_dG = dL_dG.reshape((nbatch, -1, ny))
#         dL_dA = dL_dA.reshape((nbatch, -1, ny))
#
#         # grads = (dQs, dps,   dGs,   dhs,   dAs,   dbs)
#         # print('dL_dQ.size = ', dL_dQ.shape)
#         # print('dL_dx.size = ', dL_dx.shape)
#         # print('dL_dG.size = ', dL_dG.shape)
#         # print('dL_dh.size = ', dL_dh.shape)
#         # print('dL_dA.size = ', dL_dA.shape)
#         # print('dL_db.size = ', dL_db.shape)
#         grads = (torch.from_numpy(dL_dQ).to(torch.float),
#                  torch.from_numpy(dL_dx).to(torch.float),
#                  torch.from_numpy(dL_dG).to(torch.float),
#                  torch.from_numpy(dL_dh).to(torch.float),
#                  torch.from_numpy(dL_dA).to(torch.float),
#                  torch.from_numpy(dL_db).to(torch.float), None)
#         # print('out backward')
#         return grads




class OptNet_imp_fun_F_encap(ImpFunction):
    @staticmethod
    def forward(ctx, p_, Q_, G_, h_, A_, b_):

        ############################################################
        #  The forward solver
        ############################################################
        eps = 1e-12
        verbose = 0
        notImprovedLim = 3
        maxIter = 20
        nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
        Q, _ = expandParam(Q_, nBatch, 3)
        p, _ = expandParam(p_, nBatch, 2)
        G, _ = expandParam(G_, nBatch, 3)
        h, _ = expandParam(h_, nBatch, 2)
        A, _ = expandParam(A_, nBatch, 3)
        b, _ = expandParam(b_, nBatch, 2)

        _, nineq, nz = G.size()
        ny = Q.shape[-2]
        neq = A.size(1) if A.nelement() > 0 else 0
        assert (neq > 0 or nineq > 0)
        ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz


        if solver == QPSolvers.PDIPM_BATCHED:
            ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
            zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                eps, verbose, notImprovedLim, maxIter)
        elif solver == QPSolvers.CVXPY:
            vals = torch.Tensor(nBatch).type_as(Q)
            zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
            lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                if ctx.neq > 0 else torch.Tensor()
            slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            for i in range(nBatch):
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                vals[i], zhati, nui, lami, si = solvers.cvxpy.forward_single_np(
                    *[x.cpu().numpy() if x is not None else None
                      for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
                # if zhati[0] is None:
                #     import IPython, sys; IPython.embed(); sys.exit(-1)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(lami)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(nui)

            ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks
        else:
            assert False

        # ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
        if ctx.lams is None:
            ctx.lams = torch.Tensor()
        if ctx.nus is None:
            ctx.nus = torch.Tensor()

        ############################################################
        # Define implicit functions
        ############################################################
        def stationarity(j, argv):
            # argv = tuple([x]) + tuple(params) + tuple([y]) + tuple(duals)
            # print(len(argv))
            x, Q, G, h, A, b, y, l, n = argv
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            h = h.unsqueeze(-1)
            b = b.unsqueeze(-1)
            n = n.unsqueeze(-1)
            l = l.unsqueeze(-1)

            # print('Q0.shape = ', Q.shape)
            # print('A0.shape = ', A.shape)
            # print('G0.shape = ', G.shape)
            # print('y0.shape = ', y.shape)
            # print('x0.shape = ', x.shape)
            # print('l0.shape = ', l.shape)
            # print('n0.shape = ', n.shape)
            # print('h0.shape = ', h.shape)
            # print('b0.shape = ', b.shape)

            Qy = torch.matmul(Q[:, j, :].unsqueeze(1), y[:, :, :]).squeeze(1)
            if neq is 0:
                ATn = torch.zeros_like(Qy)
            else:
                ATn = torch.matmul(torch.transpose(A, 1, 2)[:, j, :].unsqueeze(1), n[:, :, :]).squeeze(1)
            if nineq is 0:
                GTl = torch.zeros_like(Qy)
            else:
                GTl = torch.matmul(torch.transpose(G, 1, 2)[:, j, :].unsqueeze(1), l[:, :, :]).squeeze(1)

            return Qy + x[:, j, :] + ATn + GTl + 0 * h.mean() + 0 * b.mean()

        def primal_fea(j, argv):
            x, Q, G, h, A, b, y, l, n = argv
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            h = h.unsqueeze(-1)
            b = b.unsqueeze(-1)
            n = n.unsqueeze(-1)
            l = l.unsqueeze(-1)
            Ay = torch.matmul(A[:, j, :].unsqueeze(1), y[:, :, :]).squeeze(1)
            return Ay - b[:, j, :]

        def comp_slack(j, argv):
            x, Q, G, h, A, b, y, l, n = argv
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            h = h.unsqueeze(-1)
            b = b.unsqueeze(-1)
            n = n.unsqueeze(-1)
            l = l.unsqueeze(-1)
            Gy = torch.matmul(G[:, j, :].unsqueeze(1), y[:, :, :]).squeeze(1)
            return l[:, j, :] * (Gy - h[:, j, :])

        nf = ny + neq + nineq
        f_dict = {stationarity: (0, ny),
                  primal_fea: (ny, ny + neq),
                  comp_slack: (ny + neq, nf)}

        ############################################################
        # Construct the imp_struct
        ############################################################
        imp_struct = ImpStruct()
        imp_struct.x = p_
        imp_struct.params = [Q_, G_, h_, A_, b_]
        imp_struct.y = zhats
        imp_struct.duals = [ctx.lams, ctx.nus]
        imp_struct.other_inputs = []
        imp_struct.f_dict = f_dict
        ctx.imp_struct = imp_struct

        # ctx.y = zhats
        # ctx.x = p_
        # ctx.duals = [ctx.lams, ctx.nus]
        # ctx.params = [Q_, G_, h_, A_, b_]

        return zhats



if __name__ == '__main__':
    print('hello')
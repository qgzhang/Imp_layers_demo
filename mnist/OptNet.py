import torch
from torch.autograd import Function, Variable
from qpth.util import bger, expandParam, extract_nBatch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from qpth import solvers
from qpth.solvers.pdipm import batch as pdipm_b
from qpth.solvers.pdipm import spbatch as pdimp_spb

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
class OptNet(Module):
    def __init__(self, in_features, out_features, neq, nineq, eps=1e-4):
        super(OptNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neq = neq
        self.nineq = nineq
        self.eps = eps

        # M and L are to form a semi-definite positive Q
        self.M = Parameter(torch.tril(torch.ones(out_features, out_features)))
        self.L = Parameter(torch.tril(torch.rand(out_features, out_features)))
        self.p = Parameter(torch.Tensor(out_features))
        self.G = Parameter(torch.Tensor(nineq, out_features).uniform_(-1,1))
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
        e = Variable(torch.Tensor())

        return OptNet_F.apply(Q, x, G, h, e, e)




class OptNet_F(Function):
    @staticmethod
    def forward(ctx, Q_, p_, G_, h_, A_, b_):
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

        ctx.save_for_backward(zhats, Q_, p_, G_, h_, A_, b_)
        return zhats

    @staticmethod
    def backward(ctx, *dl_dzhat):
        zhats, Q, p, G, h, A, b = ctx.saved_tensors
        nBatch = extract_nBatch(Q, p, G, h, A, b)
        Q, Q_e = expandParam(Q, nBatch, 3)
        p, p_e = expandParam(p, nBatch, 2)
        G, G_e = expandParam(G, nBatch, 3)
        h, h_e = expandParam(h, nBatch, 2)
        A, A_e = expandParam(A, nBatch, 3)
        b, b_e = expandParam(b, nBatch, 2)

        # neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz
        neq, nineq = ctx.neq, ctx.nineq

        if solver == QPSolvers.CVXPY:
            ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)

        # Clamp here to avoid issues coming up when the slacks are too small.
        # TODO: A better fix would be to get lams and slacks from the
        # solver that don't have this issue.
        d = torch.clamp(ctx.lams, min=1e-8) / torch.clamp(ctx.slacks, min=1e-8)

        pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)
        dx, _, dlam, dnu = pdipm_b.solve_kkt(
            ctx.Q_LU, d, G, A, ctx.S_LU,
            dl_dzhat[0], torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, nineq).type_as(G),
            torch.zeros(nBatch, neq).type_as(G) if neq > 0 else torch.Tensor())

        dps = dx
        dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
        if G_e:
            dGs = dGs.mean(0)
        dhs = -dlam
        if h_e:
            dhs = dhs.mean(0)
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
            dbs = -dnu
            if A_e:
                dAs = dAs.mean(0)
            if b_e:
                dbs = dbs.mean(0)
        else:
            dAs, dbs = None, None
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))
        if Q_e:
            dQs = dQs.mean(0)
        if p_e:
            dps = dps.mean(0)

        grads = (dQs, dps, dGs, dhs, dAs, dbs)

        return grads

























if __name__ == '__main__':
    print('hello')
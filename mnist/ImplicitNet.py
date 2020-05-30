import torch
from torch import nn
import torch.nn.functional as F


from OptNet import OptNet
from OptNet_imp_fun import OptNet_imp_fun
from enum import Enum


class Layers(Enum):

    OptNet = 5
    OptNet_imp_fun = 7


class ImplicitNet(nn.Module):
    def __init__(self, nFeatures=784, nHiddens=[600,10], nCls=10, bn=1, whichlayer=0):
        super(ImplicitNet, self).__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHiddens[0]
        self.nHidden2 = nHiddens[1]
        self.nCls = nCls
        self.bn = bn
        if bn:
            self.bn1 = nn.BatchNorm1d(nHiddens[0])
            self.bn2 = nn.BatchNorm1d(nHiddens[1])
        # e.g., a 784 * 128 * 32 * 10 network
        self.fc1 = nn.Linear(self.nFeatures, self.nHidden1)
        self.fc2 = nn.Linear(self.nHidden1, self.nHidden2)
        self.fc_temp = nn.Linear(self.nHidden2, self.nCls)
        self.temp = None
        self.y = None
        """ 
        here, instead of using a traditional layer, say a nn.Linear(nHidden2, nCls),
        comes to play an implicit layer in the form of 
        y* = argmin_y f_obj(theta, x, y), where 
        x in R^nHidden2 is the output of self.fc2,
        theta represents the parameters of the implicit layer,
        y in R^nCls is the output of the layer, 
        and F is some objective function.
        
        Tentatively, f_obj can be in the form of
        f_obj = |y - Wx -b|^2,  where W and b compose theta.
        (Choosing this formula to comply with the traditional definition: y = Wx + b) 
        
        The KKT condition of y* gives
        I_y * y - Wx - b = 0,
        and this is the definition of implicit function
        F(y, x, W, b) : I_y * y - Wx - b = 0,
        which apparently is consistent to the traditional y = Wx + b.
        
        However, we do not use explicit differential formulae to calculate the derivatives of y 
        w.r.t. W, x, and b.  Instead, we use the formulae for implicit functions to take gradients.
        """

        if whichlayer == Layers.OptNet.value:
            self.cusLayer = OptNet(self.nHidden2, self.nCls, 0, 10)
        elif whichlayer == Layers.OptNet_imp_fun.value:
            self.cusLayer = OptNet_imp_fun(self.nHidden2, self.nCls, 0, 10)

    def forward(self, x: torch.Tensor):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        # customised layer

        y = self.cusLayer(x)
        y.retain_grad()
        # y = F.sigmoid(y)

        return F.log_softmax(y, dim=1)

        # self.temp = temp
        # self.y = y
        # return F.log_softmax(y + temp, dim=1)






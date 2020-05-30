from torch.autograd import Function
import torch

'''
Instructions on using the Implicit Function Toolbox to define a implicitly defined layer.

1. Implement the layer from nn.Module as would normally do for explicit layers; 
    it take in x_ from the previous layer and output y_ to the following layer.
2. Define learnable parameters p_.
3. Define a sequence of operation for the forward path.  The core of the sequence might be
    y_ = F.apply(x_, p_, o_), where o_ denotes the non-learnable variables, 
    and F is an instance of class ImpFunction.
    
4. Implement F.forward(ctx, x_, p_, o_) 
    4.1 Solve for y_ and d_, where d_ (duals) is the output that are not really needed by 
            the following layers.
    4.2 Define implicit functions and store them to dictionary f_dict
    4.3 Construct an instance of ImpStruct (see below) and attach it to ctx.
'''

class ImpStruct:
    '''
    This class defines necessary fields that need to be assigned by the forward() function of
    any customised descendant of class ImpFunction.
    '''
    def __init__(self):
        '''
        self.f_dict: the dictionary storing implicit functions.

        The arguments of any implicit function are in the following order:
            index:          the sequential order of calling the implicit function
            x:              the output of the previous layer
            params:         learnable parameters of this layer
            y:              the desired output of this layer
                            (y is fed into the following layer,
                            and dL/dy is provided by auto-differentiation during backpropagation)
            duals:          the non-desired output of this layer 
            other_inputs:   non-learnable inputs, which do not need computing gradients
        
        self.none_grad: number of arguments of Function.forward that requires None as gradient
        
        It is users responsibility to ensure correctly assemble and unpack arguments
        '''
        self.f_dict = {}
        self.x = torch.Tensor()
        self.params = []
        self.y = torch.Tensor()
        self.duals = []
        self.other_inputs = []
        self.none_grad = 0


class ImpFunction(Function):
    '''
    Inherit this class by implementing the static method forward()
    '''
    @staticmethod
    def backward(ctx, *grad_outputs):
        imp_struct = ctx.imp_struct
        f_dict = imp_struct.f_dict
        # print(f_dict)
        y = imp_struct.y.clone().detach()
        y.requires_grad = True
        dL_dy = grad_outputs[0].clone().detach().cpu()
        duals = imp_struct.duals
        x = imp_struct.x.clone().detach()
        x.requires_grad = True
        params = imp_struct.params
        others = imp_struct.other_inputs

        nbatch = y.shape[0]
        ny = int(y.numel() / nbatch)
        nx = int(x.numel() / nbatch)

        nf = ny
        for i in range(len(duals)):
            nf += int(duals[i].numel() / nbatch)

        J_F_y = torch.zeros((nbatch, nf, ny))
        J_F_x = torch.zeros((nbatch, nf, nx))

        n_params = [0]*len(params)
        n_duals = [0]*len(duals)

        J_F_duals = []
        for i in range(len(duals)):
            n_duals[i] = int(duals[i].numel() / nbatch)
            J_F_duals.append(torch.zeros((nbatch, nf, n_duals[i])))
            duals[i] = duals[i].clone().detach()
            duals[i].requires_grad = True
            # print('duals[', i, '].shape = ', duals[i].shape, J_F_duals[i].shape)

        J_F_params = []
        for i in range(len(params)):
            n_params[i] = int(params[i].numel() / nbatch)
            J_F_params.append(torch.zeros((nbatch, nf, n_params[i])))
            params[i] = params[i].clone().detach()
            params[i].requires_grad = True
            # print('params[', i, '].shape = ', params[i].shape, J_F_params[i].shape)

        argv = tuple([x]) + tuple(params) + tuple([y]) + tuple(duals) + tuple(others)
        
        # the following code block is to ensure variables has valid 'grad' field
        with torch.set_grad_enabled(True):
            aux = 0 * torch.sum(x) + 0 * torch.sum(y)
            for i in range(len(duals)):
                aux += 0 * torch.sum(duals[i])
            for i in range(len(params)):
                aux += 0 * torch.sum(params[i])
            aux.backward(torch.ones_like(aux))

        for func in f_dict.keys():
            for j in range(f_dict[func][0], f_dict[func][1]):
                with torch.set_grad_enabled(True):
                    # print(len(argv))
                    aux = func(j - f_dict[func][0], argv)
                    aux.backward(torch.ones_like(aux))

                J_F_x[:, j, :] = x.grad.view(nbatch, -1)
                x.grad.zero_()
                J_F_y[:, j, :] = y.grad.view(nbatch, -1)
                y.grad.zero_()
                for i in range(len(duals)):
                    if n_duals[i] is not 0:
                        J_F_duals[i][:, j, :] = duals[i].grad.view(nbatch, -1)
                        duals[i].grad.zero_()
                for i in range(len(params)):
                    if n_params[i] is not 0:
                        J_F_params[i][:, j, :] = params[i].grad.view(nbatch, -1)
                        params[i].grad.zero_()

        J_F_out = torch.cat(tuple([J_F_y]) + tuple(J_F_duals), dim=-1)
        # print('J_F_out.shape = ', J_F_out.shape)
        J_F_out_inv = torch.pinverse(J_F_out)
        J_out_x = - torch.matmul(J_F_out_inv, J_F_x)
        J_y_x = J_out_x[:, :ny, :]

        dL_dx = torch.matmul(torch.transpose(J_y_x, 1,2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
        dL_dx = dL_dx.view(x.shape).to(grad_outputs[0].device)

        dL_dparams = []
        for i in range(len(params)):
            J_out_pi = - torch.matmul(J_F_out_inv, J_F_params[i])
            J_y_pi = J_out_pi[:, :ny, :]
            dL_dpi = torch.matmul(torch.transpose(J_y_pi, 1, 2), torch.unsqueeze(dL_dy, 2)).squeeze(-1)
            dL_dpi = dL_dpi.view(params[i].shape).to(grad_outputs[0].device)
            dL_dparams.append(dL_dpi)

        grads = (dL_dx,) + tuple(dL_dparams) + tuple([None] * imp_struct.none_grad)

        return grads

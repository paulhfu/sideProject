import torch
import numpy as np

class CstmTensor(torch.Tensor):

    def backward(self, gradient=None, retain_graph=True, create_graph=False):
        super.backward(gradient=gradient, retain_graph=True, create_graph=create_graph)


if __name__ == '__main__':
    ta = np.zeros(100)
    tt = torch.tensor(ta)
    tct = torch.tensor(ta)

    tct = torch.Tensor(tct)

    a=1
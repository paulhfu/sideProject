from torch import optim


# Non-centered RMSprop update with shared statistics (without momentum)
class CstmAdam(optim.Adam):
  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    super(CstmAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

  def share_memory(self):
    for group in self.param_groups:
      for p in group['params']:
        state = self.state[p]
        state['step'].share_memory_()
        state['square_avg'].share_memory_()

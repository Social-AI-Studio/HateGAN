import torch

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model, opt):
    for name, value in model.named_parameters():
        if name=='senti_emb.emb.weight':
            value.requires_grad=False
    params = filter(lambda p: p.requires_grad, model.parameters())
    return NoamOpt(opt.NUM_HIDDEN, 2, opt.WARM_UP, torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9))

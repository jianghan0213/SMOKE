import torch
import linklink as link

# Taken from Prototype
class DistModule(torch.nn.Module):
   def __init__(self, module, sync=False):
      super(DistModule, self).__init__()
      self.module = module
      self.broadcast_params()

      self.sync = sync
      if not sync:
            self._grad_accs = []
            self._register_hooks()

   def forward(self, *inputs, **kwargs):
      return self.module(*inputs, **kwargs)

   def _register_hooks(self):
      for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
               p_tmp = p.expand_as(p)
               grad_acc = p_tmp.grad_fn.next_functions[0][0]
               grad_acc.register_hook(self._make_hook(name, p, i))
               self._grad_accs.append(grad_acc)

   def _make_hook(self, name, p, i):
      def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
      return hook

   def sync_gradients(self):
      """ average gradients """
      if self.sync and link.get_world_size() > 1:
            for name, param in self.module.named_parameters():
               if param.requires_grad:
                  link.allreduce(param.grad.data)
      else:
            link.synchronize()

   def broadcast_params(self):
      """ broadcast model parameters """
      for name, param in self.module.state_dict().items():
            link.broadcast(param, 0)
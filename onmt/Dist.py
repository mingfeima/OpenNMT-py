import sys
import os
import torch
import torch.distributed as dist

def allReduceScalar(s):
    "AllReduces scalar accross all ranks."
    if dist.get_world_size() > 1:
        st = torch.Tensor([s])
        dist.all_reduce(st)
        return st[0]
    else:
        return s

class Dist(object):
    """
    Class for managing distibuted learning.
    """
    def __init__(self, dist_enable, backend, debug_print=False):

        self.dist_enable = dist_enable
        self.backend = backend
        self.debug_print = debug_print

        self.size = 1
        self.rank = 0

        if self.dist_enable:
            dist.init_process_group(backend='mpi')

            self.size = dist.get_world_size()
            self.rank = dist.get_rank()

            self.initPrint()

            print("Dist initialized with %s backend, world size %d" % (self.backend, self.size))

    def initPrint(self):
        "Initialize print with distibuted learning."

        if not self.debug_print:
            # print only on master rank
            if self.rank > 0:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
        else:
            # label print with info of rank/size
            old_out = sys.stdout

            class LabeledStdout:
                def __init__(self, rank, size):
                    self._r = rank
                    self._s = size
                    self.flush = sys.stdout.flush

                def write(self, x):
                    if x == '\n':
                        old_out.write(x)
                    else:
                        old_out.write('[%d/%d] %s' % (self._r, self._s, x))

            sys.stdout = LabeledStdout(self.rank, self.size)

    def broadcast(self, tensor, src=0):
        "Broadcast the tensor data from rank 0(default)."
        if self.size > 1:
            dist.broadcast(tensor, src)

    def allreduce(self,tensor):
        "AllReduces the tensor data accross all ranks."
        if self.size > 1:
            dist.all_reduce(tensor)

    def accGradParams(self, params):
        "Accumulate the gradient parameters from the different ranks."
        if self.size > 1:
            self.param_groups = list(params)

            for p in self.param_groups:
                if p.grad is None:
                    continue
                else:
                    dist.all_reduce(p.grad.data)

    def syncParams(self, params):
        "Sync parameters from main model to different ranks."
        if self.size > 1:
            self.param_groups = list(params)

            for p in self.param_groups:
                dist.broadcast(p.data, 0)

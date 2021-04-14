import torch

from .node import node

from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import PowerOfTwoParameter

class _matmul(node):
    def __init__(self, M, K, N):
        super(_matmul, self).__init__()
        self.M, self.K, self.N = M, K, N

        self.inshapes = [(M, K), (K, N)]
        self.outshape = (M, N)
        self.ref = torch.matmul

        self.verbose = False

        self.constructor()

        self.loopnest.prepare_for_codegen()
        stmt = torch._C._te.simplify(self.loopnest.root_stmt())
        self.codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [self.X, self.Y, self.Z]])

    def __repr__(self):
        return 'matmul'

    def gflops(self):
        return 2*self.M * self.N * self.K* 1e-9

    def create_search_space(self, manipulator):
        manipulator.add_parameter(PowerOfTwoParameter('xfactor', 1, self.M))
        manipulator.add_parameter(PowerOfTwoParameter('yfactor', 1, self.N))
        return manipulator

    def constructor(self):
        M, K, N = self.M, self.K, self.N
        def get_dim_args(dims):
            dim_args = []
            for dim in dims:
                dim_args.append(torch._C._te.DimArg(dim, 'i' + str(len(dim_args))))
            return dim_args
 
        (MM, KK, NN) = [torch._C._te.ExprHandle.int(x) for x in [M, K, N]]
 
        dtype = torch._C._te.Dtype.Float
        X = torch._C._te.Placeholder('X', dtype, [MM, KK])
        Y = torch._C._te.Placeholder('Y', dtype, [KK, NN])
 
        def compute(dims):
            m, n, k = dims[0], dims[1], dims[2]                                                                                                           
            return X.load([m, k]) * Y.load([k, n])
 
        Z = torch._C._te.Reduce(
             'Z',
             get_dim_args([MM, NN]),
             torch._C._te.Sum(),
             compute,
             get_dim_args([KK]),
             )
 
        self.X, self.Y, self.Z = X, Y, Z
        self.loopnest = torch._C._te.LoopNest([Z])

    def apply_config(self, cfg):
        xfactor, yfactor = cfg['xfactor'], cfg['yfactor']
        if xfactor % 2 != 0 or yfactor % 2 != 0:
            return
        
        if self.verbose is True:
            stmt = self.loopnest.root_stmt()
            print(f"Original Stmt:\n{stmt}")
        self.constructor()

        xloop, yloop, kloop = None, None, None
        loops = self.loopnest.get_loops_for(self.Z)
        if self.M>1:
            xloop = loops[0]
            if self.N>1:
                yloop = loops[1]
        if self.K>1:
            kloop = loops[-1]
        if not yloop and self.N>1:
            yloop = loops[0]

        # split xloop and yloop by xfactor, yfactor
        xinner, xouter, yinner, youter = None, None, None, None
        if yloop and yfactor > 1 and yfactor < self.N:
            youter, yinner, _ = self.loopnest.split_with_tail(yloop, yfactor)
        if xloop and xfactor > 1 and xfactor < self.M:
            xouter, xinner, _ = self.loopnest.split_with_tail(xloop, xfactor)
 
        if self.verbose is True:
            stmt = self.loopnest.root_stmt()
            print(f"After split [xfactor {xfactor}, yfactor {yfactor}]:\n{stmt}")

        # reorder to xouter, youter, kloop, xinner yinner
        loops = self.loopnest.get_loops_for(self.Z)
        if xinner and youter:
            xinner, youter = loops[1], loops[2]
            self.loopnest.reorder(xinner, youter)
            if kloop:    
                yinner, kloop = loops[3], loops[4]
                self.loopnest.reorder(kloop, yinner)
                loops = self.loopnest.get_loops_for(self.Z)
                xinner, kloop = loops[2], loops[3]
                self.loopnest.reorder(kloop, xinner)
        elif xinner and yloop:
            xinner, yloop = loops[1], loops[2]
            self.loopnest.reorder(xinner, yloop)
            if kloop:
                kloop = loops[3]
                xinner = self.loopnest.get_loops_for(self.Z)[2]
                self.loopnest.reorder(xinner, kloop)
        elif yinner and kloop:
            if xloop:
                yinner, kloop = loops[2], loops[3]
            else:
                yinner, kloop = loops[1], loops[2]
            self.loopnest.reorder(yinner, kloop)
 
        self.loopnest.prepare_for_codegen()
        stmt = torch._C._te.simplify(self.loopnest.root_stmt())
        if self.verbose is True:
            print(f"After reorder [xo, yo, k, xi, yi]:\n{stmt}")
        self.codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [self.X, self.Y, self.Z]])                

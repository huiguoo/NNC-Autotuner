import ast
import torch
from functools import partial, reduce
from operator import mul
from collections import defaultdict

from .node import node
from ..utils import divisors

from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import PowerOfTwoParameter, IntegerParameter, EnumParameter, PermutationParameter, BooleanParameter

def get_constants(conv, image):
    # These depend on conv
    out_channels = conv.out_channels
    padding = conv.padding
    dilation = conv.dilation
    kernel_size = conv.kernel_size
    stride = conv.stride
    in_channels = conv.in_channels
    groups = conv.groups
    weight = conv.weight
    stride0, stride1 = stride
    dilation0, dilation1 = dilation
    padding0, padding1 = padding
    kernel_size0, kernel_size1 = kernel_size
    assert dilation == (1, 1)  # TODO(jansel): fix dilation
    assert conv.padding_mode == "zeros"  # TODO(jansel): support other padding

    # These depend on conv + image, plan to make these dynamic in the future
    batch_size, _, *in_sizes = image.shape
    out_sizes = [ 
        (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1 
        for i, v in enumerate(in_sizes)]
    in_sizes0, in_sizes1 = in_sizes
    out_sizes0, out_sizes1 = out_sizes
    # TODO(jansel): support conv3d

    out = torch.zeros([batch_size, out_channels,
                       out_sizes0, out_sizes1], dtype=image.dtype)

    # TODO(jansel): make these dynamic
    image_stride0, image_stride1, image_stride2, image_stride3 = image.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    image_numel = image.numel()
    weight_numel = weight.numel()
    out_numel = out.numel()

    if conv.bias is not None:
        bias_stride0, = conv.bias.stride()
        bias_data_ptr = conv.bias.data_ptr()
        bias = 1 
        bias_numel = conv.bias.numel()
    else:
        bias_numel = 1 
        bias = 0 

    weight_data_ptr = conv.weight.data_ptr()
    return {k: v for k, v in locals().items() if isinstance(v, int)}

class conv2d_dims():
    def __init__(self, conv2d, image):
        constants = get_constants(conv2d, image)
        N = constants["batch_size"]
        C, Co = constants['in_channels'], constants['out_channels']
        H, W = constants['in_sizes0'], constants['in_sizes1']
        Ho, Wo = constants['out_sizes0'], constants['out_sizes1']
        R, S = constants['kernel_size0'], constants['kernel_size1']
        groups = constants['groups']

        self.dims = {}
        # batch size dim
        self.dims['n'] = {'bound': N, 'start': 0, 'end': N, 'nparts': 1, 'split_bounds':[N], 'vars':[]}
        # out_channel size dim
        self.dims['k'] = {'bound': Co//groups, 'start': 0,  'end': Co//groups,'nparts': 1, 'split_bounds':[Co//groups], 'vars':[]}
        # out height dim
        self.dims['h'] = {'bound': Ho, 'start': 0, 'end': Ho, 'nparts': 1, 'split_bounds':[Ho], 'vars':[]}
        # out width dim
        self.dims['w'] = {'bound': Wo, 'start': 0, 'end': Wo, 'nparts': 1, 'split_bounds':[Wo], 'vars':[]}
        # group size dim
        self.dims['g'] = {'bound': groups, 'start': 0, 'end': groups, 'nparts': 1, 'split_bounds':[groups], 'vars':[]}
        # in_channel size dim
        self.dims['c'] = {'bound': C//groups, 'start': 0, 'end': C//groups, 'nparts': 1, 'split_bounds':[C//groups], 'vars':[]}
        # kernel height dim
        self.dims['r'] = {'bound': R, 'start': 0, 'end': R, 'nparts': 1, 'split_bounds':[R], 'vars':[]}
        # kernel width dim
        self.dims['s'] = {'bound': S, 'start': 0, 'end': S, 'nparts': 1, 'split_bounds':[S], 'vars':[]}

        self.basic_vars()
        self.basic_looporder = ["n", "g", "k", "h", "w", "c", "r", "s"]

    def basic_vars(self):
        dint = torch._C._te.Dtype.Int
        n, g, k, h, w, c, r, s = [torch._C._te.VarHandle(s, dint) for s in ["n", "g", "k", "h", "w", "c", "r", "s"]]
        self.dims['n']['vars'] = [(n, "n")]
        self.dims['g']['vars'] = [(g, "g")]
        self.dims['k']['vars'] = [(k, "k")]
        self.dims['h']['vars'] = [(h, "h")]
        self.dims['w']['vars'] = [(w, "w")]
        self.dims['c']['vars'] = [(c, "c")]
        self.dims['r']['vars'] = [(r, "r")]
        self.dims['s']['vars'] = [(s, "s")]
        
    def update_dim(self, s):
        if s not in self.dims:
            return 
        self.dims[s]['bound'] = int(reduce(mul, self.dims[s]['split_bounds']))
        self.dims[s]['nparts'] = int(len(self.dims[s]['split_bounds']))

        self.dims[s]['end'] = self.dims[s]['start'] + self.dims[s]['bound']

    def basic_loopnest(self, stmt):
        body = stmt

        for d in self.basic_looporder[::-1]:
            var = self.dims[d]['vars'][0][0]
            start, end = torch._C._te.ExprHandle.int(self.dims[d]['start']), torch._C._te.ExprHandle.int(self.dims[d]['end'])
            body = torch._C._te.For.make(var, start, end, body)
        return body    
        

def loop(v, start, end, body):
    return torch._C._te.For.make(v, torch._C._te.ExprHandle.int(start), torch._C._te.ExprHandle.int(end), body)


class _convolution(node):
    def __init__(self, conv_args, input_shape):
        super(_convolution, self).__init__()
        self.conv2d = torch.nn.Conv2d(*conv_args)
        self.image = torch.randn(*input_shape)
        self.ref = self.conv2d

        self.verbose = True
        self.constructor()

    def __repr__(self):
        return 'conv2d'

    def gflops(self):
        conv, image = self.conv2d, self.image
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        groups = conv.groups
        padding = conv.padding
        dilation = conv.dilation
        kernel_size = list(conv.kernel_size)
        stride = conv.stride
        batch_size, _, *in_sizes = image.shape
        out_sizes = [ 
            (v + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1 
            for i, v in enumerate(in_sizes)]
        product = partial(reduce, mul)
        gflops = product([2,
                          batch_size,
                          groups,
                          out_channels // groups,
                          in_channels // groups] +
                         out_sizes +
                         kernel_size)
        if conv.bias is None:
            gflops -= product([batch_size, out_channels] + out_sizes)
        return gflops / 1000000000.0


    def create_search_space(self, manipulator):
        # slice for paddings pass : applied in default
        #manipulator.add_parameter(BooleanParameter("slicePadding"))
        
        # tile pass
        dims = self.sliced_dims()
        def tile(name, bound):
            divs = divisors(bound)
            manipulator.add_parameter(EnumParameter("tile_"+name, divs))
        for d in ['h', 'w', 'g']:
            tile(d, dims.dims[d]['bound'])

        # reorder pass
        axes_to_sort = [d for d in dims.basic_looporder]
        manipulator.add_parameter(PermutationParameter("reorder", axes_to_sort))

        # unroll pass
        manipulator.add_parameter(IntegerParameter("unroll", 0, 1))

        return manipulator

    def store_stmt(self, dims):
        def idx_var(dim):
            var, bound = dim['vars'][0][0], torch._C._te.ExprHandle.int(dim['split_bounds'][0])
            for i in range(1, dim['nparts']):
                v, b = dim['vars'][i][0], torch._C._te.ExprHandle.int(dim['split_bounds'][i])
                var, bound = var + v * bound, b * bound
            return var

        dims = dims.dims
        (n, g, k, h, w) = [idx_var(dims[x]) for x in ["n", "g", "k", "h", "w"]]
        (c, r, s) = [idx_var(dims[x]) for x in ["c", "r", "s"]]
        
        constants = self.constants
        H, W = constants['in_sizes0'], constants['in_sizes1']
        C, Co = constants['in_channels'], constants['out_channels']
        P0, P1 = constants['padding0'], constants['padding1'] 
        S0, S1 = constants['stride0'], constants['stride1']   
        groups = constants['groups']
        CC, KK, SS0, PP0, SS1, PP1 = [torch._C._te.ExprHandle.int(x) for x in [C//groups, Co//groups, S0, P0, S1, P1]]

        in_chan = c + g * CC
        out_chan = k + g * KK 
        h_image = h * SS0 + r - PP0
        w_image = w * SS1 + s - PP1 
        Pimage, Pweight, Pconv = self.Pimage, self.Pweight, self.Pconv 
        stmt = Pconv.store([n, out_chan, h, w], Pconv.load([n, out_chan, h, w]) + Pimage.load([n, in_chan, h_image, w_image]) * Pweight.load([out_chan, c, r, s]))
        cond = (h_image >= torch._C._te.ExprHandle.int(0)) & (h_image < torch._C._te.ExprHandle.int(H))  
        cond = (w_image >= torch._C._te.ExprHandle.int(0)) & (w_image < torch._C._te.ExprHandle.int(W)) & cond
        stmt = torch._C._te.Cond.make(cond, stmt, None)
        return stmt            

    def sliced_dims(self):
        constants = get_constants(self.conv2d, self.image)
        N = constants["batch_size"]
        C, Co = constants['in_channels'], constants['out_channels']
        H, W = constants['in_sizes0'], constants['in_sizes1']
        Ho, Wo = constants['out_sizes0'], constants['out_sizes1']
        R, S = constants['kernel_size0'], constants['kernel_size1']
        P0, P1 = constants['padding0'], constants['padding1'] 
        S0, S1 = constants['stride0'], constants['stride1']   
        groups = constants['groups'] 

        ## compute padding area [0, h0], [h1, Ho] for dim height
        h0, h1 = (P0-1)//S0 , -(-(P0+H-R+1)//S0)          
        ## compute padding area [0, w0], [w1, Wo] for dim width
        w0, w1 = (P1-1)//S1 , -(-(P1+W-S+1)//S1)          
 
        w0, w1 = max(w0+1, 0), min(w1, Wo)
        h0, h1 = max(h0+1, 0), min(h1, Ho)
        dims = conv2d_dims(self.conv2d, self.image)
        dims.dims['w']['start'], dims.dims['w']['split_bounds'] = w0, [w1 - w0]
        dims.update_dim('w')
        dims.dims['h']['start'], dims.dims['h']['split_bounds'] = h0, [h1 - h0]
        dims.update_dim('h')

        return dims

    def slicing_paddings(self):
        constants = get_constants(self.conv2d, self.image)
        N = constants["batch_size"]
        C, Co = constants['in_channels'], constants['out_channels']
        H, W = constants['in_sizes0'], constants['in_sizes1']
        Ho, Wo = constants['out_sizes0'], constants['out_sizes1']
        R, S = constants['kernel_size0'], constants['kernel_size1']
        P0, P1 = constants['padding0'], constants['padding1'] 
        S0, S1 = constants['stride0'], constants['stride1']   
        groups = constants['groups'] 

        ## compute padding area [0, h0], [h1, Ho] for dim height
        h0, h1 = (P0-1)//S0 , -(-(P0+H-R+1)//S0)          
        ## compute padding area [0, w0], [w1, Wo] for dim width
        w0, w1 = (P1-1)//S1 , -(-(P1+W-S+1)//S1)          
        
        loops = []
        # left padding on dim width
        if w0 >= 0 and w0 <Wo:
            dims = conv2d_dims(self.conv2d, self.image)
            dims.dims['w']['split_bounds'] = [w0+1]
            dims.update_dim('w')
        
            stmt = self.store_stmt(dims)
            loop = dims.basic_loopnest(stmt)
            loops.append(loop)
        # right padding on dim width
        if w1 >= 0 and w1 <Wo:
            dims = conv2d_dims(self.conv2d, self.image)
            dims.dims['w']['start'], dims.dims['w']['split_bounds']  = w1, [Wo - w1]
            dims.update_dim('w')
            
            stmt = self.store_stmt(dims)
            loop = dims.basic_loopnest(stmt)
            loops.append(loop)

        w0, w1 = max(w0+1, 0), min(w1, Wo)
        # head padding on dim height
        if h0 >= 0 and h0 <Ho:
            dims = conv2d_dims(self.conv2d, self.image)
            dims.dims['w']['start'], dims.dims['w']['split_bounds'] = w0, [w1 - w0]
            dims.update_dim('w')
            dims.dims['h']['split_bounds'] = [h0+1]
            dims.update_dim('h')
        
            stmt = self.store_stmt(dims)
            loop = dims.basic_loopnest(stmt)
            loops.append(loop)
        # tail padding on dim height
        if h1 >= 0 and h1 <Wo:
            dims = conv2d_dims(self.conv2d, self.image)
            dims.dims['w']['start'], dims.dims['w']['split_bounds'] = w0, [w1 - w0]
            dims.update_dim('w')
            dims.dims['h']['start'], dims.dims['h']['split_bounds'] = h1, [Ho - h1]
            dims.update_dim('h')
            
            stmt = self.store_stmt(dims)
            loop = dims.basic_loopnest(stmt)
            loops.append(loop)
        
        # main area without paddings
        h0, h1 = max(h0+1, 0), min(h1, Ho)
        dims = conv2d_dims(self.conv2d, self.image)
        dims.dims['w']['start'], dims.dims['w']['split_bounds'] = w0, [w1 - w0]
        dims.update_dim('w')
        dims.dims['h']['start'], dims.dims['h']['split_bounds'] = h0, [h1 - h0]
        dims.update_dim('h')
        stmt = self.store_stmt(dims)
        loop = dims.basic_loopnest(stmt.true_stmt())
        loops.append(loop)

        return loops, dims

    def constructor(self):
        constants = get_constants(self.conv2d, self.image)
        N = constants["batch_size"]
        C, Co = constants['in_channels'], constants['out_channels']
        H, W = constants['in_sizes0'], constants['in_sizes1']
        Ho, Wo = constants['out_sizes0'], constants['out_sizes1']
        R, S = constants['kernel_size0'], constants['kernel_size1']
        P0, P1 = constants['padding0'], constants['padding1'] 
        S0, S1 = constants['stride0'], constants['stride1']   
        groups = constants['groups'] 
        
        # construct input, weights(kernel), output
        dtype = torch._C._te.Dtype.Float 
        Pimage = torch._C._te.Placeholder("image", dtype, [torch._C._te.ExprHandle.int(x) for x in [N, C, H, W]])
        Pweight = torch._C._te.Placeholder("weight", dtype, [torch._C._te.ExprHandle.int(x) for x in [Co, C//groups, R, S]])
        Pconv = torch._C._te.Placeholder("conv", dtype, [torch._C._te.ExprHandle.int(x) for x in [N, Co, Ho, Wo]])
        
        self.Pimage, self.Pweight, self.Pconv = Pimage, Pweight, Pconv
        self.outshape = [N, Co, Ho, Wo]
        self.constants = constants
        
        # output axes/bounds
       # dint = torch._C._te.Dtype.Int
       # n, g, k, h, w = [torch._C._te.VarHandle(s, dint) for s in ["n", "g", "k", "h", "w"]]
       # NN, GG, KK, HH, WW = [torch._C._te.ExprHandle.int(x) for x in [N, groups, Co//groups, Ho, Wo]]
       # # reduction axes/bounds
       # c, r, s = [torch._C._te.VarHandle(s, dint) for s in ["c", "r", "s"]]
       # CC, RR, SS = [torch._C._te.ExprHandle.int(x) for x in [C//groups, R, S]]
       # # Strides and Paddings
       # PP0, PP1, SS0, SS1 = [torch._C._te.ExprHandle.int(x) for x in [P0, P1, S0, S1]]
       
        # build loopnest
       # in_chan = c + g * CC
       # out_chan = k + g * KK 
       # h_image = h * SS0 + r - PP0
       # w_image = w * SS1 + s - PP1 
       # stmt = Pconv.store([n, out_chan, h, w], Pconv.load([n, out_chan, h, w]) + Pimage.load([n, in_chan, h_image, w_image]) * Pweight.load([out_chan, c, r, s]))
       # cond = (h_image >= torch._C._te.ExprHandle.int(0)) & (h_image < torch._C._te.ExprHandle.int(H))  
       # cond = (w_image >= torch._C._te.ExprHandle.int(0)) & (w_image < torch._C._te.ExprHandle.int(W)) & cond
       # stmt = torch._C._te.Cond.make(cond, stmt, None)
       # for v, bound in zip([s, r, c, w, h, k, g, n], [SS, RR, CC, WW, HH, KK, GG, NN]):
       # #for v, bound in zip([s, w, c, r, h, k, g, n], [SS, WW, CC, RR, HH, KK, GG, NN]):
       #     stmt = torch._C._te.For.make(v, torch._C._te.ExprHandle.int(0), bound, stmt)

        loops, _ = self.slicing_paddings()
        stmt = torch._C._te.Stmt(loops)
        if self.verbose is True:
            print(f"Sliced Stmt:\n{stmt}")

        loopnest = torch._C._te.LoopNest(stmt, [Pconv.buf()])
        loopnest.prepare_for_codegen()
        stmt = torch._C._te.simplify(loopnest.root_stmt())
        if self.verbose is True:
            print(f"Original Stmt:\n{stmt}")
        self.loopnest = loopnest
        self.codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [Pimage, Pweight, Pconv]])

        self.op_axes = [("n", N), ("g", groups), ("k", Co//groups), ("h", Ho), ("w", Wo)]
        self.reduce_axes = [("c", C//groups), ("r", R), ("s", S)]

    def apply_config(self, cfg):
        stmt = torch._C._te.simplify(self.loopnest.root_stmt())
        if self.verbose is True:
            print(f"Original Stmt:\n{stmt}")

        # apply 'slicng' pass
        loops, dims = self.slicing_paddings()

        # apply 'tile and reorder' pass
        dint = torch._C._te.Dtype.Int
        for d in ['h', 'w', 'g']:
            name, bound = d, dims.dims[d]['bound']
            no, bo = name, bound
            print(f"tile_{name} : {cfg['tile_'+name]}")
            if cfg['tile_'+name]!=1:
                assert(bound % cfg['tile_'+name] ==0)
                ni, no = name+'i', name+'o'
                bi, bo = cfg['tile_'+name], bound // cfg['tile_'+name]
                dims.dims[name]["vars"]=[(torch._C._te.VarHandle(ni, dint), ni), (torch._C._te.VarHandle(no, dint), no)]
                dims.dims[name]["split_bounds"] = [bi, bo]
                dims.dims[name]["nparts"] = 2
                   
        stmt = self.store_stmt(dims).true_stmt()
        order = cfg['reorder']
        for axis in order:
            if axis not in set(['h','w','g']):
                continue
            if dims.dims[axis]["nparts"] == 1:
                continue
            v, start, end = dims.dims[axis]["vars"][0][0], dims.dims[axis]['start'], dims.dims[axis]['start']+dims.dims[axis]['split_bounds'][0]
            stmt = loop(v, start, end, stmt)
        
        for axis in order:
            if dims.dims[axis]["nparts"] == 1:
                v, start, end = dims.dims[axis]["vars"][0][0], dims.dims[axis]['start'], dims.dims[axis]['end']
            else:
                v, start, end = dims.dims[axis]["vars"][1][0], 0, dims.dims[axis]['split_bounds'][1]
            stmt = loop(v, start, end, stmt)

        
        block = torch._C._te.Stmt(loops[:-1]+[stmt])
        Pimage, Pweight, Pconv = self.Pimage, self.Pweight, self.Pconv 
        loopnest = torch._C._te.LoopNest(block, [Pconv.buf()])
        loopnest.prepare_for_codegen()

        # apply 'vectorize' pass
        if self.verbose is True:
            print(f"Before vectorization:\n{loopnest.root_stmt()}")
        stmt = loopnest.flatten([stmt])
        stmt = loopnest.normalize(stmt)
        loopnest.vectorize(stmt)
        
        # apply 'unrolling' pass
        if self.verbose is True:
            print(f"Before unrolling:\n{loopnest.root_stmt()}")
        loopnest.simplify()
        stmt = loopnest.get_innermost_loops_for(Pconv.buf())[-1]
        for i in range(cfg['unroll']):
            stmt = loopnest.unroll(stmt)
            stmt = loopnest.get_parent_loop(stmt)

        stmt = torch._C._te.simplify(loopnest.root_stmt())
        if self.verbose is True:
            print(f"Transformed Stmt:\n{stmt}")
        self.codegen = torch._C._te.construct_codegen('llvm', stmt, [torch._C._te.BufferArg(x) for x in [Pimage, Pweight, Pconv]])

    def gen_inputs(self):
        return [self.image]

    def run(self, inputs):
        out = torch.zeros(self.outshape)
        image, weight = inputs[0], self.conv2d.weight
        self.codegen.call([image, weight]+[out])
        return out            

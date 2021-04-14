import torch

class node():
    def __init__(self):
        super(node, self).__init__()
        self.sIdx, self.rIdx = [], []
        self.loopnest = None
        self.ref = None
        self.codegen = None
        self.inshapes, self.outshape = [], None

    def reorder(self):
        pass

    def split(self, loop, factor):
        self.loopnest.splitwithTail(loop, factor)
        
    def vectorize(self):
        pass

    def gen_inputs(self):
        inputs = []
        for inshape in self.inshapes:
            inputs.append(torch.rand(inshape))
        return inputs

    def run(self, inputs):
        out = torch.empty(self.outshape)
        self.codegen.call(inputs+[out])
        return out            

    def baseline(self, inputs):
        with torch.no_grad():
            return self.ref(*inputs)

    def validation(self):
        inputs = self.gen_inputs()
        nnc_out = self.run(inputs)
        torch_out = self.baseline(inputs)
        #print(f"nnc out=\n{nnc_out}")
        #print(f"torch out=\n{torch_out}")
        torch.testing.assert_allclose(nnc_out, torch_out)

    def gflops():
        return 0        

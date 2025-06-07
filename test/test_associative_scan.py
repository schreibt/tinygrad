import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context

def helper_test_op(shapes, torch_fxn, tinygrad_fxn, atol=1e-6, rtol=1e-3, forward_only=False, vals=None):
    torch = __import__("torch")
    if vals is None:
        if shapes is None:
            shapes = [(3, 4, 5)]
        ts = [torch.tensor(np.random.randn(*shape).astype(np.float32)) for shape in shapes]
        tst = [Tensor(x.numpy()) for x in ts]
    else:
        ts = [torch.tensor(x) for x in vals]
        tst = [Tensor(x) for x in vals]
    out = torch_fxn(*ts)
    ret = tinygrad_fxn(*tst)
    np.testing.assert_allclose(ret.numpy(), out.numpy(), atol=atol, rtol=rtol)
    if not forward_only:
        out.mean().backward()
        ret.mean().backward()
        for t, tt in zip(ts, tst):
            np.testing.assert_allclose(t.grad.numpy(), tt.grad.numpy(), atol=atol, rtol=rtol)

class TestAssociativeScan(unittest.TestCase):
    def test_associative_scan_add(self):
        helper_test_op([(10,)], lambda x: torch.cumsum(x, dim=0), lambda x: x.associative_scan(lambda a, b: a + b, axis=0))
        helper_test_op([(20,)], lambda x: torch.cumsum(x, dim=0), lambda x: x.associative_scan(lambda a, b: a + b, axis=0))
        helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=0), lambda x: x.associative_scan(lambda a, b: a + b, axis=0))
        helper_test_op([(20,30)], lambda x: torch.cumsum(x, dim=1), lambda x: x.associative_scan(lambda a, b: a + b, axis=1))

    def test_associative_scan_mul(self):
        helper_test_op([(10,)], lambda x: torch.cumprod(x, dim=0), lambda x: x.associative_scan(lambda a, b: a * b, axis=0))
        helper_test_op([(20,)], lambda x: torch.cumprod(x, dim=0), lambda x: x.associative_scan(lambda a, b: a * b, axis=0))
        helper_test_op([(20,30)], lambda x: torch.cumprod(x, dim=0), lambda x: x.associative_scan(lambda a, b: a * b, axis=0))
        helper_test_op([(20,30)], lambda x: torch.cumprod(x, dim=1), lambda x: x.associative_scan(lambda a, b: a * b, axis=1))

    def test_associative_scan_max(self):
        helper_test_op([(10,)], lambda x: torch.cummax(x, dim=0).values, lambda x: x.associative_scan(lambda a, b: a.maximum(b), axis=0))
        helper_test_op([(20,)], lambda x: torch.cummax(x, dim=0).values, lambda x: x.associative_scan(lambda a, b: a.maximum(b), axis=0))
        helper_test_op([(20,30)], lambda x: torch.cummax(x, dim=0).values, lambda x: x.associative_scan(lambda a, b: a.maximum(b), axis=0))
        helper_test_op([(20,30)], lambda x: torch.cummax(x, dim=1).values, lambda x: x.associative_scan(lambda a, b: a.maximum(b), axis=1))

    def test_associative_scan_reverse(self):
        helper_test_op([(10,)], lambda x: torch.flip(torch.cumsum(torch.flip(x, [0]), dim=0), [0]), 
                      lambda x: x.associative_scan(lambda a, b: a + b, axis=0, reverse=True))
        helper_test_op([(20,)], lambda x: torch.flip(torch.cumsum(torch.flip(x, [0]), dim=0), [0]), 
                      lambda x: x.associative_scan(lambda a, b: a + b, axis=0, reverse=True))
        helper_test_op([(20,30)], lambda x: torch.flip(torch.cumsum(torch.flip(x, [0]), dim=0), [0]), 
                      lambda x: x.associative_scan(lambda a, b: a + b, axis=0, reverse=True))
        helper_test_op([(20,30)], lambda x: torch.flip(torch.cumsum(torch.flip(x, [1]), dim=1), [1]), 
                      lambda x: x.associative_scan(lambda a, b: a + b, axis=1, reverse=True))

if __name__ == '__main__':
    unittest.main() 
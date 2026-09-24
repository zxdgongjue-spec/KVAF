import unittest
import numpy as np
from locality_sweep import plan
from random656_sweep import source_ids, payload

class LocalityTest(unittest.TestCase):
    def test_same_set_sorted(self):
        a=source_ids(np,np.random.default_rng(12),10000,1000,'random')
        b=source_ids(np,np.random.default_rng(12),10000,1000,'sorted')
        np.testing.assert_array_equal(np.sort(a),b)
    def test_contiguous(self):
        a=source_ids(np,np.random.default_rng(12),10000,1000,'contiguous')
        self.assertTrue(np.all(np.diff(a)==1))
        self.assertLess(a[-1],10000)
    def test_plan(self):
        cases=plan('locality')
        self.assertEqual(len(cases),7)
        self.assertEqual({c['blocks'] for c in cases},{8192})
        self.assertEqual({c['requests'] for c in cases},{7740})
        for c in plan('all'):
            self.assertLessEqual(c['requests'],c['working_tokens'])
            self.assertLessEqual(c['working_tokens'],c['blocks']*128)
    def test_sizes(self):
        for n in (256,656,1024,4096):
            x=payload(np,np.array([1,2]),n)
            self.assertEqual(x.shape,(2,n))
            self.assertFalse(np.array_equal(x[0],x[1]))

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from locality_sweep import plan
from random656_sweep import source_ids, payload, aligned_pool, byte_payload

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
    def test_alignment_bounds_and_payload(self):
        bases=np.array([0x1000030,0x2000010],dtype=np.uint64)
        blocks, offsets=aligned_pool(np,bases,128*656)
        self.assertTrue(np.all((bases[blocks]+offsets.astype(np.uint64))%4096==0))
        raw=payload(np,np.arange(256)).reshape(-1)
        for offset in (0,16,64,3392,3456,3840):
            local=offsets+offset
            self.assertTrue(np.all(local+656<=128*656))
            addresses=bases[blocks]+local.astype(np.uint64)
            crossed=addresses//4096 != (addresses+655)//4096
            self.assertTrue(np.all(crossed==(offset>=3456)))
            starts=blocks*128*656+local
            expected=np.stack([raw[start:start+656] for start in starts])
            np.testing.assert_array_equal(byte_payload(np,starts),expected)

    def test_sizes(self):
        for n in (256,656,1024,4096):
            x=payload(np,np.array([1,2]),n)
            self.assertEqual(x.shape,(2,n))
            self.assertFalse(np.array_equal(x[0],x[1]))

if __name__=='__main__':unittest.main()

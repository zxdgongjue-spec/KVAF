import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from random656_sweep import plan, payload, summarize


class SweepTest(unittest.TestCase):
    def test_plan(self):
        self.assertEqual(len(plan()), 8)
        self.assertEqual(len(set(plan())), 8)
        self.assertTrue(all(j <= 2 for _, _, j in plan()))

    def test_identity_payload(self):
        ids = np.array([0, 1, 128, 65536, 131071])
        data = payload(np, ids)
        self.assertEqual(data.shape, (5, 656))
        self.assertEqual(data.dtype, np.uint8)
        self.assertEqual(len({bytes(row) for row in data}), 5)

    def test_bandwidth_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(2):
                (root/f'case-{i}.json').write_text(json.dumps(dict(requests=7740, cores=30,
                    jettys=2, device_p50_us=207.466, correctness=True)))
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root)
            row = json.loads((root/'summary.json').read_text())['rows'][0]
            self.assertAlmostEqual(row['GB_s'], 24.474, places=3)
            self.assertEqual(row['runs'], 2)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from src.research.fit_i2_vnext_opening_total import (
    TRAIN, fit_half, half_predictions, predictions,
)


class VnextOpeningTotalTest(unittest.TestCase):
    def test_chronological_training_map(self):
        self.assertEqual(TRAIN[2024], (2023,))
        self.assertEqual(TRAIN[2025], (2023, 2024))
        self.assertTrue(all(max(prior)<test for test,prior in TRAIN.items()))

    def test_constant_and_total_features(self):
        data=pd.DataFrame({'total':[7.,8.,9.,10.]*10,
                           'top_p':[.25]*40,'bottom_p':[.25]*40,
                           'top_y':[0,0,1,1]*10,
                           'bottom_y':[0,1,0,1]*10,
                           'under_p':[.5625]*40})
        constant=fit_half(data,'top',False)
        total=fit_half(data,'top',True)
        self.assertEqual(constant[1],0)
        self.assertGreater(total[1],0)
        p=half_predictions(data,'top',total)
        self.assertLess(p[0],p[3])
        out=predictions(data,{'top':total,'bottom':constant})
        np.testing.assert_allclose(out['under'],(1-out['top'])*(1-out['bottom']))


if __name__=='__main__':
    unittest.main()

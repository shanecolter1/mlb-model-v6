import unittest

import numpy as np
from scipy.special import expit, logit

from src.research.validate_i2_opening_total_increment import half_prob, probability


class OpeningTotalIncrementTest(unittest.TestCase):
    def test_total_recenter_identity_and_direction(self):
        raw=np.array([.35,.50])
        unchanged=probability(raw,np.array([.45,.45]),.45)
        np.testing.assert_allclose(unchanged,raw,atol=1e-12)
        moved=probability(raw,np.array([.55,.35]),.45)
        self.assertGreater(moved[0],raw[0])
        self.assertLess(moved[1],raw[1])
        self.assertAlmostEqual(logit(moved[0])-logit(raw[0]),logit(.55)-logit(.45))

    def test_half_shift_recombines_exactly(self):
        top,bottom=half_prob(np.array([.2,.3]),np.array([.25,.35]),np.array([.3,.6]))
        np.testing.assert_allclose(1-(1-top)*(1-bottom),[.3,.6],atol=1e-12)


if __name__=='__main__':
    unittest.main()

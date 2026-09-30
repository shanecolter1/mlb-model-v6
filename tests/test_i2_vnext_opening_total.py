import unittest

import numpy as np
import pandas as pd

from src.research.fit_i2_vnext_opening_total import (
    TRAIN, fit_half, half_predictions, predictions, within_season,
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

    def test_weekly_refits_use_only_prior_dates(self):
        rows=[]
        for day in range(45):
            for game in range(12):
                top=int((day+game)%4==0)
                bottom=int((day+2*game)%5==0)
                rows.append({'date':f'2022-04-{day+1:02d}' if day<30 else
                                     f'2022-05-{day-29:02d}',
                             'gid':f'{day}-{game}','total':8.+(game%3)/2,
                             'top_p':.25,'bottom_p':.25,'under_p':.5625,
                             'top_y':top,'bottom_y':bottom,
                             'under_y':int(not(top or bottom))})
        result=within_season(pd.DataFrame(rows))
        self.assertTrue(result['weekly_fits'])
        for block in result['weekly_fits']:
            self.assertLess(block['training_last_date'],block['test_start'])
            self.assertGreaterEqual(block['training_games'],250)


if __name__=='__main__':
    unittest.main()

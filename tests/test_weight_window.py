"""
TODO: Implement tests for all the functions and properties use cases
"""
import os
import unittest

import numpy as np

from iww_gvr.weight_window import WW


class TestWW(unittest.TestCase):
    def test_properties_simple_cyl(self):
        ww = WW.read_from_ww_file('ww_simple_cyl')

        self.assertEqual(['n'], ww.particles)
        self.assertEqual('cyl', ww.coordinates)
        np.testing.assert_array_equal(np.array([0, 7.5, 15]), ww.vector_i)
        np.testing.assert_array_almost_equal(np.array([0., 5.333333, 10.666667, 16.]), ww.vector_j)
        np.testing.assert_array_almost_equal(np.array([0., 1.]), ww.vector_k)
        self.assertEqual({'n': [100]}, ww.energies)
        expected_values = {'n': {100.0: np.array([[[0.5, 0.10463],
                                                   [0.52965, 0.084479],
                                                   [0.14258, 0.03275]]])}}
        for particle in expected_values.keys():
            for energy in expected_values[particle].keys():
                np.testing.assert_array_equal(expected_values[particle][energy], ww.values[particle][energy])

    def test_properties_complex_cart(self):
        ww = WW.read_from_ww_file('ww_complex_cart')

        self.assertEqual(['n', 'p'], ww.particles)
        self.assertEqual('cart', ww.coordinates)
        expected_energies = {'n': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 100.0], 'p': [1.2, 2.3]}
        self.assertEqual(expected_energies, ww.energies)
        np.testing.assert_array_almost_equal(np.array([-15., -4.95, 5.1, 8.65, 12.2, 12.75, 13.3]),
                                             ww.vector_i)
        expected_values = {'n': {1.1: np.array([[[9.4638e-02, 7.9364e-01, 7.9103e-01, 9.6254e-02, 4.0716e-02,
                                                  9.8324e-03, 6.5408e-03],
                                                 [6.9929e-01, 4.2589e+01, 0.0000e+00, 0.0000e+00, 2.9634e-01,
                                                  1.9691e-01, 3.3515e-02],
                                                 [2.6938e-01, 5.1002e-01, 0.0000e+00, 2.0829e-01, 2.8583e-02,
                                                  1.2900e-02, 1.1203e-02]]]),
                                 2.2: np.array([[[6.3137e-02, 5.6489e-01, 7.2160e-02, 0.0000e+00, 2.0076e-02,
                                                  6.6921e-03, 6.6921e-03],
                                                 [3.5328e-01, 2.9492e+01, 1.0601e+00, 0.0000e+00, 2.0407e-01,
                                                  3.4011e-02, 1.1337e-02],
                                                 [0.0000e+00, 2.8219e-01, 0.0000e+00, 1.5279e-02, 1.1298e-02,
                                                  5.6489e-03, 5.6489e-03]]]),
                                 3.3: np.array([[[0.046149, 0.38326, 0., 0.021803, 0.024316,
                                                  0.007493, 0.007493],
                                                 [0.31873, 0., 0., 0.25534, 0.096616,
                                                  0., 0.01232],
                                                 [0.12302, 0.29319, 0.058485, 0.0048739, 0.016242,
                                                  0., 0.]]]),
                                 4.4: np.array([[[0.042816, 0.43559, 0.080572, 0.025398, 0.016225,
                                                  0.0064901, 0.0064901],
                                                 [0.26117, 5.8358, 0., 0., 0.027459,
                                                  0.0068648, 0.0068648],
                                                 [0.028517, 0.16653, 0.026879, 0.0067176, 0.013435,
                                                  0., 0.]]]),
                                 5.5: np.array([[[0.043231, 0.98546, 0.16724, 0.0301, 0., 0.,
                                                  0.],
                                                 [0.81101, 5.228, 0., 0., 0.12812, 0.,
                                                  0.],
                                                 [0.023126, 0.09525, 0.029612, 0., 0., 0.,
                                                  0.]]]),
                                 6.6: np.array([[[0.044074, 0.61927, 0.029518, 0.01461, 0.0074036,
                                                  0.0074674, 0.0074214],
                                                 [0.1588, 3.6451, 0.30551, 0.094602, 0.013649,
                                                  0.010599, 0.0070662],
                                                 [0.023673, 0.085316, 0.019117, 0.026867, 0.0067166,
                                                  0., 0.]]]),
                                 100.0: np.array([[[0.020299, 0.083632, 0.019789, 0.015584, 0.012812,
                                                    0.0075478, 0.0075485],
                                                   [0.079784, 0.5, 0.065101, 0.033571, 0.013813,
                                                    0.0078807, 0.007273],
                                                   [0.026228, 0.055085, 0.022905, 0.031785, 0.010889,
                                                    0.007543, 0.007543]]])},
                           'p': {1.2: np.array([[[0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.]]]),
                                 2.3: np.array([[[0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.]]])}}
        for particle in expected_values.keys():
            for energy in expected_values[particle].keys():
                np.testing.assert_array_almost_equal(expected_values[particle][energy], ww.values[particle][energy])

    def test_write_simple_cyl(self):
        ww = WW.read_from_ww_file('ww_simple_cyl')
        ww.write_ww_file('delete.txt')
        expected_text = """         1         1         1        16                     Generated with iww_gvr 
         1
   2.0000       3.0000       1.0000       0.0000       0.0000      -5.0000    
   1.0000       1.0000       1.0000       0.0000       0.0000       11.000    
   15.000       0.0000      -5.0000       2.0000    
   0.0000       2.0000       15.000       1.0000    
   0.0000       3.0000       16.000       1.0000    
   0.0000       1.0000       1.0000       1.0000    
   100.00    
  5.00000E-01  1.04630E-01  5.29650E-01  8.44790E-02  1.42580E-01  3.27500E-02  
"""
        with open('delete.txt', 'r') as infile:
            actual_text = infile.read()
        os.remove('delete.txt')
        self.assertEqual(expected_text, actual_text)

    def test_vector_i(self):
        ww = WW.read_from_ww_file('ww_complex_cart')
        expected_vector = np.array([-15., -4.95, 5.1, 8.65, 12.2, 12.75, 13.3])
        actual_vector = ww.vector_i
        np.testing.assert_array_almost_equal(expected_vector, actual_vector)

    def test_info_simple_cyl(self):
        ww = WW.read_from_ww_file('ww_simple_cyl')
        expected_text = """ww_simple_cyl weight window:
               -----From----- -----To----- ---No. Bins---
 I -->             0.00         15.00         2
 J -->             0.00         16.00         3
 K -->             0.00         1.00         1

 The mesh coordinates are cylindrical

 The weight window contains 1 particle/s of 6 voxels.

 Energy bins of particle n:
 [100.0]"""
        actual_text = ww.info
        self.assertEqual(expected_text, actual_text)

    def test_calculate_array_ratio_cart(self):
        array = np.array([[[1, 1],
                           [1, 1]],
                          [[2, 1],
                           [1, 0.5]],
                          ])
        expected_ratio = np.array([[[2, 1],
                                    [1, 2]],
                                   [[2, 2],
                                    [2, 2]],
                                   ])
        actual_ratio = WW._calculate_array_ratio(array)
        np.testing.assert_array_almost_equal(expected_ratio, actual_ratio)

    def test_calculate_ratios(self):
        """TODO: implement the testing of cyl last and first angular indexes are neighbours"""
        ww = WW.read_from_ww_file('ww_simple_cyl')
        expected_ratio = {'n': {100.0: np.array([[[4.77874415, 4.77874415],
                                                  [6.26960546, 6.26960546],
                                                  [4.35358779, 4.35358779]]])}}
        ww.calculate_ratios()
        actual_ratio = ww.ratios
        for particle in actual_ratio.keys():
            for energy in actual_ratio[particle].keys():
                np.testing.assert_array_almost_equal(expected_ratio[particle][energy], actual_ratio[particle][energy])

    # TODO: implement once the ratio cyl neighbour is solved
    # def test_info_analyse_simple_cyl(self):
    #     ww = WW.read_from_ww_file('ww_simple_cyl')
    #     expected_text = ""
    #     actual_text = ww.info_analyse
    #     self.assertEqual(expected_text, actual_text)

    def test_info_analyse_complex_cart(self):
        ww = WW.read_from_ww_file('ww_complex_cart')
        expected_text = """The following weight window has been analysed: ww_complex_cart
---------------Par. N---------
Min Value       : 0.00000
Max Value       : 42.58900
Max Ratio       : 104.51114
No.Bins > 0 [%] : 79.59
Neg. Value      : NO
 
---------------Par. P---------
Min Value       : 0.00000
Max Value       : 0.00000
Max Ratio       : 0.00000
No.Bins > 0 [%] : 0.00
Neg. Value      : NO
 
Coordinates     : cart
 
Voxel dimensions [x, y, z]: 10.05, 10.33, 35.00
Voxel volume: 3634.75 cm3
"""
        actual_text = ww.info_analyse
        self.assertEqual(expected_text, actual_text)


if __name__ == '__main__':
    unittest.main()

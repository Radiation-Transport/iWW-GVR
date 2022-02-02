"""
The MCNP WW writer is inconsistent, for some reason sometimes write WW values as 0.50000 and sometimes write
 0.934E1... For that reason the testing of the writer is done on custom strings with a consistent data formatting.
"""
import os
import unittest

from iww_gvr.ww_parser import read_ww, write_ww
from iww_gvr.utils.resource import filename_resolver

find_data_file = filename_resolver("tests")


class TestParser(unittest.TestCase):
    def test_parse_simple_cart(self):
        expected_dict = {'if_': 1,
                         'iv': 1,
                         'ni': 1,
                         'nr': 10,
                         'probid': '06/03/21 14:52:07',
                         'ne': [1],
                         'nfx': 2.0,
                         'nfy': 3.0,
                         'nfz': 1.0,
                         'origin': [-15.0, -15.0, -15.0],
                         'ncx': 1.0,
                         'ncy': 1.0,
                         'ncz': 1.0,
                         'b2_vector_i': [-15.0, 2.0, 15.0, 1.0],
                         'b2_vector_j': [-15.0, 3.0, 16.0, 1.0],
                         'b2_vector_k': [-15.0, 1.0, 20.0, 1.0],
                         'energies': [[100.0]],
                         'values': [[0.11576, 0.093197, 0.67316, 0.5, 0.099821, 0.0898]]}
        actual_dict = read_ww(find_data_file('data/ww_simple_cart'))
        self.assertDictEqual(expected_dict, actual_dict)

    def test_parse_simple_cyl(self):
        expected_dict = {'if_': 1,
                         'iv': 1,
                         'ni': 1,
                         'nr': 16,
                         'probid': '06/04/21 17:17:28',
                         'ne': [1],
                         'nfx': 2.0,
                         'nfy': 3.0,
                         'nfz': 1.0,
                         'origin': [0.0, 0.0, -5.0],
                         'ncx': 1.0,
                         'ncy': 1.0,
                         'ncz': 1.0,
                         'b2_vector_i': [0.0, 2.0, 15.0, 1.0],
                         'b2_vector_j': [0.0, 3.0, 16.0, 1.0],
                         'b2_vector_k': [0.0, 1.0, 1.0, 1.0],
                         'energies': [[100.0]],
                         'values': [[0.5, 0.10463, 0.52965, 0.084479, 0.14258, 0.03275]],
                         'director_1': [0.0, 0.0, 11.0],
                         'director_2': [15.0, 0.0, -5.0]}
        actual_dict = read_ww(find_data_file('data/ww_simple_cyl'))
        self.assertDictEqual(expected_dict, actual_dict)

    def test_parse_complex_cart(self):
        expected_dict = {'if_': 1,
                         'iv': 1,
                         'ni': 2,
                         'nr': 10,
                         'probid': '06/05/21 11:33:41',
                         'ne': [7, 2],
                         'nfx': 7.0,
                         'nfy': 3.0,
                         'nfz': 1.0,
                         'origin': [-15.0, -15.0, -15.0],
                         'ncx': 3.0,
                         'ncy': 1.0,
                         'ncz': 1.0,
                         'b2_vector_i': [-15.0, 2.0, 5.1, 1.0, 3.0, 12.2, 1.0, 2.0, 13.3, 1.0],
                         'b2_vector_j': [-15.0, 3.0, 16.0, 1.0],
                         'b2_vector_k': [-15.0, 1.0, 20.0, 1.0],
                         'energies': [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 100.0], [1.2, 2.3]],
                         'values': [[0.094638, 0.79364, 0.79103, 0.096254, 0.040716, 0.0098324, 0.0065408, 0.69929,
                                     42.589, 0.0, 0.0, 0.29634, 0.19691, 0.033515, 0.26938, 0.51002, 0.0, 0.20829,
                                     0.028583, 0.0129, 0.011203, 0.063137, 0.56489, 0.07216, 0.0, 0.020076, 0.0066921,
                                     0.0066921, 0.35328, 29.492, 1.0601, 0.0, 0.20407, 0.034011, 0.011337, 0.0, 0.28219,
                                     0.0, 0.015279, 0.011298, 0.0056489, 0.0056489, 0.046149, 0.38326, 0.0, 0.021803,
                                     0.024316, 0.007493, 0.007493, 0.31873, 0.0, 0.0, 0.25534, 0.096616, 0.0, 0.01232,
                                     0.12302, 0.29319, 0.058485, 0.0048739, 0.016242, 0.0, 0.0, 0.042816, 0.43559,
                                     0.080572, 0.025398, 0.016225, 0.0064901, 0.0064901, 0.26117, 5.8358, 0.0, 0.0,
                                     0.027459, 0.0068648, 0.0068648, 0.028517, 0.16653, 0.026879, 0.0067176, 0.013435,
                                     0.0, 0.0, 0.043231, 0.98546, 0.16724, 0.0301, 0.0, 0.0, 0.0, 0.81101, 5.228, 0.0,
                                     0.0, 0.12812, 0.0, 0.0, 0.023126, 0.09525, 0.029612, 0.0, 0.0, 0.0, 0.0, 0.044074,
                                     0.61927, 0.029518, 0.01461, 0.0074036, 0.0074674, 0.0074214, 0.1588, 3.6451,
                                     0.30551, 0.094602, 0.013649, 0.010599, 0.0070662, 0.023673, 0.085316, 0.019117,
                                     0.026867, 0.0067166, 0.0, 0.0, 0.020299, 0.083632, 0.019789, 0.015584, 0.012812,
                                     0.0075478, 0.0075485, 0.079784, 0.5, 0.065101, 0.033571, 0.013813, 0.0078807,
                                     0.007273, 0.026228, 0.055085, 0.022905, 0.031785, 0.010889, 0.007543, 0.007543],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}
        actual_dict = read_ww(find_data_file('data/ww_complex_cart'))
        self.assertDictEqual(expected_dict, actual_dict)

    def test_write_simple_cart(self):
        data = {'if_': 1,
                'iv': 1,
                'ni': 1,
                'nr': 10,
                'probid': '06/03/21 14:52:07',
                'ne': [1],
                'nfx': 2.0,
                'nfy': 3.0,
                'nfz': 1.0,
                'origin': [-15.0, -15.0, -15.0],
                'ncx': 1.0,
                'ncy': 1.0,
                'ncz': 1.0,
                'b2_vector_i': [-15.0, 2.0, 15.0, 1.0],
                'b2_vector_j': [-15.0, 3.0, 16.0, 1.0],
                'b2_vector_k': [-15.0, 1.0, 20.0, 1.0],
                'energies': [[100.0]],
                'values': [[0.11576, 0.093197, 0.67316, 0.5, 0.099821, 0.0898]]}
        write_ww('delete.txt', data)
        expected_text = """         1         1         1        10                     06/03/21 14:52:07 
         1
   2.0000       3.0000       1.0000      -15.000      -15.000      -15.000    
   1.0000       1.0000       1.0000       1.0000    
  -15.000       2.0000       15.000       1.0000    
  -15.000       3.0000       16.000       1.0000    
  -15.000       1.0000       20.000       1.0000    
   100.00    
  1.15760E-01  9.31970E-02  6.73160E-01  5.00000E-01  9.98210E-02  8.98000E-02  
"""
        with open('delete.txt', 'r') as infile:
            actual_text = infile.read()
        os.remove('delete.txt')
        self.assertEqual(expected_text, actual_text)

    def test_write_simple_cyl(self):
        data = {'if_': 1,
                'iv': 1,
                'ni': 1,
                'nr': 16,
                'probid': 'Generated with iww_gvr',
                'ne': [1],
                'nfx': 2.0,
                'nfy': 3.0,
                'nfz': 1.0,
                'origin': [0.0, 0.0, -5.0],
                'ncx': 1.0,
                'ncy': 1.0,
                'ncz': 1.0,
                'b2_vector_i': [0.0, 2.0, 15.0, 1.0],
                'b2_vector_j': [0.0, 3.0, 16.0, 1.0],
                'b2_vector_k': [0.0, 1.0, 1.0, 1.0],
                'energies': [[100.0]],
                'values': [[0.5, 0.10463, 0.52965, 0.084479, 0.14258, 0.03275]],
                'director_1': [0.0, 0.0, 11.0],
                'director_2': [15.0, 0.0, -5.0]
                }
        write_ww('delete.txt', data)
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


if __name__ == '__main__':
    unittest.main()

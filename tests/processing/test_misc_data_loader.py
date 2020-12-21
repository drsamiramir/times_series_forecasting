import unittest
import os
from src.processing.misc_data_loader import *



class DataLoaderTest(unittest.TestCase):
    def test_csv_loader(self):
        load_cvs_file("daily-total-female-births.csv")
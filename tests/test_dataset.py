from unittest import TestCase
import numpy as np
import datetime as dt
from collections import namedtuple

from ds_workflow.dataset.dataset import Dataset


class TestArguments(TestCase):
    """Test Dataset arguments, ensuring they raise the correct exceptions when incorrect."""
    def test_base_cases(self):
        try:
            ds1 = Dataset(path='data/test-data.csv', labels=None, is_derived=False)
            ds2 = Dataset(path='data/titanic.csv', labels=None, is_derived=False)
            ds3 = Dataset(path='data/wine-reviews.csv', labels=None, is_derived=False)
        except FileNotFoundError:
            self.fail(f"Base cases are not loading the example dataset paths correctly.")

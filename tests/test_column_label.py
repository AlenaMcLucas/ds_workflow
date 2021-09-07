from unittest import TestCase
import numpy as np

from ds_workflow.dataset.column_label import ColumnLabel


class TestValues(TestCase):

    def test_base_case(self):
        base_case = ColumnLabel('categorical', 'str', True)
        result = str(base_case)
        expected = 'category: categorical, type: str, is_active: True'
        self.assertEqual(result, expected)

    def test_category_type(self):
        with self.assertRaises(ValueError):
            ColumnLabel(category=0.1, type='str', is_active=True)

    def test_type_value(self):
        with self.assertRaises(ValueError):
            ColumnLabel(category='numeric', type='potato', is_active=False)

    def test_is_active_type(self):
        with self.assertRaises(ValueError):
            ColumnLabel(category='date/time', type='str', is_active=np.nan)

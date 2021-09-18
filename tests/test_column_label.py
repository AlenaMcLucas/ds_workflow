from unittest import TestCase
import numpy as np
import datetime as dt
from collections import namedtuple
from itertools import chain

from ds_workflow.dataset.column_label import ColumnLabel, CAST_MAP, COLUMN_VALUES


class TestArguments(TestCase):
    """Test ColumnLabel arguments, ensuring they raise the correct exceptions when incorrect."""
    def test_base_case(self):
        base_case = ColumnLabel(category='categorical', type='str', is_active=True)
        result = str(base_case)
        expected = 'category: categorical, type: str, is_active: True'
        self.assertEqual(result, expected)

    def test_category_type(self):
        with self.assertRaises(TypeError):
            ColumnLabel(category=0.1, type='str', is_active=True)

    def test_type_value(self):
        with self.assertRaises(ValueError):
            ColumnLabel(category='numeric', type='potato', is_active=False)

    def test_is_active_type(self):
        with self.assertRaises(TypeError):
            ColumnLabel(category='datetime', type='str', is_active=np.nan)

    def test_category_type_match(self):
        with self.assertRaises(ValueError):
            ColumnLabel(category='categorical', type='datetime', is_active=False)


class TestCastValues(TestCase):
    """Check the developer's own work on the CAST_MAP variable, ensuring all values are acceptable."""
    def test_categories(self):
        coded_categories = set(chain.from_iterable([pairs['category'] for pairs in CAST_MAP.values()]))
        accepted_categories = set(COLUMN_VALUES['category'].values)
        coded_categories_diff = coded_categories.difference(accepted_categories)
        self.assertEqual(len(coded_categories_diff), 0,
                         msg=f"{coded_categories_diff} are not accepted categories.")

    def test_types(self):
        all_coded_types = chain.from_iterable([pairs['type'] for pairs in CAST_MAP.values()])
        coded_types = set([t.__name__ for t in all_coded_types])
        accepted_types = set(COLUMN_VALUES['type'].values)
        coded_types_diff = coded_types.difference(accepted_types)
        self.assertEqual(len(coded_types_diff), 0,
                         msg=f"{coded_types_diff} are not accepted types.")

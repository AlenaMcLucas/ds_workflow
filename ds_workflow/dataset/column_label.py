import numpy as np
import datetime as dt
from collections import namedtuple


AcceptableAttributes = namedtuple('Attribute', ['type', 'values'])

COLUMN_VALUES = {
    'category': AcceptableAttributes(str, ['categorical', 'numeric', 'text', 'datetime']),
    'type': AcceptableAttributes(str, ['int', 'float', 'str', 'datetime']),
    'is_active': AcceptableAttributes(bool, [True, False])
}

NAME_TYPES_MAP = {'str': str, 'float': float, 'int': int, 'datetime': dt.datetime}

CATEGORY_MAP = {object: 'categorical', np.object_: 'categorical', str: 'categorical', int: 'numeric',
                np.int64: 'numeric', float: 'numeric', np.float64: 'numeric', dt.datetime: 'datetime'}

# first element in 'category' will be default after data type successfully casted
CAST_MAP = {
    int: {
        'category': ['numeric', 'categorical'],
        'type': [float, dt.datetime]
    },
    np.int64: {
        'category': ['numeric', 'categorical'],
        'type': [float, dt.datetime]
    },
    float: {
        'category': ['numeric'],
        'type': [int]
    },
    np.float64: {
        'category': ['numeric'],
        'type': [int]
    },
    str: {
        'category': ['categorical', 'text'],
        'type': [int, float, dt.datetime]
    },
    object: {
        'category': ['categorical', 'text'],
        'type': [int, float, dt.datetime]
    },
    np.object_: {
        'category': ['categorical', 'text'],
        'type': [int, float, dt.datetime]
    },
    dt.datetime: {
        'category': ['datetime'],
        'type': [int]
    }
}


class ColumnLabel:
    """Labels for Dataset columns, also used by Statistic and Algorithm.

    Attributes
    ----------
    category : str
        Data category, can be 'categorical', 'numeric', 'text', or 'datetime'.
    type : str
        Data type, can be 'int', 'float', 'str', or 'datetime'.
    is_active : bool
        If the column is enabled by the user.

    Methods
    -------
    category_type_match()
        Checks that the column's category and type are an acceptable match. Raises a ValueError if not.
    """
    def __init__(self, category, type, is_active):
        self.category = category
        self.type = type
        self.is_active = is_active
        self.category_type_match()

    def __repr__(self):
        return f"category: {self.category}, type: {self.type}, is_active: {self.is_active}"

    def __setattr__(self, key, value):
        acceptable_type = COLUMN_VALUES[key].type
        if not isinstance(value, acceptable_type):
            raise TypeError(f"'{key}' must be of type '{acceptable_type.__name__}'.")

        acceptable_values = COLUMN_VALUES[key].values
        if value not in acceptable_values:
            raise ValueError(f"'{value}' must be an accepted value: {acceptable_values}.")

        self.__dict__[key] = value

    def category_type_match(self):
        categories_from_type = CAST_MAP[NAME_TYPES_MAP[self.type]]['category']
        if self.category not in categories_from_type:
            raise ValueError(f"'{self.category}' is not an acceptable category for data type '{self.type}'.")

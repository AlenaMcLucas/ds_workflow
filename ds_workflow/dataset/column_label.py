from collections import namedtuple


AcceptableAttributes = namedtuple('Attribute', ['type', 'values'])

COLUMN_VALUES = {
    'category': AcceptableAttributes(str, ['categorical', 'numeric', 'date/time']),
    'type': AcceptableAttributes(str, ['int', 'float', 'str', 'text', 'date', 'time', 'datetime']),
    'is_active': AcceptableAttributes(bool, [True, False])
}


class ColumnLabel:
    """Labels for Dataset columns, also used by Statistic and Algorithm.

    Attributes
    ----------
    category : str
        Data category, can be 'categorical', 'numeric', or 'date/time'.
    type : str
        Data type, can be 'int', 'float', 'str', 'text', 'date', 'time', or 'datetime'.
    is_active : bool
        If the column is enabled by the user.
    """
    def __init__(self, category, type, is_active):
        self.category = category
        self.type = type
        self.is_active = is_active

    def __repr__(self):
        return f"category: {self.category}, type: {self.type}, is_active: {self.is_active}"

    def __setattr__(self, key, value):
        acceptable_type = COLUMN_VALUES[key].type
        if not isinstance(value, acceptable_type):
            raise ValueError(f"'{key}' must be of type '{acceptable_type.__name__}'.")

        acceptable_values = COLUMN_VALUES[key].values
        if value not in acceptable_values:
            raise ValueError(f"'{value}' must be an accepted value: {acceptable_values}.")

        self.__dict__[key] = value

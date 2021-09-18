import json
import math
import random
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path

from ds_workflow.dataset.utils import get_current_path
from ds_workflow.dataset.column_label import ColumnLabel, CATEGORY_MAP, NAME_TYPES_MAP, CAST_MAP, COLUMN_VALUES


class DataTypeNotFound(TypeError):
    def __init__(self, error):
        super().__init__(f"A data type for column '{error}' could not be found.")


class Dataset:
    """A Dataset manages a data's labels, transformations, and train-test split. It interfaces with Statistic
    and Model, not in use yet.

    Parameters
    ----------
    path : str
        Path to data file.
    labels : dict or None
        Not in use uet. Dictionary of dictionaries with column name (str), dictionary pairs. The second dictionary will
        use keys 'category', 'type', and 'is_active'. Values passed will be validated and stored by
        ColumnLabel. If no dictionary is passed, these values are automatically generated.
    is_derived : bool
        Not in use yet.

    Attributes
    ----------
    path : str
        Path to data file.
    df : pandas.DataFrame
        Stores data as a pandas dataframe.
    split_indices : dict of lists
        Keys are 'train', 'test', and optionally 'validate' (str). Values are lists of indices.
    is_split : bool
        True if has been train-test split, False otherwise.
    is_derived : bool
        Not in use yet.
    labels : dict of ColumnLabels (namedtuples)
        Keys are column names (str), values are ColumnLabels, containing category (str), type (type), and is_active
        (bool).
    target : str, default None
        Target variable to predict, if supervised learning.
    """
    def __init__(self, path, labels=None, is_derived=False):
        self.path = path
        self.df = pd.read_csv(f"{get_current_path()}/{self.path}")
        self.split_indices = {}
        self.is_split = False
        self.is_derived = is_derived

        # if no labels passed, auto-generate ColumnLabels
        if labels is None:
            self.labels = {col_name: self.auto_assign(col_name) for col_name in self.df.columns}
        elif isinstance(labels, dict):
            pass

        self.target = None

    def __repr__(self):
        _repr = f"Path: {self.path}\nIs split: {self.is_split}\nIs derived: {self.is_derived}\n\n"
        for col_name, col_label in self.labels.items():
            _repr += f"{col_name} - {col_label}\n"
        return _repr

    def validate_column_name(self, col_name):
        """Decorator that checks for a column name in Dataset before doing some operation."""
        if col_name not in self.df.columns:
            raise ValueError(f"'{col_name}' was not found in the Dataset.")

    def auto_type(self, col_name):
        """Searches passed column's values for its data type."""
        for value in self.df[col_name]:
            value_type = type(value)
            if value_type not in (None, np.nan):
                return value_type
        else:
            raise DataTypeNotFound(col_name)

    def auto_category(self, col_type, col_name):
        """Automatically assigns a category to a column based on its data type. If it is a string and the length of the
        first value is greater than or equal to 20, then it categorizes the column as 'text' instead of 'categorical'.
        This method uses CATEGORY_MAP from the column_label module.

        Parameters
        ----------
        col_type : type
        col_name : str

        Returns
        -------
        str
            Column's new category.
        """
        if col_type == str:
            for value in self.df[col_name]:
                if value not in (None, np.nan) and len(value) >= 20:
                    return 'text'
                else:
                    return 'categorical'
        else:
            return CATEGORY_MAP[col_type]

    def auto_assign(self, col_name):
        """Automatically assign a column a category, type, and is_active.

        Returns
        -------
        ColumnLabel
            Column's new label, including its category, type, and it sets 'is_active' to True.
        """
        self.validate_column_name(col_name)
        auto_type = self.auto_type(col_name)
        auto_category = self.auto_category(auto_type, col_name)
        return ColumnLabel(category=auto_category, type=auto_type.__name__, is_active=True)

    def set_target(self, col_name):
        """Sets target to passed column."""
        self.validate_column_name(col_name)
        self.target = col_name

    def cast_active(self, col_name, is_active):
        """Sets active/inactive status for passed column.

        Parameters
        ----------
        col_name : str
        is_active : bool
        """
        self.validate_column_name(col_name)
        self.labels[col_name].is_active = is_active

    def cast_type(self, col_name, cast_type, format=None):
        """Cast data type of a column.

        Parameters
        ----------
        col_name : str
        cast_type : type
            Cast type desired.
        format : str, default None
            The strftime to parse time, eg "%d/%m/%Y". See strftime documentation for more information on choices.
        """
        self.validate_column_name(col_name)

        from_type = NAME_TYPES_MAP[self.labels[col_name].type]
        valid_types = CAST_MAP[from_type]['type']

        if cast_type == dt.datetime:
            try:
                self.df[col_name] = pd.to_datetime(self.df[col_name], format=format)
            except ValueError:
                raise ValueError(f"'{col_name}' could not be converted to datetime. Check 'format' and try again.")

        elif cast_type in valid_types:
            try:
                self.df[col_name] = self.df[col_name].astype(cast_type)
            except ValueError:
                if from_type == str:
                    raise ValueError('String contains non-numeric values, parse before casting.')
                else:
                    raise ValueError('In pandas, cannot convert float NaN to integer.')

        else:
            raise ValueError(f"You want to cast '{col_name}' to {cast_type.__name__}, "
                             f"but its current type is {from_type.__name__} "
                             f"and that's not an acceptable type cast in CAST_MAP.")

        self.labels[col_name].type = cast_type.__name__
        default_category = CAST_MAP[cast_type]['category'][0]
        self.labels[col_name].category = default_category
        self.labels[col_name].category_type_match()

    def cast_category(self, col_name, cast_category):
        """Cast data category of a column.

        Parameters
        ----------
        col_name : str
        cast_category : str
            Cast category desired.
        """
        self.validate_column_name(col_name)
        self.labels[col_name].category = cast_category
        self.labels[col_name].category_type_match()

    def drop_rows(self, to_drop):
        """Drop rows by index.

        Parameters
        ----------
        to_drop : list
            Indices to drop from the Dataset.
        """
        self.df.drop(index=to_drop, axis=0, inplace=True)

    def drop_null_rows(self, col_name):
        """Drop null rows for a particular column."""
        self.validate_column_name(col_name)
        self.drop_rows(self.df[col_name][self.df[col_name].isna()].index.tolist())

    def drop_columns(self, to_drop):
        """Drop columns by label.

        Parameters
        ----------
        to_drop : list
            Column names (str) to drop from the Dataset.
        """
        for col_name in to_drop:
            self.validate_column_name(col_name)
            self.df.drop(labels=col_name, axis=1, inplace=True)
            del self.labels[col_name]

    def add_columns(self, to_add):
        """Adds a DataFrame or Series to Dataset.

        Parameters
        ----------
        to_add : pandas.Dataframe or pandas.Series
        """
        if isinstance(to_add, pd.DataFrame):
            self.df = pd.concat([self.df, to_add], axis=1)
            for col_name in to_add.columns:
                self.labels[col_name] = self.auto_assign(col_name)

        elif isinstance(to_add, pd.Series):
            if to_add.name not in self.df.columns:
                self.labels[to_add.name] = self.auto_assign(to_add.name)
            else:
                raise ValueError(f"A column with the name '{to_add.name}' already exists in the Dataset.")

    def to_dummies(self, col_name, drop_categorical=False, drop_first=False, prefix=None, prefix_sep=None):
        """Create dummy variables.

        Parameters
        ----------
        col_name : str
        drop_categorical : bool, default False
            If true, drop the original categorical variable you are converting to dummy variables.
        drop_first : bool, default False
            If true, drop one of the dummy variables. Useful for certain models.
        prefix : str, default None
            Prefix for dummy names. When None is passed, prefix will be the column name.
        prefix_sep : str, default None
            Prefix separator for dummy names. When None is passed, separator will be '_'.
        """
        self.validate_column_name(col_name)
        df_to_add = pd.get_dummies(self.df[col_name], prefix=prefix if prefix else col_name,
                                   prefix_sep=prefix_sep if prefix_sep else '_',
                                   drop_first=drop_first, dtype=np.int64)
        self.add_columns(df_to_add)
        if drop_categorical:
            self.drop_columns([col_name])

    def handle_nulls(self, col_name, strategy):
        """Resolve null values in a given column.

        Parameters
        ----------
        col_name : str
        strategy : str
            Current options are dropping null rows ('drop_rows'), dropping that column entirely ('drop_column'),
            or filling null values with the column average ('fill_average').
        """
        if strategy == 'drop_rows':
            self.drop_null_rows(col_name)
        if strategy == 'drop_column':
            self.drop_columns([col_name])
        if strategy == 'fill_average':
            # later replace with Statistic
            average = self.df[col_name].mean(skipna=True)
            self.df[col_name].fillna(value=average, inplace=True)
        else:
            raise ValueError(f"'{strategy}' is not an accepted value for 'strategy'.")

    def split(self, test, validate=0.0, seed=0):
        """Split the data, either train / test or train / validate / test.

        Parameters
        ----------
        test : float
            Percentage of data to allocate to the test set.
        validate : float, default 0.0
            Percentage of data to allocate to the validation set.
        seed : int, default 0
            Set random seed to ensure the same results when splitting the data into sets.
        """
        random.seed(seed)
        size = len(self.df.index)
        indices = list(range(size))
        random.shuffle(indices)
        test_size, val_size = int(test * size), int(validate * size)

        self.split_indices['test'] = indices[:test_size]
        if validate > 0:
            self.split_indices['validate'] = indices[test_size:test_size+val_size]
        self.split_indices['train'] = indices[test_size+val_size:]
        self.is_split = True


if __name__ == '__main__':
    ds1 = Dataset(path='data/titanic.csv', labels=None, is_derived=False)
    print(ds1)

    # ds1.cast_category('date', 'text')
    # ds1.cast_type('date', dt.datetime, format="%Y-%m-%d")

    # ds1.drop_rows([0, 1, 2])
    # ds1.drop_columns(['wind_speed'])

    # ds1.to_dummies('Embarked', drop_first=False, drop_categorical=True)

    # ds1.split(test=0.6, validate=0.15, seed=42)
    # print(ds1.split_indices['validate'])

    # to-do list
    # before commit ideas below
    # upload to pypi

    # after writing code, before commit:
    # think through each attribute and method, the larger scope, and if what's defined in the class makes sense
    # check for repetitive code
    # clean up docstrings, check for each parameter, return, and attribute
    # run tests
    # commit


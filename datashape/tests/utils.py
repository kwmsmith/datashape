import sys
import pandas as pd
import numpy as np
from odo.numpy_dtype import unit_to_dtype
import string
import datashape as ds
from hypothesis import given, settings, Verbosity
import operator
from functools import reduce
from hypothesis.strategies import (text, dictionaries, sampled_from, composite,
                                   one_of, characters, lists, integers, floats, just)

scalar_dshapes = {'bool',
                  'int8',
                  'int16',
                  'int32',
                  'int64',
                  'uint8',
                  'uint16',
                  'uint32',
                  'uint64',
                  'float32',
                  'float64',
                  'complex64',
                  'complex128',
                  'string',
                  'json',
                  'date',
                  'time',
                  'datetime',
                  'int',
                  'real',
                  'complex',
                  'intptr',
                  'uintptr'}

non_opt_col_types = {ds.bool_,
                     ds.c_byte,
                     ds.c_int,
                     ds.c_size_t,
                     ds.c_ulonglong,
                     ds.c_double,
                     ds.c_long,
                     ds.c_ssize_t,
                     ds.c_ushort,
                     ds.c_float,
                     ds.c_longlong,
                     ds.c_ubyte,
                     ds.c_half,
                     ds.c_short,
                     ds.c_ulong,
                     # ds.string,
                     # ds.unicode,
                     ds.date_,
                     ds.time_,
                     ds.datetime_,
                     ds.timedelta_,
                     ds.float16,
                     ds.float32,
                     ds.float64,
                     ds.complex64,
                     ds.complex128,
                     ds.intptr,
                     ds.uintptr,
                     ds.double,
                     ds.real,
                     ds.int8,
                     ds.int16,
                     ds.int32,
                     ds.int64,
                     ds.uint8,
                     ds.uint16,
                     ds.uint32,
                     ds.uint64,}

opt_col_types = {ds.Option(ct) for ct in non_opt_col_types}
all_col_types = non_opt_col_types.union(opt_col_types)

chars = string.ascii_letters + string.digits + ' \t'

def record_dicts(col_types, min_cols, max_cols):
    return dictionaries(text(chars, min_size=1).map(str.strip),
                        sampled_from(col_types),
                        min_size=min_cols,
                        max_size=max_cols)

def at_most_one_ellipsis(l):
    return l.count('...') <= 1

dimensions = (lists(integers(min_value=0).map(str)
                    | sampled_from(['...', 'var']), min_size=1)
              .filter(at_most_one_ellipsis))

@composite
def struct_dshape(draw, col_types=all_col_types, min_cols=None, max_cols=None):
    return ds.Record(draw(record_dicts(col_types, min_cols, max_cols)).items())

@composite
def str_column(draw, dshape, n, name, with_na):
    if with_na:
        elements = text(chars) | just(None)
    else:
        elements = text(chars)
    return pd.Series(draw(lists(elements,
                                min_size=n,
                                max_size=n)),
                     dtype=unit_to_dtype(dshape),
                     name=name)

@composite
def int_column(draw, dshape, n, name, with_na):
    issigned = ds.matches_typeset(dshape, ds.unsigned)
    nbits = {ds.dshape('int64'): 64,
             ds.dshape('uint64'): 64,
             ds.dshape('int32'): 32,
             ds.dshape('uint32'): 32,}.get(dshape, 32)
    maxsize = (2 ** nbits) - 1
    min_value = 0 if not issigned else -(maxsize >> 1)
    max_value = maxsize if not issigned else (maxsize >> 1)
    if with_na:
        elements = (floats(allow_infinity=False,
                           min_value=min_value,
                           max_value=max_value)
                    .map(lambda f: round(f, 0))) | just(np.nan)
        dshape = 'float64' if dshape in (ds.dshape('int64'), ds.dshape('uint64')) else 'float32'
    else:
        elements = integers(min_value=min_value,
                            max_value=max_value)
    return pd.Series(draw(lists(elements,
                                min_size=n,
                                max_size=n)),
                     dtype=unit_to_dtype(dshape),
                     name=name)

@composite
def float_column(draw, dshape, n, name, with_na):
    if with_na:
        elements = floats(allow_nan=True)
    else:
        elements = floats(allow_nan=False)
    return pd.Series(draw(lists(elements,
                                min_size=n,
                                max_size=n)),
                     dtype=unit_to_dtype(dshape),
                     name=name)


@composite
def dataframes(draw,
               col_types={'string', '?string', 'int32', '?int32', 'uint32', '?uint32', 'float32', 'float64'},
               min_rows=0,
               max_rows=10,
               min_cols=1,
               max_cols=10):
    col_types = {ds.dshape(ct) for ct in col_types}
    struct_ds = draw(struct_dshape(col_types, min_cols, max_cols))
    nrows = draw(integers(min_value=(min_rows or 0), max_value=max_rows))
    cols = []
    for name, dshape in struct_ds.fields:
        is_option = isinstance(dshape, ds.Option)
        dshape = getattr(dshape.measure, 'ty', dshape.measure)
        if ds.matches_typeset(dshape, ds.integral):
            cols.append((name, draw(int_column(dshape, nrows, name, is_option))))
        elif ds.matches_typeset(dshape, ds.floating):
            cols.append((name, draw(float_column(dshape, nrows, name, is_option))))
        elif dshape == ds.string:
            cols.append((name, draw(str_column(dshape, nrows, name, is_option))))
        else:
            raise NotImplementedError(dshape)
    return pd.DataFrame.from_items(cols)

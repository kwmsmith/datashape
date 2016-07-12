import datashape as ds
from datashape.tests.utils import dimensions, struct_dshape
from hypothesis import given, settings, Verbosity

@given(dims=dimensions, struct=struct_dshape())
@settings(max_examples=200, verbosity=Verbosity.verbose)
def test_dshape_record(dims, struct):
    from datashape import dshape
    dd = ds.dshape(' * '.join(dims + [str(struct)]))
    assert eval(repr(dd)) == dd

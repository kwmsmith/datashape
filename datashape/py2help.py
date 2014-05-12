import sys
import itertools

# Portions of this taken from the six library, licensed as follows.
#
# Copyright (c) 2010-2013 Benjamin Peterson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


PY2 = sys.version_info[0] == 2

if PY2:
    import __builtin__
    reduce = __builtin__.reduce
    _inttypes = (int, long)
    unicode = __builtin__.unicode
    basestring = __builtin__.basestring
    _strtypes = (str, unicode)
else:
    from functools import reduce
    _inttypes = (int,)
    unicode = str
    basestring = str
    _strtypes = (str,)

if sys.version_info[:2] >= (2, 7):
    from unittest import skip, skipIf
else:
    from nose.plugins.skip import SkipTest
    class skip(object):
        def __init__(self, reason):
            self.reason = reason

        def __call__(self, func):
            from nose.plugins.skip import SkipTest
            def wrapped(*args, **kwargs):
                raise SkipTest("Test %s is skipped because: %s" %
                                (func.__name__, self.reason))
            wrapped.__name__ = func.__name__
            return wrapped
    class skipIf(object):
        def __init__(self, condition, reason):
            self.condition = condition
            self.reason = reason

        def __call__(self, func):
            if self.condition:
                from nose.plugins.skip import SkipTest
                def wrapped(*args, **kwargs):
                    raise SkipTest("Test %s is skipped because: %s" %
                                    (func.__name__, self.reason))
                wrapped.__name__ = func.__name__
                return wrapped
            else:
                return func

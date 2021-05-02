"""
Converting and subsetting TrueType fonts to PS types 3 and 42, and PDF type 3.
"""

from . import _api
from ._ttconv import convert_ttf_to_ps, get_pdf_charprocs  # noqa


_api.warn_deprecated('3.3', name=__name__, obj_type='module')

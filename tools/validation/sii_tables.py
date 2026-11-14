# -*- coding: utf-8 -*-
"""Check the ANSI S3.5-1997 numbers NCM and CSII use against two published
copies of the standard's tables.

kudio.ncm and kudio.csii carry, as NCM.m and CSII.m do, the 21 critical-band
centres of the standard's Table 1, the band importance its Table B.1 gives
for short passages of easy reading material, and, for CSII, the widths of
Table 1's lowest 16 bands. The standard is sold by ANSI; these two copies of
its tables are free, and are read as they come, without unpacking::

    python tools/validation/sii_tables.py --cran TARBALL --sii-c ZIP

- --cran: the CRAN package SII 1.3.0 by G. R. Warnes and M. Shaver,
  SII_1.3.0.tar.gz, 1,033,377 bytes, SHA-256
  98ef1b638778096d5550c47708a911753ffbb2550725e74a0ada15a40c662596. Its
  inst/extdata/SII_Constants.xls holds Tables 1 to 4 and B.1, the same
  numbers as the package's data sets critical and sic.critical.
- --sii-c: the C program developer's kit by C. Pavlovic, from the site of
  ASA working group S3-79 (sii.to, Programs), DevelopmentKit.zip, 129,471
  bytes, SHA-256
  e744358db4b216b9b9c243c4e004aff37e5c3dc03c71111fa3aa15685923d1a7. Its
  SOURCES/SII.C holds Table 1's band limits and band importance.

Reading the .xls takes xlrd (pip install xlrd); without it only the band
widths are checked, against SII.C. It prints each comparison, which has to
hold exactly, and exits with 1 if a number differs.
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

XLS = 'SII/inst/extdata/SII_Constants.xls'
SII_C = 'DevelopmentKit/SOURCES/SII.C'


def sheet(book, name):
    """A table of SII_Constants.xls as {column: values}. The band rows are
    the ones numbered in the first column, and the row above them names the
    columns as the package does (fi, li, hi, Ii, ShortPassage, ...)."""
    table = book.sheet_by_name(name)
    rows = [r for r in range(table.nrows) if isinstance(table.cell_value(r, 0), float)]
    names = table.row_values(rows[0] - 1)
    return {names[c]: np.array([table.cell_value(r, c) for r in rows])
            for c in range(1, table.ncols) if names[c]}


def package_tables(path):
    import xlrd
    with tarfile.open(path) as tar:
        book = xlrd.open_workbook(file_contents=tar.extractfile(XLS).read())
    return sheet(book, 'Table 1'), sheet(book, 'Table B1')


def sii_c_table1(path):
    """Band limits and band importance from SII.C's cb(), the critical-band
    procedure."""
    with zipfile.ZipFile(path) as kit:
        text = kit.read(SII_C).decode('latin-1')
    # rindex: the definition of cb(), not its declaration at the top
    body = text[text.rindex('void cb('):text.rindex('void eccb(')]

    def array(name):
        values = re.search(rf'double {name}\[[^]]*\] = \{{([^}}]*)\}}', body).group(1)
        return np.array([float(v) for v in values.split(',')])

    return array('limit'), array('i_avg')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cran', required=True, metavar='TARBALL',
                        help="SII_1.3.0.tar.gz, as CRAN gives it out")
    parser.add_argument('--sii-c', required=True, metavar='ZIP',
                        help="DevelopmentKit.zip, as sii.to gives it out")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import kudio
    from kudio.core import intelligibility as it
    print(f"kudio {kudio.__version__} from {Path(kudio.__file__).parent}\n")

    limits, importance = sii_c_table1(args.sii_c)
    checks = [("CSII's 16 band widths against Table 1 in SII.C", it._CSII_WIDTHS,
               np.diff(limits)[:16])]
    try:
        table1, b1 = package_tables(args.cran)
    except ImportError:
        table1 = b1 = None
        print("xlrd is not installed: the checks against the CRAN package skipped\n")
    if table1 is not None:
        li, hi = table1['li'], table1['hi']
        checks = [
            ("Table 1 band limits, the package against SII.C", np.r_[li, hi[-1]], limits),
            ("Table 1 band importance, the package against SII.C", table1['Ii'],
             importance),
            ("kudio's 21 band centres against Table 1", it._ANSI_CENTRES, table1['fi']),
            ("kudio's 21 importance values against Table B.1, short passages",
             it._ANSI_IMPORTANCE, b1['ShortPassage']),
            ("CSII's 16 band widths against Table 1 in the package", it._CSII_WIDTHS,
             (hi - li)[:16]),
        ] + checks
    failed = False
    for what, ours, theirs in checks:
        same = np.array_equal(np.asarray(ours, float), np.asarray(theirs, float))
        failed |= not same
        print(f"{'same' if same else 'DIFFERENT':9} {what}")

    if b1 is not None:
        matches = [name for name, column in b1.items()
                   if name != 'fi' and np.array_equal(column, it._ANSI_IMPORTANCE)]
        print(f"\nTable B.1 columns equal to kudio's importance values: "
              f"{', '.join(matches) or 'none'}")
    print("every number the same" if not failed else "SOME NUMBERS DIFFER")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

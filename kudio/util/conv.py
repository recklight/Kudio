# -*- coding: utf-8 -*-
"""Data conversion/export helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from kudio.exceptions import DependencyError

__all__ = [
    'data2xlsx',
    'lowercase',
    'upercase',
    'uppercase',
]

log = logging.getLogger(__name__)


def _pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise DependencyError('pandas', extra='data') from e


def data2xlsx(data: Union[dict, "pd.DataFrame"], xlsx_dir, sheet_name: str,
              index: bool = False, fit_column: bool = True):
    """Write a dict or DataFrame to a sheet of an ``.xlsx`` file.

    If the file already exists, the sheet is replaced; other sheets are kept.
    With ``fit_column=True``, column widths are auto-sized to the content.
    """
    pd = _pandas()
    if isinstance(data, dict):
        try:
            data_frame = pd.DataFrame.from_dict(data, orient='columns')
        except Exception as e:
            log.error("data2xlsx: bad data: %s", e)
            return f'[data2xlsx] data error: {e}'
    elif isinstance(data, pd.DataFrame):
        data_frame = data
    else:
        return '[data2xlsx] check the data type'

    xlsx_dir = Path(xlsx_dir)
    xlsx_dir.parent.mkdir(parents=True, exist_ok=True)

    if xlsx_dir.is_file():
        with pd.ExcelWriter(xlsx_dir, engine='openpyxl', mode='a',
                            if_sheet_exists='replace') as writer:
            data_frame.to_excel(writer, sheet_name=sheet_name, index=index)
            if fit_column:
                from openpyxl.utils import get_column_letter
                worksheet = writer.book[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value)) for cell in column_cells)
                    worksheet.column_dimensions[
                        get_column_letter(column_cells[0].column)].width = length
    else:
        data_frame.to_excel(xlsx_dir, sheet_name=sheet_name, index=index)
    return None


def lowercase(lst) -> List[str]:
    return [_.lower() for _ in lst if isinstance(_, str)]


def uppercase(lst) -> List[str]:
    return [_.upper() for _ in lst if isinstance(_, str)]


upercase = uppercase  # backwards-compatible misspelling

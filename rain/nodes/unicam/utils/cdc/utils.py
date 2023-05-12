from __future__ import annotations
import hashlib
from typing import Union
from pandas import Series


def process_row(row: Series, tts_legend: list[Union[int, str]]) -> Union[str, str, bool, bool]:
    generated_key = generate_artificial_key(row)
    activity = generate_activity_name(row, tts_legend)
    is_start = is_start_activity(row)
    is_end = is_end_activity(row)
    return generated_key, activity, is_start, is_end


def generate_artificial_key(row: Series) -> str:
    key_fields = ['ST_ITEM_NO', 'ST_LOC_ID', 'ST_BU_CODE_SUP', 'ST_BU_TYPE_SUP', 'ST_PROD_DATE', 'ST_REG_DATE']
    values = []
    for field in key_fields:
        values.append(str(row[field]))

    joined_value = ','.join(values)
    return hashlib.sha256(joined_value.encode('utf-8')).hexdigest()


def generate_activity_name(row: Series, tts_legend: list[Union[int, str]]) -> str:
    if tt_string := row["ST_TRANS_TYPE"]:
        transaction_type = int(tt_string)
        x = next((tt for tt in tts_legend if tt[0] == transaction_type), None)
        return f"{x[1]} [{x[0]}]"
    else:
        return ""


def is_start_activity(row: Series) -> bool:
    adjust_quantity = int(row["ST_ADJUST_QTY"])
    quantity = int(row["ST_ITEM_QTY"])
    return adjust_quantity > 0 and adjust_quantity == quantity


def is_end_activity(row: Series) -> bool:
    adjust_quantity = int(row["ST_ADJUST_QTY"])
    quantity = int(row["ST_ITEM_QTY"])
    return adjust_quantity < 0 and quantity == 0
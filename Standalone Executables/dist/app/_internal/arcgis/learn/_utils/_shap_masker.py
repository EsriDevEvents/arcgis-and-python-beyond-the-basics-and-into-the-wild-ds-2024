import re


def custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API.
    This tokenizer is an standard function to wrap the explanation to the
    word boundaries compared to the token level information
    ====================================================================
    Input                   Description
    ====================================================================
    s                       Input string

    :return return a dictionary with the offset mapping for the words
    """
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\s", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out

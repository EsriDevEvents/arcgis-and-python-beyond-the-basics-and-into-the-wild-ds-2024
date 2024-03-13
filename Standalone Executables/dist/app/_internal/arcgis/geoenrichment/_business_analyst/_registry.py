import os
from pathlib import Path
from typing import Optional, Union
import winreg

import numpy as np


def get_child_key_strs(key_path) -> list:
    """
    Get the full path of first generation child keys under the parent key listed.
    Args:
        key_path: Path to the parent key in registry.
    Returns: List of the full path to child keys.
    """
    # open the parent key
    parent_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)

    # variables to track progress and store results
    error = False
    counter = 0
    key_list = []

    # while everything is going good
    while not error:
        try:
            # get the child key in the iterated position
            child_key = winreg.EnumKey(parent_key, counter)

            # add the located key to the list
            key_list.append("{}\\{}".format(key_path, child_key))

            # increment the counter
            counter += 1

        # when something blows up...typically because no key is found
        except Exception as e:
            # switch the error flag to true, stopping the iteration
            error = True

    # give the accumulated list back
    return key_list


def get_first_child_key_str(key_path, pattern) -> str:
    """
    Based on the pattern provided, find the key with a matching string in it.
    :param key_path: Full string path to the key.
    :param pattern: Pattern to be located.

    :return: Full path of the first key path matching the provided pattern.
    """
    # get a list of paths to keys under the parent key path provided
    key_list = get_child_key_strs(key_path)

    # iterate the list of key paths
    for key in key_list:
        # if the key matches the pattern
        if key.find(pattern):
            # pass back the provided key path
            return key


def get_ba_country_key_str(
    three_letter_country_code: str, year: Optional[Union[int, str]] = None
) -> str:
    """Lookup the country registry key by three letter country identifier and year."""
    cntry_cd = three_letter_country_code.upper()
    cntry_key_lst = get_child_key_strs(
        r"SOFTWARE\WOW6432Node\Esri\BusinessAnalyst\Datasets"
    )

    key_dict = {
        os.path.basename(k).split("_")[2]: k
        for k in cntry_key_lst
        if os.path.basename(k).split("_")[0] == cntry_cd
    }

    yr_lst = [int(y) for y in key_dict.keys()]
    yr_lst.sort()
    assert len(yr_lst)

    if year:
        year = (
            str(year) if isinstance(year, int) or isinstance(year, np.int64) else year
        )
        assert isinstance(year, str)
        assert len(year) == 4
        cntry_key = key_dict[year]
    else:
        cntry_key = key_dict[str(yr_lst[-1:][0])]
        assert len(
            yr_lst
        ), f"It appears {cntry_cd} {year} is not installed on this machine."

    return cntry_key


def get_ba_key_value(
    locator_key,
    three_letter_country_identifier: str = "USA",
    year: Optional[Union[int, str]] = None,
) -> Path:
    """
    In the Business Analyst key, get the value corresponding to the provided locator key.
    :param locator_key: Locator key.
    :param three_letter_country_identifier: Three letter country identification code.
    :param year: Four digit year describing the vintage of the data.

    :return: Key value.
    """
    # get the registry key string path
    country_key_str = get_ba_country_key_str(three_letter_country_identifier, year)

    # open the key to the current installation of Business Analyst data
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, country_key_str)

    # query the value of the locator key
    return winreg.QueryValueEx(key, locator_key)[0]


def get_ba_network_dataset_path(
    three_letter_country_identifier: str, vintage_year: Optional[int] = None
) -> Path:
    """
    Get the path to the transportation network dataset.
    :param three_letter_country_identifier: Three letter country identification code.

    :return: String describing resource location.
    """
    return Path(
        get_ba_key_value(
            "StreetsNetwork", three_letter_country_identifier, vintage_year
        )
    )

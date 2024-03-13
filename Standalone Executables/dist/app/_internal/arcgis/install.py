#!/usr/bin/env python

# Thanks @takluyver for your cite2c install.py.
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import print_function

import argparse
from os.path import dirname, abspath, join as pjoin


def install(user=False, symlink=False, enable=False):
    """Install the widget nbextension and optionally enable it.

    Parameters
    ----------
    user: bool
        Install for current user instead of system-wide.
    symlink: bool
        Symlink instead of copy (for development).
    """
    try:
        from notebook.nbextensions import install_nbextension
        from notebook.services.config import ConfigManager
    except ModuleNotFoundError:
        print('"notebook" not installed, silently failing...')
        return
    widgetsdir = pjoin(dirname(abspath(__file__)), "widgets")
    install_nbextension(widgetsdir, destination="arcgis", user=user, symlink=symlink)

    cm = ConfigManager()
    cm.update(
        "notebook",
        {
            "load_extensions": {
                "arcgis/mapview": True,
            }
        },
    )


def uninstall():
    try:
        from notebook.nbextensions import uninstall_nbextension

        """Uninstall the widget nbextension from user and system locations
        """
        print("Uninstalling prior versions of arcgis widget")
        uninstall_nbextension("arcgis", user=True)
        uninstall_nbextension("arcgis", user=False)
    except:
        print(
            'Manually uninstall any prior version of arcgis widget using:\n\t"jupyter nbextension uninstall arcgis --user" and \n\t"jupyter nbextension uninstall arcgis"'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Installs the ArcGIS IPython widgets")
    parser.add_argument(
        "-u",
        "--user",
        help="Install as current user instead of system-wide",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--symlink", help="Symlink instead of copying files", action="store_true"
    )
    parser.add_argument(
        "-r",
        "--remove",
        help="Remove i.e. uninstall the extension",
        action="store_true",
    )
    args = parser.parse_args()

    if args.remove:
        uninstall()
    else:
        install(user=args.user, symlink=args.symlink)

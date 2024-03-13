import json
import os
import os.path
import re
import logging

log = logging.getLogger()

import ipykernel
from notebook.notebookapp import list_running_servers
from IPython.display import HTML, display


def get_dir_of_curr_exec_notebook():
    """
    Returns the absolute path of the directory that the currently executing
    jupyter notebook resides in
    """
    try:
        kernel_id = re.search(
            "kernel-(.*).json", ipykernel.connect.get_connection_file()
        ).group(1)
        servers = list_running_servers()
        for ss in servers:
            import requests
            from requests.compat import urljoin

            response = requests.get(
                urljoin(ss["url"], "api/sessions"),
                params={"token": ss.get("token", "")},
            )
            for nn in json.loads(response.text):
                if nn["kernel"]["id"] == kernel_id:
                    relative_path = nn["notebook"]["path"]
                    nb_path = os.path.join(ss["notebook_dir"], relative_path)
                    return os.path.dirname(nb_path)
    except Exception:
        log.debug(
            "Could not discover the directory of the currently "
            "executing notebook, using os.getcwd(). To override "
            "and specify the dir manually, call `MapView.raster."
            "set_current_executing_nb_dir('/path/to/dir/'`)"
        )
        return os.getcwd()


class JupyterHorizScrollImageList:
    def __init__(self):
        self._img_el_properties = []

    def append(self, src, title):
        self._img_el_properties.append({"src": src, "title": title})

    def _ipython_display_(self):
        html = ""
        for img in self._img_el_properties:
            html += f'\n<img src="{img["src"]}" title="{img["title"]}"></img>'
        html = f'<div class="scrollmenu">{html}\n</div>'
        css = """
              div.scrollmenu {
                overflow: auto;
                white-space: nowrap;
              }
              div.scrollmenu img {
                margin-top: 1em;
                display: inline-block;
                width: 25%;
                height: auto;
                color: white;
                text-align: center;
                padding: 14px;
                text-decoration: none;
              }
              div.scrollmenu img:hover {
                background-color: #777;
              }
              """
        html = f"<style>{css}</style><html>{html}</html>"
        display(HTML(html))

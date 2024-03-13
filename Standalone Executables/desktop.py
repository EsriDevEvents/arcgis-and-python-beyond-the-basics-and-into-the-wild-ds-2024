from contextlib import redirect_stdout
from io import StringIO

from app import app

import webview


if __name__ == "__main__":
    stream = StringIO()
    with redirect_stdout(stream):
        window = webview.create_window("Web GIS Content Search", app)
        webview.start()

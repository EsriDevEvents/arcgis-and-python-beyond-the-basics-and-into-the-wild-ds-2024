import uuid
import arcgis


class SidePanel(object):
    """
    Creates a dashboard Side Panel widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    title                       Optional string. Title of the header.
    -------------------------   -------------------------------------------
    description                 Optional string. Description of the widget.
    =========================   ===========================================
    """

    def __init__(self, title=None, description=None):
        # Side Panel starts here

        self._title = title
        self._description = description
        self.type = "leftPanel"

        self._text_color = None
        self._background_color = None

        self._allow_sliding = False

        self._selectors = []

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard

        url = Dashboard._publish_random(self)
        return f"""<iframe src={url} width=300 height=300>"""

    @property
    def title(self):
        """
        :return: Title of the widget.
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set title of the widget.
        """
        if isinstance(value, str):
            self._title = value

    @property
    def description(self):
        """
        :return: Description of the widget.
        """
        return self.description

    @description.setter
    def description(self, value):
        """
        Set description of the widget.
        """
        if isinstance(value, str):
            self.description = value

    @property
    def text_color(self):
        """
        :return: Text color of the side panel.
        """
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        """
        :return: Set Text color of the side panel widget. Hex color.
        """
        self._text_color = value
        if not isinstance(value, str):
            self._text_color = "#ffffff"

    @property
    def background_color(self):
        """
        :return: Background color of the side panel widget.
        """
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        """
        Set background color of the side panel widget.
        """
        self._background_color = value
        if not isinstance(value, str):
            self._background_color = "#000000"

    @property
    def allow_sliding(self):
        """
        :return: True or False for sliding enabled or disabled.
        """
        return self._allow_sliding

    @allow_sliding.setter
    def allow_sliding(self, value):
        """
        Set True or False to enable sliding or to disable sliding.
        """
        self._allow_sliding = value
        if not isinstance(value, bool):
            self._allow_sliding = False

    def add_selector(self, selector):
        """
        Add Number Selector, Category Selector or Date Picker widget.
        """
        self._selectors.append(selector)

    def _convert_to_json(self):
        side_panel = {
            "type": "leftPanel",
            "allowSliding": self._allow_sliding,
            "selectors": [selector._convert_to_json() for selector in self._selectors],
        }

        if self._background_color:
            side_panel["backgroundColor"] = self._background_color

        if self._text_color:
            side_panel["textColor"] = self._text_color

        if self._title:
            side_panel["title"] = self._title

        if self._description:
            side_panel["description"] = self._description

        return side_panel

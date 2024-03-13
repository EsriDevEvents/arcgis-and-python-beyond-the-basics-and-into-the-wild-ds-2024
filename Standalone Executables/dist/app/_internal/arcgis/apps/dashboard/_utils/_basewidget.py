import uuid


class _BaseWidget(object):
    def __init__(self, name, title, description):
        self._title = ""
        self._name = ""
        self._description = ""
        self._background_color = None  # "#ffffff"
        self._text_color = None  # "#000000"
        self._id = str(uuid.uuid4())
        self._type = None

        self.title = title
        self.name = name
        self.description = description

        self._width = 1
        self._height = 1

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard

        url = Dashboard._publish_random(self)
        return f"""<iframe src={url} width=900 height=300>"""

    @property
    def type(self):
        """
        :return: widget type.
        """
        return self._type

    @type.setter
    def type(self, value):
        """
        Set widget type.
        """
        self._type = str(value)

    @property
    def name(self):
        """
        :return: widget name.
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Set widget name.
        """
        self._name = value

    @property
    def title(self):
        """
        :return: widget title.
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set widget title.
        """
        self._title = value

    @property
    def description(self):
        """
        :return: widget description.
        """
        return self._description

    @description.setter
    def description(self, value):
        """
        Set widget description.
        """
        self._description = value

    @property
    def text_color(self):
        """
        :return: widget text color.
        """
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        """
        Set widget text color.
        """
        if isinstance(value, str):
            self._text_color = value

    @property
    def background_color(self):
        """
        :return: widget background color.
        """
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        """
        Set widget background color.
        """
        if isinstance(value, str):
            self._background_color = value

    @property
    def height(self):
        """
        :return: Height of the widget
        """
        return self._height

    @height.setter
    def height(self, value):
        """
        Set height of the widget, between 0 and 1.
        """
        if value > 1:
            self._height = 1
        elif value < 0:
            self._height = 0
        else:
            self._height = value

    @property
    def width(self):
        """
        :return: Width of the widget
        """
        return self._width

    @width.setter
    def width(self, value):
        """
        Set width of the widget, between 0 and 1.
        """
        if value > 1:
            self._width = 1
        elif value < 0:
            self._width = 0
        else:
            self._width = value


class Legend(object):
    @classmethod
    def _create_legend(cls, visible=True, placement="bottom"):
        legend = Legend()

        if placement not in ["bottom", "side"]:
            raise Exception(
                "Please specify correct placement. Supported placement are 'bottom' and 'side'"
            )

        legend._visibility = True
        legend._placement = "bottom"

        legend.visibility = visible
        legend.placement = placement

        return legend

    @property
    def visibility(self):
        """
        :return: Legend Visible or not.
        """
        return self._visibility

    @visibility.setter
    def visibility(self, value):
        """
        Set Legends visbility.
        """
        self._visibility = bool(value)

    @property
    def placement(self):
        """
        :return: Legend placement from 'bottom', 'side'
        """
        return self._placement

    @placement.setter
    def placement(self, value):
        """
        Set Legend placement from 'bottom', 'side'.
        """
        if value in ["bottom", "side"]:
            self._placement = value


class NoDataProperties(object):
    @classmethod
    def _nodata_init(
        cls,
        text="No Data",
        alignment="middle",
        show_title=True,
        show_description=True,
    ):
        nodata = NoDataProperties()

        if alignment not in ["top", "middle", "bottom"]:
            raise Exception(
                "Please specify correct alignment. Supported alignment are 'top', 'bottom' and 'middle'"
            )

        nodata._text = text
        nodata._alignment = alignment
        nodata._show_title = show_title
        nodata._show_description = show_description

        nodata.text = text
        nodata.alignment = alignment
        nodata.show_title = show_title
        nodata.show_description = show_description

        return nodata

    @property
    def text(self):
        """
        :return: No Data text.
        """
        return self._text

    @text.setter
    def text(self, value):
        """
        Set No Data text.
        """
        self._text = str(value)

    @property
    def alignment(self):
        """
        :return: No Data text vertical alignment.
        """
        return self._alignment

    @alignment.setter
    def alignment(self, value):
        """
        Set No Data text vertical alignment.
        """
        if value in ["top", "middle", "bottom"]:
            self._alignment = str(value)

    @property
    def show_title(self):
        """
        :return: No Data show title.
        """
        return self._show_title

    @show_title.setter
    def show_title(self, value):
        """
        Set No Data show title.
        """
        self._show_title = bool(value)

    @property
    def show_description(self):
        """
        :return: No Data show description.
        """
        return self._show_description

    @show_description.setter
    def show_description(self, value):
        """
        Set No Data show description.
        """
        self._show_description = bool(value)


def _auto_calculate_width(elements):
    import arcgis

    available_width = 1
    remaining_elements = len(elements)

    for el in elements:
        element_width = getattr(el, "width", 1)
        if isinstance(el, dict) and not isinstance(el, arcgis.mapping.WebMap):
            element_width = el.get("width", 1)

        if element_width != 1:
            remaining_elements = remaining_elements - 1
            available_width = available_width - element_width

    if remaining_elements > 0:
        available_width = float(available_width / remaining_elements)

    for el in elements:
        if isinstance(el, dict) and not isinstance(el, arcgis.mapping.WebMap):
            if el.get("width", 1) == 1:
                el["width"] = available_width
        elif getattr(el, "width", 1) == 1:
            el.width = available_width

    return elements


def _auto_calculate_height(elements):
    import arcgis

    available_height = 1
    remaining_elements = len(elements)

    for el in elements:
        element_height = getattr(el, "height", 1)
        if isinstance(el, dict) and not isinstance(el, arcgis.mapping.WebMap):
            element_height = el.get("height", 1)

        if element_height != 1:
            remaining_elements = remaining_elements - 1
            available_height = available_height - element_height

    if remaining_elements > 0:
        available_height = float(available_height / remaining_elements)

    for el in elements:
        if isinstance(el, dict) and not isinstance(el, arcgis.mapping.WebMap):
            if el.get("height", 1) == 1:
                el["height"] = available_height
        elif getattr(el, "height", 1) == 1:
            el.height = available_height

    return elements


def add_row(elements, height=1):
    """
    Creates a Row Layout.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    elements                    Required List. Widgets that will be added
                                to the row in the dashboard layout.
    -------------------------   -------------------------------------------
    height                      Optional int. Height of the row.
    =========================   ===========================================
    """

    elements = _auto_calculate_height(elements)
    json = {
        "type": "stackLayoutElement",
        "orientation": "row",
        "elements": [],
        "width": 1,
        "height": height,
        "widgets": [],
    }

    for el in elements:
        if not hasattr(el, "_id"):
            json["widgets"] = json["widgets"] + el["widgets"]
            del el["widgets"]
            json["elements"].append(el)
        else:
            json["elements"].append(
                {
                    "type": "itemLayoutElement",
                    "id": el._id,
                    "height": el.height,
                    "width": el.width,
                }
            )
            json["widgets"].append(el)

    return json


def add_column(elements, width=1):
    """
    Creates a Column Layout.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    elements                    Required List. Widgets that will be added to
                                the column in the dashboard layout.
    -------------------------   -------------------------------------------
    width                       Optional int. Width of the Column.
    =========================   ===========================================
    """

    elements = _auto_calculate_width(elements)
    json = {
        "type": "stackLayoutElement",
        "orientation": "col",
        "elements": [],
        "width": width,
        "height": 1,
        "widgets": [],
    }

    for el in elements:
        if not hasattr(el, "_id"):
            json["widgets"] = json["widgets"] + el["widgets"]
            del el["widgets"]
            json["elements"].append(el)
        else:
            json["elements"].append(
                {
                    "type": "itemLayoutElement",
                    "id": el._id,
                    "height": el.height,
                    "width": el.width,
                }
            )
            json["widgets"].append(el)

    return json

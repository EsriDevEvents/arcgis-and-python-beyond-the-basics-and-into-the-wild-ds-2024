import uuid
import arcgis
from .._utils._basewidget import _BaseWidget


class MapLegend(_BaseWidget):
    """
    Create a MapLegend widget for Dashboard

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    map_widget                  Required web map widget. Legend for this
                                Map widget is displayed.
                                This map widget needs to be a part of the
                                final Dashboard as well.
    -------------------------   -------------------------------------------
    name                        Optional String. Name of the widget.
    -------------------------   -------------------------------------------
    title                       Optional string. Title of the widget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description of the widget.
    =========================   ===========================================
    """

    def __init__(self, map_widget, name="MapLegend", title="", description=""):
        super().__init__(name, title, description)

        self._map_widget = map_widget
        self._type = "legendWidget"

    @classmethod
    def _from_json(cls, widget_json):
        map_widget = widget_json["mapWidgetId"]
        name = widget_json["name"]
        title = widget_json["caption"]
        description = widget_json["description"]
        map_legend = MapLegend(map_widget, name, title, description)
        map_legend._id = widget_json["id"]
        map_legend.no_data.alignment = widget_json["noDataVerticalAlignment"]
        map_legend.no_data.show_title = widget_json["showCaptionWhenNoData"]
        map_legend.no_data.show_description = widget_json["showDescriptionWhenNoData"]

        return map_legend

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard

        url = Dashboard._publish_random(self)
        return f"""<iframe src={url} width=900 height=300>"""

    def _convert_to_json(self):
        data = {
            "type": "legendWidget",
            "mapWidgetId": self._map_widget._id,
            "id": self._id,
            "name": self.name,
            "caption": self._title,
            "description": self._description,
            "showLastUpdate": True,
            "noDataVerticalAlignment": "middle",
            "showCaptionWhenNoData": True,
            "showDescriptionWhenNoData": True,
        }

        if self._background_color:
            data["backgroundColor"] = self._background_color

        if self._text_color:
            data["textColor"] = self._text_color

        return data

import uuid
import arcgis
from .._utils._basewidget import _BaseWidget, NoDataProperties


class RichText(_BaseWidget):
    """
    Creates a dashboard Rich Text widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    html_text                   Required HTML text. This text will be
                                displayed in Rich Text format.
    -------------------------   -------------------------------------------
    name                        Optional String. Name of the widget.
    -------------------------   -------------------------------------------
    title                       Optional String. Title of the widget.
    -------------------------   -------------------------------------------
    description                 Optional String. Description of the widget.
    =========================   ===========================================
    """

    def __init__(self, html_text, name="RichText", title="", description=""):
        super().__init__(name, title, description)

        self._type = "richTextWidget"
        self._text = ""

        self.text = html_text
        self._nodata = NoDataProperties._nodata_init()

    @classmethod
    def _from_json(cls, widget_json):
        txt = widget_json["text"]
        name = widget_json["name"]
        title = widget_json["caption"]
        description = widget_json["description"]
        rtxt = RichText(txt, name, title, description)
        rtxt._id = widget_json["id"]
        rtxt.no_data.alignment = widget_json["noDataVerticalAlignment"]
        rtxt.no_data.show_title = widget_json["showCaptionWhenNoData"]
        rtxt.no_data.show_description = widget_json["showDescriptionWhenNoData"]

        return rtxt

    @property
    def type(self):
        """
        :return: Widget type.
        """
        return self._type

    @property
    def text(self):
        """
        :return: Text field for rich text
        """
        return self._text

    @text.setter
    def text(self, value):
        """
        Set text field for rich text
        """
        self._text = value

    @property
    def no_data(self):
        """
        :return: Nodata Object, set various nodata properties
        """
        return self._nodata

    def _convert_to_json(self):
        json_data = {
            "type": "richTextWidget",
            "text": self.text,
            "id": self._id,
            "name": self.name,
            "showLastUpdate": True,
            "noDataVerticalAlignment": self._nodata.alignment,
            "showCaptionWhenNoData": self._nodata.show_title,
            "showDescriptionWhenNoData": self._nodata.show_description,
        }

        if self._text_color:
            json_data["textColor"] = self._text_color

        if self._background_color:
            json_data["backgroundColor"] = self._background_color

        return json_data

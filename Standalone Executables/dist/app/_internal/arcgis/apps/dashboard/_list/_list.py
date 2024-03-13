import uuid
import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import NoDataProperties


class List(_BaseWidget):
    """
    Creates a dashboard List widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal :class:`~arcgis.gis.Item` object. Item object can
                                be a :class:`~arcgis.features.FeatureLayer`  or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the List widget.
    -------------------------   -------------------------------------------
    layer                       Optional integer. Layer number when item is
                                a mapwidget.
    -------------------------   -------------------------------------------
    title                       Optional string. Title or Caption for the
                                widget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description for the widget.
    =========================   ===========================================
    """

    def __init__(self, item, name="List", layer=0, title=None, description=None):
        super().__init__(name, title, description)

        if item.type not in ["Feature Service", "mapWidget"]:
            raise Exception("Please specify an item")

        self.item = item
        self.type = "listWidget"
        self.layer = layer

        self._seperator_color = "#f0f0f0"
        self._selection_color = "#0f0f0f"
        self._selection_text_color = "#123210"

        self._max_features = 25
        self._selection_mode = "single"

        self._show_last_update = True

        self._nodata = NoDataProperties._nodata_init()

        # List Block
        self._list_text = None
        self._list_icon = "symbol"
        self._events = Events._create_events()

    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        description = widget_json["description"]
        lists = List(item, name, title, description)

        return lists

    @property
    def events(self):
        """
        :return: list of events attached to the widget.
        """
        return self._events

    @property
    def max_features(self):
        """
        :return: max number of features to display.
        """
        return self._max_features

    @max_features.setter
    def max_features(self, value):
        """
        Set max number of features to display.
        """
        self._max_features = int(value)

    @property
    def show_last_update(self):
        """
        :return: show last update or not.
        """
        return self._show_last_update

    @show_last_update.setter
    def show_last_update(self, value):
        """
        Set show last update or not.
        """
        self._show_last_update = value
        if not isinstance(value, bool):
            self._show_last_update = True

    @property
    def seperator_color(self):
        """
        :return: Separator color.

        """
        return self._seperator_color

    @seperator_color.setter
    def seperator_color(self, value):
        """
        Set seperator color.
        """
        self._seperator_color = value
        if not isinstance(value, str):
            self._seperator_color = "#f0f0f0"

    @property
    def selection_color(self):
        """
        :return: selection color.
        """
        return self._selection_color

    @selection_color.setter
    def selection_color(self, value):
        """
        Set selection color.+
        """
        self._selection_color = value
        if not isinstance(value, str):
            self._selection_color = "#0f0f0f"

    @property
    def selection_text_color(self):
        """
        :return: selection text color.
        """
        return self._selection_text_color

    @selection_text_color.setter
    def selection_text_color(self, value):
        """
        Set selection text color.
        """
        self._selection_text_color = value
        if not isinstance(value, str):
            self._selection_text_color = "#123210"

    @property
    def list_text(self):
        """
        :return: list text.
        """
        return self._list_text

    @list_text.setter
    def list_text(self, value):
        """
        Set list text.
        """
        self._list_text = value
        if not isinstance(value, str):
            self._list_text = ""

    @property
    def list_icon(self):
        """
        :return: use icon for list or not.
        """
        return self._list_icon

    @list_icon.setter
    def list_icon(self, value):
        """
        Set use icon for list or not.
        """
        self._list_icon = value
        if not isinstance(value, str):
            self._list_icon = "symbol"

    @property
    def selection_mode(self):
        """
        :return: selection mode.
        """
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, value):
        """
        Set selection mode
        """
        self._selection_mode = value
        if not isinstance(value, str):
            self._selection_mode = "single"

    @property
    def no_data(self):
        """
        :return: Nodata Object, set various nodata properties
        """
        return self._nodata

    def _convert_to_json(self):
        if self.item.type == "mapWidget":
            wlayer = self.item.layers[self.layer]
            widget_id = self.item._id
            layer_id = wlayer["id"]
            self._datasource = {"id": str(widget_id) + "#" + str(layer_id)}
        else:
            self._datasource = {
                "type": "featureServiceDataSource",
                "itemId": self.item.itemid,
                "layerId": 0,
                "table": True,
            }
        json_data = {
            "type": "listWidget",
            "iconType": self._list_icon,
            "selectionMode": self._selection_mode,
            "separatorColor": self._seperator_color,
            "selectionColor": self._selection_color,
            "selectionTextColor": self._selection_text_color,
            "events": [],
            "selectionMode": "multi",
            "datasets": [
                {
                    "type": "serviceDataset",
                    "dataSource": self._datasource,
                    "outFields": ["*"],
                    "groupByFields": [],
                    "orderByFields": [],
                    "statisticDefinitions": [],
                    "maxFeatures": self._max_features,
                    "querySpatialRelationship": "esriSpatialRelIntersects",
                    "returnGeometry": False,
                    "clientSideStatistics": False,
                    "name": "main",
                }
            ],
            "id": self._id,
            "name": self.name,
            "showLastUpdate": self.show_last_update,
            "noDataVerticalAlignment": self._nodata.alignment,
            "showCaptionWhenNoData": self._nodata.show_title,
            "showDescriptionWhenNoData": self._nodata.show_description,
        }
        if self.events.enable:
            json_data["events"].append(
                {"type": self.events.type, "actions": self.events.synced_widgets}
            )
            json_data["selectionMode"] = self.events.selection_mode

        return json_data


class Events(object):
    @classmethod
    def _create_events(cls, enable=False):
        events = Events()

        events._enable = False
        events._selection_mode = "single"
        events._type = "selectionChanged"
        events._actions = []

        events.enable = enable

        return events

    @property
    def selection_mode(self):
        """
        :return: Selection mode of events.
        """
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, value):
        """
        Set Selection mode of events.
        """
        if value in ["single", "multi"]:
            self._selection_mode = value
        else:
            raise Exception("Please specify selection_mode from 'single' and 'multi'")

    @property
    def enable(self):
        """
        :return: if events are enabled or not.
        """
        return self._enable

    @enable.setter
    def enable(self, value):
        """
        Set if events are enabled or not.
        """
        self._enable = bool(value)

    @property
    def type(self):
        """
        :return: type of trigger for event.
        """
        return self._type

    @property
    def synced_widgets(self):
        """
        :return: List of synced widgets.
        """
        return self._actions

    def sync_map(self, action_type, widget):
        """
        Synchronize a mapWidget with List for triggered events.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        action_type                 Required string. Actions can be one of
                                    "zoom", "flash", "show_popup", "pan".
        -------------------------   -------------------------------------------
        widget                      Required MapWidget item. Name of the map
                                    widget.
        =========================   ===========================================
        """
        if self.enable == False:
            raise Exception("Please enable events")
        else:
            if widget.type == "mapWidget":
                if action_type in ["zoom", "flash", "show_popup", "pan"]:
                    self._actions.append({"type": action_type, "targetId": widget._id})
                elif action_type == "filter":
                    if self._targetid is not None:
                        self._actions.append(
                            {
                                "type": action_type,
                                "by": "whereClause",
                                "targetId": self._targetid,
                            }
                        )
                    else:
                        raise Exception(
                            "This operation is not suitable for given dataSource."
                        )
                else:
                    raise Exception(
                        "Please select action_type from 'zoom', 'flash', 'show_popup' and 'pan'"
                    )
            else:
                raise Exception("Please select a map widget")

    def sync_widget(self, widgets):
        """
        Synchronize non-mapWidget type widgets with List for triggered events.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        widget                      Required widget item or list of widget items
                                    .Name of the widgets to be synced.
        =========================   ===========================================
        """
        if self.enable == False:
            raise Exception("Please enable events")

        else:
            if isinstance(widgets, list):
                for widget in widgets:
                    if widget.type == "mapWidget":
                        raise Exception(
                            "Use sync_map method to add actions for map widgets"
                        )  ##duplicate or erase
                    else:
                        action_type = "filter"
                        widget_id = str(widget._id) + "#main"
                        self._actions.append(
                            {
                                "type": action_type,
                                "by": "whereClause",
                                "targetId": widget_id,
                            }
                        )
            else:
                if widgets.type == "mapWidget":
                    raise Exception(
                        "Use sync_map method to add actions for map widgets"
                    )  ##duplicate or erase
                else:
                    action_type = "filter"
                    widget_id = str(widgets._id) + "#main"
                    self._actions.append(
                        {
                            "type": action_type,
                            "by": "whereClause",
                            "targetId": widget_id,
                        }
                    )

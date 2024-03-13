import atexit
import json
import arcgis
from arcgis._impl.common._deprecate import deprecated

_DASHBOARD_VERSION = 27

_created_dashboards = []


@deprecated(deprecated_in="2.0.1", removed_in=None, current_version="2.0.1")
class Dashboard(object):
    """
    Creates a :class:`~arcgis.apps.dashboard.Dashboard` Object.

    :return:
        :class:`~arcgis.apps.dashboard.Dashboard` object
    """

    def __init__(self):
        self.elements = []

        self._theme = "light"

        self._header = None
        self._side_panel = None
        self._layout = {}
        self._widgets = []

        # if item is not None:
        #     return _from_dashboard(item)

    def save(
        self, title, description="", summary="", tags=None, gis=None, overwrite=False
    ):
        """
        Publishes a Dashboard Object.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        title                       Required string. Title or Caption for the
                                    Dashboard.
        -------------------------   -------------------------------------------
        description                 Optional string. Description for the Dashboard.
        -------------------------   -------------------------------------------
        summary                     Optional string. Summary of the Dashboard.
        -------------------------   -------------------------------------------
        tags                        Optional string. Comma separated tags.
        -------------------------   -------------------------------------------
        gis                         Optional :class:`~arcgis.gis.GIS` to publish dashboard.
                                    By default uses active gis.
        -------------------------   -------------------------------------------
        overwrite                   Optional Boolean.
                                    Overwrite existing dashboard.
        =========================   ===========================================
        """
        if not title:
            raise Exception("Please specify a title.")

        self.title = title
        self.description = description

        self.summary = summary
        self.tags = tags if tags else ""
        return self._publish(gis=gis, overwrite=overwrite)

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        self._theme = value
        if value not in ["light", "dark"]:
            self._theme = "light"

    @property
    def header(self):
        """
        :return:
            :class:`~arcgis.apps.dashboard.Header` object
        """
        return self._header

    @header.setter
    def header(self, value):
        """
        Set the header object
        """
        self._header = value

    @property
    def side_panel(self):
        """
        :return:
            :class:`~arcgis.apps.dashboard.SidePanel` object
        """
        return self._side_panel

    @side_panel.setter
    def side_panel(self, value):
        """
        Set the Side Panel object
        """
        self._side_panel = value

    def _convert_to_json(self):
        json_data = {
            "version": _DASHBOARD_VERSION,
            "widgets": [element._convert_to_json() for element in self.elements],
            "settings": {"maxPaginationRecords": 50000, "allowElementResizing": False},
            "mapOverrides": {"trackedFeatureRadius": 60},
            "theme": self.theme,
            "themeOverrides": {},
            "numberPrefixOverrides": [],
            "layout": self._layout,
            "authoringApp": "ArcGIS API for Python",
            "authoringAppVersion": arcgis.__version__,
            "typeKeywords": "Python",
        }

        if self.header:
            json_data["headerPanel"] = self.header._convert_to_json()

        if self.side_panel:
            json_data["leftPanel"] = self.side_panel._convert_to_json()

        # print(json_data)
        return json_data

    @property
    def layout(self):
        """
        :return: Layout of the dashboard
        """
        return self._layout

    @layout.setter
    def layout(self, value):
        """
        Set the layout of the dashboard, using add_row and add_column functions.
        """
        self.elements = value["widgets"]
        del value["widgets"]
        self._layout = {"rootElement": value}

    # @property
    # def widgets(self):
    #     """
    #     :return: widgets of the dashboard
    #     """
    #     return self._widgets

    def _repr_html_(self):
        url = self._dash_publish()
        return f"""<iframe src={url} width=900 height=300>"""

    def _dash_publish(self):
        gis = arcgis.env.active_gis
        import random
        import string

        letters = string.ascii_lowercase
        title = "".join(random.choice(letters) for i in range(10))
        summary = "".join(random.choice(letters) for i in range(10))

        db = Dashboard()

        db.title = title
        db.summary = summary
        db.description = ""
        db.summary = summary
        db.tags = ""

        db._layout = self._layout
        db.elements = self.elements
        db = db._publish(gis)
        _created_dashboards.append((gis, db))
        url = f"{gis.url}/apps/opsdashboard/index.html#/{db.itemid}"

        return url

    def _publish(self, gis=None, overwrite=False):
        if gis is None:
            gis = arcgis.env.active_gis

        items = gis.content.search(f"title:{self.title}", "Dashboard")
        for item in items:
            if item.title.lower() == self.title.lower() and overwrite is False:
                raise Exception(
                    "A dashboard with same name already exists, to continue set `overwrite` = True"
                )
            elif item.title.lower() == self.title.lower():
                item.delete(force=True)

        return gis.content.add(
            {
                "type": "Dashboard",
                "description": self.description,
                "title": self.title,
                "overwrite": str(overwrite).lower(),
                "text": json.dumps(self._convert_to_json()),
            }
        )

    def _from_dashboard(self, dashboard_item):
        widget_map = {
            "indicatorWidget": arcgis.apps.dashboard.Indicator,
            "gaugeWidget": arcgis.apps.dashboard.Gauge,
            "pieChartWidget": arcgis.apps.dashboard.PieChart,
            "serialChartWidget": arcgis.apps.dashboard.SerialChart,
            "detailsWidget": arcgis.apps.dashboard.Details,
            "richTextWidget": arcgis.apps.dashboard.RichText,
            "listWidget": arcgis.apps.dashboard.List,
            "embeddedContentWidget": arcgis.apps.dashboard.EmbeddedContent,
            "legendWidget": arcgis.apps.dashboard.MapLegend,
        }
        item_json = dashboard_item.get_data()
        for widget_json in item_json["widgets"]:
            type = widget_json["type"]
            self._widgets.append(widget_map[type]._from_json(widget_json))

        return dashboard_item

    @staticmethod
    def _publish_random(widget):
        from arcgis.apps.dashboard import add_row

        gis = arcgis.env.active_gis
        import random
        import string

        letters = string.ascii_lowercase
        title = "".join(random.choice(letters) for i in range(10))
        summary = "".join(random.choice(letters) for i in range(10))
        db = Dashboard()

        if widget.type == "headerPanel":
            db.header = widget
        elif widget.type == "leftPanel":
            db.side_panel = widget
        elif widget.type == "legendWidget":
            map_widget_width = widget._map_widget.width
            widget._map_widget.width = 0.5
            db.layout = add_row([widget._map_widget, widget])
            widget._map_widget.width = map_widget_width
        else:
            if (
                hasattr(widget, "item")
                and getattr(widget.item, "type", None) == "mapWidget"
            ):
                db.layout = add_row([widget.item, widget])
            else:
                db.layout = add_row([widget])

        db = db.save(title, summary, gis=gis, overwrite=True)
        _created_dashboards.append((gis, db))
        url = f"{gis.url}/apps/opsdashboard/index.html#/{db.itemid}"

        return url


@atexit.register
def _delete_objects():
    for dashboard in _created_dashboards:
        gis = dashboard[0]
        db = dashboard[1]
        db.delete(force=True)

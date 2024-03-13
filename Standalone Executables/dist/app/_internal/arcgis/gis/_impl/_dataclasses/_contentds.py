from __future__ import annotations
from arcgis.auth.tools._lazy import LazyLoader

from typing import Any
from dataclasses import dataclass, field
from enum import Enum

arcgis = LazyLoader("arcgis")

__all__ = [
    "ItemTypeEnum",
    "ItemProperties",
    "CreateServiceParameter",
    "MetadataFormatEnum",
    "ServiceTypeEnum",
]
###########################################################################


def _parse_enum(value: Enum | Any | None) -> Any | None:
    """returns the Enum's value or the current value"""
    if isinstance(value, Enum):
        return value.value
    else:
        return value


###########################################################################
class ItemTypeEnum(Enum):
    VR_EXPERIENCE = "360 VR Experience"
    CITYENGINE_WEB_SCENE = "CityEngine Web Scene"
    MAP_AREA = "Map Area"
    PRO_MAP = "Pro Map"
    WEB_MAP = "Web Map"
    WEB_SCENE = "Web Scene"
    FEATURE_COLLECTION = "Feature Collection"
    FEATURE_COLLECTION_TEMPLATE = "Feature Collection Template"
    FEATURE_SERVICE = "Feature Service"
    GEODATA_SERVICE = "Geodata Service"
    GLOBE_SERVICE = "Globe Service"
    IMAGE_SERVICE = "Image Service"
    KML = "KML"
    KML_COLLECTION = "KML Collection"
    MAP_SERVICE = "Map Service"
    OGCFEATURESERVER = "OGCFeatureServer"
    ORIENTED_IMAGERY_CATALOG = "Oriented Imagery Catalog"
    RELATIONAL_DATABASE_CONNECTION = "Relational Database Connection"
    SCENE_SERVICE = "Scene Service"
    VECTOR_TILE_SERVICE = "Vector Tile Service"
    WFS = "WFS"
    WMS = "WMS"
    WMTS = "WMTS"
    GEOMETRY_SERVICE = "Geometry Service"
    GEOCODING_SERVICE = "Geocoding Service"
    GEOPROCESSING_SERVICE = "Geoprocessing Service"
    NETWORK_ANALYSIS_SERVICE = "Network Analysis Service"
    WORKFLOW_MANAGER_SERVICE = "Workflow Manager Service"
    APPBUILDER_EXTENSION = "AppBuilder Extension"
    APPBUILDER_WIDGET_PACKAGE = "AppBuilder Widget Package"
    CODE_ATTACHMENT = "Code Attachment"
    DASHBOARD = "Dashboard"
    DEEP_LEARNING_STUDIO_PROJECT = "Deep Learning Studio Project"
    ESRI_CLASSIFICATION_SCHEMA = "Esri Classification Schema"
    EXCALIBUR_IMAGERY_PROJECT = "Excalibur Imagery Project"
    EXPERIENCE_BUILDER_WIDGET = "Experience Builder Widget"
    EXPERIENCE_BUILDER_WIDGET_PACKAGE = "Experience Builder Widget Package"
    FORM = "Form"
    GEOBIM_APPLICATION = "GeoBIM Application"
    GEOBIM_PROJECT = "GeoBIM Project"
    HUB_EVENT = "Hub Event"
    HUB_INITIATIVE = "Hub Initiative"
    HUB_INITIATIVE_TEMPLATE = "Hub Initiative Template"
    HUB_PAGE = "Hub Page"
    HUB_PROJECT = "Hub Project"
    HUB_SITE_APPLICATION = "Hub Site Application"
    INSIGHTS_WORKBOOK = "Insights Workbook"
    INSIGHTS_WORKBOOK_PACKAGE = "Insights Workbook Package"
    INSIGHTS_MODEL = "Insights Model"
    INSIGHTS_PAGE = "Insights Page"
    INSIGHTS_THEME = "Insights Theme"
    INSIGHTS_DATA_ENGINEERING_WORKBOOK = "Insights Data Engineering Workbook"
    INSIGHTS_DATA_ENGINEERING_MODEL = "Insights Data Engineering Model"
    INVESTIGATION = "Investigation"
    MISSION = "Mission"
    MOBILE_APPLICATION = "Mobile Application"
    NOTEBOOK = "Notebook"
    NATIVE_APPLICATION = "Native Application"
    NATIVE_APPLICATION_INSTALLER = "Native Application Installer"
    OPERATION_VIEW = "Operation View"
    OPERATIONS_DASHBOARD_ADD_IN = "Operations Dashboard Add In"
    OPERATIONS_DASHBOARD_EXTENSION = "Operations Dashboard Extension"
    ORTHO_MAPPING_PROJECT = "Ortho Mapping Project"
    ORTHO_MAPPING_TEMPLATE = "Ortho Mapping Template"
    SOLUTION = "Solution"
    STORYMAP = "StoryMap"
    WEB_APPBUILDER_WIDGET = "Web AppBuilder Widget"
    WEB_EXPERIENCE = "Web Experience"
    WEB_EXPERIENCE_TEMPLATE = "Web Experience Template"
    WEB_MAPPING_APPLICATION = "Web Mapping Application"
    WORKFORCE_PROJECT = "Workforce Project"
    ADMINISTRATIVE_REPORT = "Administrative Report"
    APACHE_PARQUET = "Apache Parquet"
    CAD_DRAWING = "CAD Drawing"
    COLOR_SET = "Color Set"
    CONTENT_CATEGORY_SET = "Content Category Set"
    CSV = "CSV"
    DOCUMENT_LINK = "Document Link"
    EARTH_CONFIGURATION = "Earth configuration"
    ESRI_CLASSIFIER_DEFINITION = "Esri Classifier Definition"
    EXPORT_PACKAGE = "Export Package"
    FILE_GEODATABASE = "File Geodatabase"
    GEOJSON = "GeoJson"
    GEOPACKAGE = "GeoPackage"
    GML = "GML"
    IMAGE = "Image"
    IWORK_KEYNOTE = "iWork Keynote"
    IWORK_NUMBERS = "iWork Numbers"
    IWORK_PAGES = "iWork Pages"
    MICROSOFT_EXCEL = "Microsoft Excel"
    MICROSOFT_POWERPOINT = "Microsoft Powerpoint"
    MICROSOFT_WORD = "Microsoft Word"
    PDF = "PDF"
    REPORT_TEMPLATE = "Report Template"
    SERVICE_DEFINITION = "Service Definition"
    SHAPEFILE = "Shapefile"
    SQLITE_GEODATABASE = "SQLite Geodatabase"
    STATISTICAL_DATA_COLLECTION = "Statistical Data Collection"
    STORYMAP_THEME = "StoryMap Theme"
    STYLE = "Style"
    SYMBOL_SET = "Symbol Set"
    VISIO_DOCUMENT = "Visio Document"
    ARCPAD_PACKAGE = "ArcPad Package"
    COMPACT_TILE_PACKAGE = "Compact Tile Package"
    EXPLORER_LAYER = "Explorer Layer"
    IMAGE_COLLECTION = "Image Collection"
    LAYER = "Layer"
    LAYER_PACKAGE = "Layer Package"
    PRO_REPORT = "Pro Report"
    SCENE_PACKAGE = "Scene Package"
    MOBILE_SCENE_PACKAGE = "Mobile Scene Package"
    PROJECT_PACKAGE = "Project Package"
    PROJECT_TEMPLATE = "Project Template"
    PUBLISHED_MAP = "Published Map"
    SCENE_DOCUMENT = "Scene Document"
    TASK_FILE = "Task File"
    TILE_PACKAGE = "Tile Package"
    VECTOR_TILE_PACKAGE = "Vector Tile Package"
    WINDOWS_MOBILE_PACKAGE = "Windows Mobile Package"
    DESKTOP_STYLE = "Desktop Style"
    ARCGIS_PRO_CONFIGURATION = "ArcGIS Pro Configuration"
    DEEP_LEARNING_PACKAGE = "Deep Learning Package"
    GEOPROCESSING_PACKAGE = "Geoprocessing Package"
    GEOPROCESSING_PACKAGE_PRO_VERSION = "Geoprocessing Package (Pro version)"
    GEOPROCESSING_SAMPLE = "Geoprocessing Sample"
    LOCATOR_PACKAGE = "Locator Package"
    RASTER_FUNCTION_TEMPLATE = "Raster function template"
    RULE_PACKAGE = "Rule Package"
    ARCGIS_PRO_ADD_IN = "ArcGIS Pro Add In"
    CODE_SAMPLE = "Code Sample"
    DESKTOP_ADD_IN = "Desktop Add In"
    DESKTOP_APPLICATION = "Desktop Application"
    DESKTOP_APPLICATION_TEMPLATE = "Desktop Application Template"
    EXPLORER_ADD_IN = "Explorer Add In"
    SURVEY123_ADD_IN = "Survey123 Add In"
    WORKFLOW_MANAGER_PACKAGE = "Workflow Manager Package"


###########################################################################
class MetadataFormatEnum(Enum):
    FGDB = "fgdb"
    INSPIRE = "inspire"
    ISO19139 = "iso19139"
    ISO19139_32 = "iso19139-3.2"
    ISO19115 = "iso19115"


###########################################################################
class ServiceTypeEnum(Enum):
    FEATURE_SERVICE = "featureService"
    IMAGE_SERVICE = "imageService"
    RELATIONSHIP_SERVICE = "relationalCatalogService"


###########################################################################
@dataclass
class ItemProperties:
    """
    Item parameters correspond to properties of an item that are available
    to update on the :meth:`~arcgis.gis.ContentManager.add` and
    :meth:`~arcgis.gis.Item.update` operations.
    """

    title: str
    item_type: ItemTypeEnum | str
    tags: list[str] | None = None
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    metadata: str | None = None
    metadata_editable: bool | None = None
    metadata_formats: MetadataFormatEnum | str | None = MetadataFormatEnum.ISO19139
    type_keywords: list[str] | None = None
    description: str | None = None
    snippet: str | None = None
    extent: str | list = None
    spatial_reference: str | None = None
    access_information: str | None = None
    license_info: str | None = None
    culture: str | None = None
    properties: dict | None = None
    app_categories: list[str] | None = None
    industries: list[str] | None = None
    listing_properties: dict | None = None
    service_username: str | None = None
    service_password: str | None = None
    service_proxy: dict | None = None
    categories: list[str] | None = None
    text: dict | str | None = None
    extension: str | None = None
    _dict_data: dict | None = field(init=False)

    def __str__(self):
        return f"<ItemProperties: title={self.title}, type={self.item_type}>"

    def __repr__(self):
        return self.__str__()

    def __post_init__(self):
        self._dict_data = {
            "title": self.title,
            "type": _parse_enum(self.item_type),
            "tags": ",".join(self.tags or []),
            "thumbnail": self.thumbnail,
            "thumbnailurl": self.thumbnail_url,
            "metadata": self.metadata,
            "metadataEditable": self.metadata_editable,
            "metadataFormats": _parse_enum(self.metadata_formats),
            "typeKeywords": ",".join(self.type_keywords or []),
            "description": self.description or "",
            "snippet": self.snippet,
            "extent": self.extent,
            "spatialReference": self.spatial_reference or "",
            "accessInformation": self.access_information,
            "licenseInfo": self.license_info,
            "culture": self.culture,
            "properties": self.properties,
            "appCategories": ",".join(self.app_categories or []),
            "industries": ",".join(self.industries or []),
            "listingProperties": self.listing_properties,
            "serviceUsername": self.service_username,
            "servicePassword": self.service_password,
            "serviceProxyFilter": self.service_proxy,
            "categories": ",".join(self.categories or []),
            "text": self.text or None,
            "extension": self.extension or None,
        }

    def to_dict(self):
        return {
            "title": self.title,
            "type": _parse_enum(self.item_type),
            "tags": ",".join(self.tags or []),
            "thumbnail": self.thumbnail,
            "thumbnailurl": self.thumbnail_url,
            "metadata": self.metadata,
            "metadataEditable": self.metadata_editable,
            "metadataFormats": _parse_enum(self.metadata_formats),
            "typeKeywords": ",".join(self.type_keywords or []),
            "description": self.description or "",
            "snippet": self.snippet,
            "extent": self.extent,
            "spatialReference": self.spatial_reference or "",
            "accessInformation": self.access_information,
            "licenseInfo": self.license_info,
            "culture": self.culture,
            "properties": self.properties,
            "appCategories": ",".join(self.app_categories or []),
            "industries": ",".join(self.industries or []),
            "listingProperties": self.listing_properties,
            "serviceUsername": self.service_username,
            "servicePassword": self.service_password,
            "serviceProxyFilter": self.service_proxy,
            "categories": ",".join(self.categories or []),
            "text": self.text or None,
            "extension": self.extension or None,
        }

    @classmethod
    def fromitem(cls, item: arcgis.gis.Item) -> ItemProperties:
        """Loads the ItemProperties from an `Item`"""
        return ItemProperties(
            title=item.title,
            item_type=ItemTypeEnum._value2member_map_[item.type],
            tags=item.tags,
            type_keywords=item.typeKeywords,
            description=item.description,
            snippet=item.snippet,
            extent=item.extent,
            spatial_reference=item.spatialReference,
            access_information=item.accessInformation,
            license_info=item.licenseInfo,
            culture=item.culture,
            properties=item.properties,
            app_categories=item.appCategories,
            service_proxy=item.proxyFilter,
            industries=item.industries,
            categories=item.categories,
        )


###########################################################################
@dataclass
class CreateServiceParameter:
    """

    The create service parameter description.

    =======================    =============================================================
    **Parameter**               **Description**
    -----------------------    -------------------------------------------------------------
    name                       Required String. Name of the Service
    -----------------------    -------------------------------------------------------------
    output_type                Required :class:`~arcgis.gis._impl._dataclasses.ServiceTypeEnum`
                               or string. The type of service to create.
    -----------------------    -------------------------------------------------------------
    service_description        Optional String. Description given to the service.
    -----------------------    -------------------------------------------------------------
    is_view                    Optional Bool. Specifies if the request is generating a view. The default is False.
    -----------------------    -------------------------------------------------------------
    description                Optional String. A user-friendly description for the published dataset.
    -----------------------    -------------------------------------------------------------
    tags                       Optional String. The tags for the resulting Item
    -----------------------    -------------------------------------------------------------
    snippet                    Optional String. A short description of the Item.
    -----------------------    -------------------------------------------------------------
    static_data                Optional Boolean. Boolean value indicating whether the data changes.
    -----------------------    -------------------------------------------------------------
    max_record_count           Optional Integer. An integer value indicating any constraints enforced on query operations.
    -----------------------    -------------------------------------------------------------
    query_formats              Optional String. The formats in which query results are returned. The default is JSON.
    -----------------------    -------------------------------------------------------------
    capabilities               Optional String. Specify the service capabilities.
    -----------------------    -------------------------------------------------------------
    copyright_text             Optional String. Copyright information associated with the dataset.
    -----------------------    -------------------------------------------------------------
    spatial_reference          Optional Dictionary. All layers added to a hosted feature service need to have the same
                               spatial reference defined for the feature service. When creating a new
                               empty service without specifying its spatial reference, the spatial
                               reference of the hosted feature service is set to the first layer added
                               to that feature service.
    -----------------------    -------------------------------------------------------------
    initial_extent             Optional Dictionary. The initial extent set for the service.
    -----------------------    -------------------------------------------------------------
    allow_geometry_updates     Optional Boolean. Boolean value indicating if updating the geometry of the service is permitted.
    -----------------------    -------------------------------------------------------------
    units                      Optional String. Units used by the feature service.
    -----------------------    -------------------------------------------------------------
    xssPreventionInfo          Optional Dictionary. A JSON object specifying the properties of cross-site scripting prevention.
    -----------------------    -------------------------------------------------------------
    overwrite                  Optional Boolean. Overwrite an existing service when true. The default is False
    -----------------------    -------------------------------------------------------------
    itemid_to_create           Optional String. The item id to create.  This is only valid on ArcGIS Enterprise.
    =======================    =============================================================
    """

    name: str
    output_type: ServiceTypeEnum | str = ServiceTypeEnum.FEATURE_SERVICE
    itemid_to_create: str | None = None
    overwrite: bool = False
    service_description: str | None = ""
    is_view: bool = False
    description: str | None = None
    tags: list[str] | None = None
    snippet: str | None = None
    static_data: bool = True
    max_record_count: int = 1000
    query_formats: str = "JSON"
    capabilities: str = "Query"
    copyright_text: str | None = None
    spatial_reference: dict | None = None
    initial_extent: dict | None = None
    allow_geometry_updates: bool | None = None
    units: str | None = None
    xssPreventionInfo: dict | None = None
    provider: str | None = None
    connection: dict | None = None
    _dict_data: dict | None = field(init=False)

    def __str__(self):
        return "<CreateServiceParameter>"

    def __repr__(self):
        return "<CreateServiceParameter>"

    def __post_init__(self):
        self._dict_data = self._create_dict()

    def _create_dict(self):
        default_sr = {"wkid": 102100}
        init_extent = {
            "xmin": -20037507.0671618,
            "ymin": -30240971.9583862,
            "xmax": 20037507.0671618,
            "ymax": 18398924.324645,
            "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        }
        init_xss = {
            "xssPreventionEnabled": True,
            "xssPreventionRule": "input",
            "xssInputRule": "rejectInvalid",
        }
        if (
            _parse_enum(self.output_type) == "imageService"
            and self.capabilities is None
        ):
            self.capabilities = "Image,Catalog,Mensuration"
        self._dict_data = {
            "outputType": _parse_enum(self.output_type),
            "isView": self.is_view,
            "description": self.description or "",
            "snippet": self.snippet or "",
            "tags": ",".join(self.tags or []),
            "itemIdToCreate": self.itemid_to_create or None,
            "createParameters": {
                "name": self.name,
                "serviceDescription": self.service_description,
                "hasStaticData": self.static_data,
                "maxRecordCount": self.max_record_count,
                "supportedQueryFormats": self.query_formats or "JSON",
                "capabilities": self.capabilities or "Query",
                "copyRightText": self.copyright_text or "",
                "spatialReference": self.spatial_reference or default_sr,
                "initialExtent": self.initial_extent or init_extent,
                "allowGeometryUpdates": self.allow_geometry_updates,
                "units": self.units or "esriMeters",
                "xssPreventionInfo": self.xssPreventionInfo or init_xss,
            },
        }
        if _parse_enum(self.output_type) == "relationalCatalogService":
            self._dict_data["createParameters"]["provider"] = "ADS"
            self._dict_data["createParameters"][
                "connectionProperties"
            ] = self.connection

        if self._dict_data["itemIdToCreate"] in [None, ""]:
            del self._dict_data["itemIdToCreate"]
        return self._dict_data

    def to_dict(self) -> dict:
        return self._create_dict()

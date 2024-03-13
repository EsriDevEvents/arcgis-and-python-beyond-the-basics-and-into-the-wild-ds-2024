"""
Classes for administering your GIS.

The gis.admin property is dynamically set at runtime based on what kind of GIS (ArcGIS Enterprise or ArcGIS Online) an
administrator connects to.
For ArcGIS Online GIS, administrators will get an instance of AGOLAdminManager from the gis.admin property.
For ArcGIS Enterprise GIS , administrators will get an instance of PortalAdminManager from the gis.admin property.
"""
from .portaladmin import PortalAdminManager
from .agoladmin import AGOLAdminManager
from ._federation import Federation
from ._logs import Logs
from ._license import LicenseManager, License, Bundle
from ._livingatlas import LivingAtlas
from ._machines import Machines, Machine
from ._metadata import MetadataManager
from ._security import EnterpriseGroups, EnterpriseUsers, OAuth
from ._security import Security, SSLCertificate, SSLCertificates
from ._site import Site
from ._socialproviders import SocialProviders
from ._system import Directory, Licenses, System, Indexer, EmailManager
from ._system import PortalLicense
from ._system import WebAdaptor, WebAdaptors
from ._collaboration import Collaboration, CollaborationManager
from ._ux import (
    UX,
    MapSettings,
    HomePageSettings,
    ItemSettings,
    SecuritySettings,
    StockImage,
)
from ._creditmanagement import CreditManager
from ._security import PasswordPolicy
from ._resources import PortalResourceManager
from ._catagoryschema import CategoryManager
from ._idp import IdentityProviderManager
from ._wh import WebhookManager, Webhook
from ._usage import AGOLUsageReports
from ._dsmgr import (
    DataStoreMetricsManager,
    DataStoreAggregation,
    DataStoreTimeUnit,
    DataStoreMetric,
)

__all__ = ["PortalAdminManager", "AGOLAdminManager"]

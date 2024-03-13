from __future__ import annotations
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap
from arcgis.geocoding import geocode
from arcgis.apps.hub.initiatives import InitiativeManager, Initiative
from arcgis.apps.hub.events import EventManager, Event
from arcgis.apps.hub.sites import SiteManager, Site
from arcgis.apps.hub.pages import PageManager, Page
from datetime import datetime
from collections import OrderedDict
import json
from functools import wraps


def _lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    # http://stevenloria.com/lazy-evaluated-properties-in-python/
    attr_name = "_lazy_" + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class Hub(object):
    """
    Entry point into the Hub module. Lets you access an individual hub and its components.


    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    url                 Required string. If no URL is provided by user while connecting
                        to the GIS, then the URL will be ArcGIS Online.
    ----------------    ---------------------------------------------------------------
    GIS                 Required authenticated GIS object for the ArcGIS Online
                        organization associated with your Hub.
    ================    ===============================================================

    """

    def __init__(self, gis: GIS):
        self.gis = gis
        try:
            self._gis_id = self.gis.properties.id
        except AttributeError:
            self._gis_id = None

    @property
    def _hub_enabled(self):
        """
        Returns True if Hub Premium is enabled on this org
        """
        try:
            self.gis.properties.portalProperties["hub"]["enabled"]
            return True
        except:
            return False

    @property
    def enterprise_org_id(self) -> str:
        """
        Returns the AGOL org id of the Enterprise Organization associated with this Premium Hub.
        """

        if self._hub_enabled:
            try:
                _e_org_id = (
                    self.gis.properties.portalProperties.hub.settings.enterpriseOrg.orgId
                )
                return _e_org_id
            except AttributeError:
                try:
                    if (
                        self.gis.properties.subscriptionInfo.companionOrganizations.type
                        == "Enterprise"
                    ):
                        return "Enterprise org id is not available"
                except:
                    return self._gis_id
        else:
            raise Exception("Hub does not exist or is inaccessible.")

    @property
    def community_org_id(self) -> str:
        """
        Returns the AGOL org id of the Community Organization associated with this Premium Hub.
        """
        if self._hub_enabled:
            try:
                _c_org_id = (
                    self.gis.properties.portalProperties.hub.settings.communityOrg.orgId
                )
                return _c_org_id
            except AttributeError:
                try:
                    if (
                        self.gis.properties.subscriptionInfo.companionOrganizations.type
                        == "Community"
                    ):
                        return "Community org id is not available"
                except:
                    return self._gis_id
        else:
            raise Exception("Hub does not exist or is inaccessible.")

    @property
    def enterprise_org_url(self) -> str:
        """
        Returns the AGOL org url of the Enterprise Organization associated with this Premium Hub.
        """
        if self._hub_enabled:
            try:
                self.gis.properties.portalProperties.hub.settings.enterpriseOrg
                try:
                    _url = self.gis.properties.publicSubscriptionInfo.companionOrganizations[
                        0
                    ][
                        "organizationUrl"
                    ]
                except:
                    _url = self.gis.properties.subscriptionInfo.companionOrganizations[
                        0
                    ]["organizationUrl"]
                return "https://" + _url
            except AttributeError:
                return self.gis.url
        else:
            raise Exception("Hub does not exist or is inaccessible.")

    @property
    def community_org_url(self) -> str:
        """
        Returns the AGOL org id of the Community Organization associated with this Premium Hub.
        """
        if self._hub_enabled:
            try:
                self.gis.properties.portalProperties.hub.settings.communityOrg
                try:
                    _url = self.gis.properties.publicSubscriptionInfo.companionOrganizations[
                        0
                    ][
                        "organizationUrl"
                    ]
                except:
                    _url = self.gis.properties.subscriptionInfo.companionOrganizations[
                        0
                    ]["organizationUrl"]
                return "https://" + _url
            except AttributeError:
                return self.gis.url
        else:
            raise Exception("Hub does not exist or is inaccessible.")

    @_lazy_property
    def initiatives(self):
        """
        The resource manager for Hub initiatives. See :class:`~arcgis.apps.hub.InitiativeManager`.
        """
        if self._hub_enabled:
            return InitiativeManager(self)
        else:
            raise Exception(
                "Initiatives are only available with Hub Premium. Please upgrade to Hub Premium to use this feature."
            )

    @_lazy_property
    def events(self):
        """
        The resource manager for Hub events. See :class:`~arcgis.apps.hub.EventManager`.
        """
        if self._hub_enabled:
            return EventManager(self)
        else:
            raise Exception(
                "Events is only available with Hub Premium. Please upgrade to Hub Premium to use this feature."
            )

    @_lazy_property
    def sites(self):
        """
        The resource manager for Hub sites. See :class:`~hub.sites.SiteManager`.
        """
        return SiteManager(self)

    @_lazy_property
    def pages(self):
        """
        The resource manager for Hub pages. See :class:`~hub.sites.PageManager`.
        """
        return PageManager(self)

from __future__ import annotations
from arcgis.gis.kubernetes._admin._base import _BaseKube
from arcgis.gis import GIS
from typing import Dict, Any


class ExternalContentManager(_BaseKube):
    """
    Provides management of the external content resources.
    """

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        if url.lower().endswith("/content") == False:
            url += "/content"
        self._url = url
        self._gis = gis
        self._con = gis._con

    @property
    def external_content(self) -> dict[str, Any]:
        """
        The external_content resource returns whether access to external
        content has been enabled or disabled for an organization. If the
        resource returns true, an organization's Esri content will contain
        external URLs that reference sites and resources hosted outside of
        the organization. If the resource returns false, Esri content
        containing external URLs will be removed from the organization. Any
        Esri content remaining after external content is disabled will not
        contain external URLs. If you are configuring ArcGIS Enterprise on
        Kubernetes in an environment in which no internet connection is
        available, or internet access is prohibited, access to external
        content should be disabled to avoid discovering content containing
        inaccessible URLs to external sites.

        :return: dict[str,Any]
        """
        url: str = f"{self._url}/externalcontent"
        params: dict[str, Any] = {
            "f": "json",
        }
        return self._con.get(url, params)

    def update(self, value: bool) -> dict[str, Any]:
        """
        The update operation enables and disables access to Esri-provided
        content containing external URLs that reference sites and resources
        hosted outside of the organization.

        :returns: dict[str,Any]
        """
        url: str = f"{self._url}/externalcontent/update"
        params: dict[str, Any] = {
            "f": "json",
            "externalContentEnabled": value,
        }
        return self._con.post(url, params)


###########################################################################
class LanguageManager(_BaseKube):
    """
    Provides management of the language resources. The languages resource
    provides a list of current languages for an organization.

    """

    _gis = None
    _con = None
    _properties = None
    _url = None

    def __init__(self, url: str, gis: GIS):
        super()
        if url.lower().endswith("/languages") == False:
            url += "/languages"
        self._url = url
        self._gis = gis
        self._con = gis._con

    @property
    def languages(self) -> Dict[str, bool]:
        """
        This resource returns a list of all Esri supported languages and
        their associated language codes, as well as the current status of
        the language, either enabled (true) or disabled (false). After your
        organization has been configured, English (language code en) will
        be the only enabled language by default. Additional languages can
        be enabled using the Add operation. Once enabled, the Esri provided
        content for these languages will be searchable and accessible to
        users in your organization. Enabled languages can be disabled using
        the Remove operation, which will remove their Esri provided content
        from the organization. Note that at least one language must always
        be enabled for your organization.

        :return: Dict[str, bool]
        """
        params = {"f": "json"}
        return self._con.get(self._url, params)

    @languages.setter
    def languages(self, languages: Dict[str, bool]):
        """
        This resource returns a list of all Esri supported languages and
        their associated language codes, as well as the current status of
        the language, either enabled (true) or disabled (false). After your
        organization has been configured, English (language code en) will
        be the only enabled language by default. Additional languages can
        be enabled using the Add operation. Once enabled, the Esri provided
        content for these languages will be searchable and accessible to
        users in your organization. Enabled languages can be disabled using
        the Remove operation, which will remove their Esri provided content
        from the organization. Note that at least one language must always
        be enabled for your organization.

        :return: Dict[str, bool]
        """
        existing_languages = self.languages
        adds = {}
        removes = {}
        for k in languages.keys():
            if (k in existing_languages and k in languages) and languages[
                k
            ] != existing_languages[k]:
                if languages[k]:
                    adds[k] = True
                else:
                    removes[k] = False
        if adds:
            url = f"{self._url}/add"
            params = {"f": "json", "languages": ",".join(list(adds.keys()))}
            res = self._con.post(url, params)
            del res
        if removes:
            url = f"{self._url}/remove"
            params = {"f": "json", "languages": ",".join(list(adds.keys()))}
            res = self._con.post(url, params)
            del res
        return self.languages

"""
Configures Social Providers for a Portal or ArcGIS Online
"""
from arcgis.gis import GIS
from arcgis._impl.common._mixins import PropertyMap


########################################################################
class SocialProviders(object):
    """
    Enables/Disables the Social Providers Settings for a GIS


    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    value               Required GIS.  This is an administrator connection to a GIS site.
    ===============     ====================================================================

    :return:
        :class:`~arcgis.gis.admin.SocialProviders` object

    """

    _gis = None
    _portal = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, gis):
        """Constructor"""
        self._gis = gis
        self._portal = self._gis._portal
        self._url = "%s%s" % (self._portal.resturl, "portals/self/socialProviders")
        self.properties

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the social providers configurations"""
        try:
            if self._properties is None:
                res = self._portal.con.get(path=self._url, params={"f": "json"})
                self._properties = PropertyMap(res)
            return self._properties
        except:
            self._properties = PropertyMap({})
            return self._properties

    # ----------------------------------------------------------------------
    @property
    def configuration(self):
        """
        Gets/Sets for the Social Providers on the GIS

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Optional dict or None.  If the value is None, the social provider
                            configuration is deleted.  If the value is a dictionary, a social
                            provider is setup on the site or updated.
        ===============     ====================================================================


        *Key:Value Dictionary Options for value Argument*

        =====================  =====================================================================
        **Key**                **Value**
        ---------------------  ---------------------------------------------------------------------
        signUpMode             optional string. Invitation or Automatic.
        ---------------------  ---------------------------------------------------------------------
        providers              required string. This is a list of strings seperated by a comma. The
                               allowed values are: facebook and google
        ---------------------  ---------------------------------------------------------------------
        role                   optional string. This is the default role setup when users login to
                               a GIS.
        ---------------------  ---------------------------------------------------------------------
        level                  optional integer.  This is the default level set when a social
                               provider user logins.
        ---------------------  ---------------------------------------------------------------------
        userCreditAssignment   optional integer. The default is -1, which means infinite credit
                               usage. The
        ---------------------  ---------------------------------------------------------------------
        groups                 optional string. A comma seperated list of group ids to assign new
                               users to when they login to using a social provider.
        ---------------------  ---------------------------------------------------------------------
        user_type              optional string. A default user license type.
        =====================  =====================================================================

        """
        if "config" in self.properties:
            return self.properties["config"]
        return self.properties

    # ----------------------------------------------------------------------
    @configuration.setter
    def configuration(self, value: dict):
        """
        See main ``configuration`` property docstring
        """
        if value is None:
            url = "%s%s" % (self._url, "/remove")
            params = {"f": "json", "clearEmptyFields": True}
            res = self._gis._con.post(path=url, files={}, postdata=params)
            if "success" in res and res["success"] == False:
                raise Exception("Could not update the Social Provider configuration")
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    value[k] = ",".join(v)
                elif v is None:
                    value[k] = ""
            url = "%s%s" % (self._url, "/configure")
            params = {"f": "json", "clearEmptyFields": True}
            params.update(value)
            res = self._gis._con.post(path=url, files={}, postdata=params)
            if "success" in res and res["success"] == False:
                raise Exception("Could not update the Social Provider configuration")
        self._properties = None

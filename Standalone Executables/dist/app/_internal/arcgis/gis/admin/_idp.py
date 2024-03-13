from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import GIS
from arcgis import env


########################################################################
class IdentityProviderManager(object):
    """
    Manages and Updates the SAML identity provider configuration for a given GIS.
    """

    _gis = None
    _portal = None
    _url = None
    _properties = None
    _allowed_keys = None

    # ----------------------------------------------------------------------
    def __init__(self, gis=None):
        """Constructor"""
        if gis is None:
            gis = env.active_gis
        isinstance(gis, GIS)
        self._gis = gis
        self._portal = self._gis._portal
        self._url = self._portal.resturl + "portals/self/idp"
        self._allowed_keys = [
            "groups",
            "supportSignedRequest",
            "updateProfileAtSignin",
            "entityId",
            "roleId",
            "bindingPostUrl",
            "certificate",
            "name",
            "logoutUrl",
            "id",
            "useSHA256",
            "encryptionCertificate",
            "bindingUrl",
            "level",
            "userCreditAssignment",
            "signUpMode",
            "supportsLogoutRequest",
            "encryptionSupported",
            "idpMetadataFile",
        ]

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return "< %s @ %s >" % (type(self).__name__, self._url)

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the IDP configuration"""
        if self._properties is None:
            params = {"f": "json"}
            res = self._gis._con.get(
                path=self._url, params=params, return_raw_response=True
            )
            try:
                self._properties = PropertyMap(res.json())
            except:
                self._properties = PropertyMap({})
        return self._properties

    # ----------------------------------------------------------------------
    @property
    def configuration(self):
        """
        Gets, updates, or Adds a SAML provider

        ======================  =======================================================================================
        **Arguement**           **Value**
        ----------------------  ---------------------------------------------------------------------------------------
        value                   required dictionary.  This property sets, updates or deletes an IDP
                                configuration for a given GIS.

                                To configure an IDP, provide the key/value
                                Example:

                                idp.configuration = {'name' : 'Enterprise IDP', 'idpMetadataFile' : 'metadata.xml'}

                                Once a site has been configured to use IDP, the configuration can be
                                updated by passing in the key/value pair dictionary.
                                Example:

                                idp.configuration = {'name' : 'Acme IDP Login'}

                                To erase an IDP configuration, set the value to None
                                Example:

                                idp.configuration = None

                                Everytime the IDP configuration is updated, the changes can be seen
                                by calling the 'configuration' property and the new results will be
                                returned as a dictionary.
        ======================  =======================================================================================

        *Key:Value Dictionary for Argument value*

        ======================  =====================================================================
        **Key**                 **Value**
        ----------------------  ---------------------------------------------------------------------
        bindingPostUrl          Optional string. If the idpMetadataFile isn't specified when an
                                administrator, this parameter is required.  It is federated identity
                                provider post url.
        ----------------------  ---------------------------------------------------------------------
        bindingUrl              Optional string. If the idpMetadataFile isn't specified when an
                                administrator, this parameter is required.  It is federated identity
                                provider url that we have to redirect the user to login to.
        ----------------------  ---------------------------------------------------------------------
        certificate             Optional string. the X509Certificate that needs to be used to
                                validate the SamlResponse from the identity provider.
        ----------------------  ---------------------------------------------------------------------
        encryptionCertificate   Optional string. the X509Certificate that needs to be used to
                                validate the SamlResponse from the identity provider.
        ----------------------  ---------------------------------------------------------------------
        encryptionSupported     Optional bool.  Tells is the SAML provider supports encryption.
        ----------------------  ---------------------------------------------------------------------
        entityId                Optional string.  Name of the entity ID.
        ----------------------  ---------------------------------------------------------------------
        groups                  Optional list. List of group ids that users will be put in on
                                when they signup to join the GIS.
        ----------------------  ---------------------------------------------------------------------
        id                      Optional string. unique identifier of the IDP provider.
        ----------------------  ---------------------------------------------------------------------
        idpMetadataFile         Optional string.  In the case the URL is not accessible, then the
                                same IDP Metadata file can be uploaded.
        ----------------------  ---------------------------------------------------------------------
        level                   Optional integer. Either value 1 or 2. The default level a user will
                                be created as.  The default is 2.
        ----------------------  ---------------------------------------------------------------------
        logoutUrl               Optional string.  The logout SAML url.
        ----------------------  ---------------------------------------------------------------------
        name                    Optional string.  It is the name of the organization's federated
                                identity provider. This is also the name we show up in the Signin
                                page.
        ----------------------  ---------------------------------------------------------------------
        roleId                  Optional string. Default role new users will be.
        ----------------------  ---------------------------------------------------------------------
        signUpMode              Optional string. This is how new users are added to the GIS. There
                                are two modes: Invitation, Automatic
                                Invitation user needs to get an invitation and then signin through
                                federated identity provider.
                                With Automatic all users that signin through the federated identity
                                provider will be added as a user. The privilege/role is set to 'user'
                                Default is Invitation.
        ----------------------  ---------------------------------------------------------------------
        supportSignedRequest    Optional boolean. Determines if signed requests are supported from
                                the provider.
        ----------------------  ---------------------------------------------------------------------
        supportsLogoutRequest   Optional boolean. Determines if logout requests are accepted.
        ----------------------  ---------------------------------------------------------------------
        updateProfileAtSignin   Optional boolean. If True, users have to update the profile.
        ----------------------  ---------------------------------------------------------------------
        useSHA256               Optional boolean. If set to true, SHA256 encryption will be used.
        ----------------------  ---------------------------------------------------------------------
        userCreditAssignment    Optional integer.  Assigns a set number of credits to new users. The
                                default is -1 (infinite).
        ======================  =====================================================================
        """
        return self.properties

    # ----------------------------------------------------------------------
    @configuration.setter
    def configuration(self, value):
        """
        See main ``configuration`` property docstring
        """
        if len(dict(self.properties)) == 0 and value is not None:
            self._add(**value)
        elif value is None:
            self._unregister()
        elif len(dict(self.properties)) > 0:
            self._update(**value)

    # ----------------------------------------------------------------------
    def _add(self, **kwargs):
        """
        registers the inital idp configuration
        """
        if "name" not in kwargs:
            import uuid

            kwargs["name"] = uuid.uuid4().hex[:7]
        url = self._url + "/register"
        params = {"f": "json"}
        file = kwargs.pop("idpMetadataFile", None)
        params.update(dict(kwargs))
        for k, v in params.items():
            if isinstance(v, list):
                params[k] = ",".join(v)
            else:
                params[k] = v
        if file is not None:
            files = {"idpMetadataFile": file}
        else:
            files = {}
        res = self._gis._con.post(path=url, postdata=params, files=files)
        self._properties = None
        return res

    # ----------------------------------------------------------------------
    def _unregister(self):
        """unregisters the current IDP settings"""
        if len(dict(self.properties)) == 0:
            return True
        url = self._url + "/%s/unregister" % self.properties["id"]
        params = {"f": "json"}
        self._gis._con.post(path=url, postdata=params)
        self._properties = None

    # ----------------------------------------------------------------------
    def _update(self, **kwargs):
        """
        updates the idp configuration
        """
        import json

        if "id" in self._properties:
            url = self._url + "/%s/update" % self.properties["id"]
            params = {"f": "json"}
            file = kwargs.pop("idpMetadataFile", None)
            params.update(dict(kwargs))
            for k, v in params.items():
                if isinstance(v, list):
                    params[k] = ",".join(v)
                else:
                    params[k] = v
            if file is not None:
                files = {"idpMetadataFile": file}
            else:
                files = {}
            res = self._gis._con.post(
                path=url, postdata=params, files=files, try_json=False
            ).replace('",}', '"}')

            self._properties = None
            try:
                return json.loads(res)["success"]
            except:
                return True
        return False

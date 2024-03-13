"""
This resource lists all the portal machines in a site. Each portal machine
has a status that indicates whether the machine is ready to accept
requests.
"""
from ._base import BasePortalAdmin
from ..._impl.common._mixins import PropertyMap


########################################################################
class Machines(BasePortalAdmin):
    """
    This resource lists all the portal machines in a site. Each portal machine
    has a status that indicates whether the machine is ready to accept
    requests.
    """

    _pa = None
    _url = None
    _properties = None
    _json_dict = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(self, url, gis, portaladmin, **kwargs):
        """Constructor"""
        super(Machines, self).__init__(url=url, gis=gis, portaladmin=portaladmin)
        initialize = kwargs.pop("initialize", False)
        self._url = url
        self._pa = portaladmin
        if initialize:
            self._init()

    # ----------------------------------------------------------------------
    def list(self):
        """
        provides a list of all registered machines with the local GIS
        """
        machines = []
        for m in self.properties.machines:
            machines.append(
                Machine(
                    name=m["machineName"],
                    url=self._url,
                    info=dict(m),
                    gis=self._gis,
                    portaladmin=self._pa,
                )
            )
        return machines

    # ----------------------------------------------------------------------
    def get(self, name: str):
        """
        allows for retrieval of a single instance of Machine by it's
        registered name.
        """
        for m in self.properties.machines:
            if m["machineName"].lower() == name.lower():
                return Machine(
                    name=m["machineName"],
                    gis=self._gis,
                    url=self._url,
                    info=dict(m),
                    portaladmin=self._pa,
                )
        return


########################################################################
class Machine(BasePortalAdmin):
    """
    Represents a single machine instance registered with the GIS

    Parameters:
    :param name: machine name
    :param url: base url for the machine
    :param info: dictionary representing a single machine
    :param portaladmin: PortalAdminManager object
    :param initialize: (optional) if True, properties are loaded on
    creation
    """

    _pa = None
    _con = None
    _info = None
    _name = None
    _url = None
    _properties = None
    _json_dict = None
    _json = None

    # ----------------------------------------------------------------------
    def __init__(self, name, url, info, gis, portaladmin, **kwargs):
        """Constructor"""
        self._info = info
        super(Machine, self).__init__(
            url=url, gis=gis, name=name, info=info, portaladmin=portaladmin
        )
        self._info = info
        initialize = kwargs.pop("initialize", False)
        self._name = name
        self._url = url
        self._pa = portaladmin
        self._con = portaladmin._con
        self._gis = gis
        if initialize:
            self._init()

    # ----------------------------------------------------------------------
    def _init(self, connection=None):
        self._properties = PropertyMap(self._info)
        self._json_dict = self._info

    # ----------------------------------------------------------------------
    def status(self):
        """
        This operation checks whether a portal machine is ready to receive
        requests.
        """
        url = "%s/status/%s" % (self._url, self._name)
        params = {"f": "json"}
        res = self._con.get(path=url, params=params)
        if "status" in res:
            return res["status"] == "success"
        return False

    # ----------------------------------------------------------------------
    def unregister(self):
        """
        This operation unregisters a portal machine from a portal site. The
        operation can only performed when there are two machines
        participating in a portal site.
        """
        params = {"f": "json", "machineName": self._name}
        url = "%s/machines/unregister" % self._url
        return self._con.post(path=url, postdata=params)

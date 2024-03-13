from __future__ import annotations
import concurrent.futures
from arcgis.gis import GIS, Item
from arcgis.gis.nb import NotebookManager
from arcgis.gis.agonb import AGOLNotebookManager


def _list_instances(gis: GIS = None) -> list[dict[str:str]]:
    """
    Returns a list of avaialable Machine Instances.

    :returns: list[dict[str:str]]
    """

    if gis is None:
        from arcgis import env

        gis = env.active_gis
        assert gis
    mgrs = gis.notebook_server
    if len(mgrs) > 0:
        if gis._portal.is_arcgisonline:
            mgr = mgrs[0]
            ip = mgr.instance_preferences
            return ip.instances.get("instanceTypePreferences", [])
    return []


def list_runtimes(gis: GIS = None) -> list:
    """
    Returns a list of avaialable runtimes.

    :returns: list[dict[str:str]]
    """

    if gis is None:
        from arcgis import env

        gis = env.active_gis
        assert gis
    mgrs = gis.notebook_server
    if len(mgrs) > 0:
        if gis._portal.is_arcgisonline:
            mgr = mgrs[0]
            rt = mgr.runtimes
            return rt.list()
        else:
            mgr = mgrs[0]
            nb = mgr.notebooks
            rt = nb.runtimes
            return [r.properties for r in rt]
    return []


def execute_notebook(
    item: Item,
    *,
    timeout: int = 50,
    update_portal_item: bool = True,
    parameters: list = None,
    save_parameters: bool = False,
    server_index: int = 0,
    gis: GIS = None,
    future: bool = False,
) -> dict | concurrent.futures.Future:
    """

    The Execute Notebook operation allows administrators and users with
    the `Create and Edit Notebooks` privilege to remotely run a notebook
    that they own.  The notebook specified in the operation will be run
    with all cells in order.

    Using this operation, you can schedule the execution of a notebook,
    either once or with a regular occurrence. This allows you to
    automate repeating tasks such as data collection and cleaning,
    content updates, and portal administration. On Linux machines, use
    a cron job to schedule the executeNotebook operation; on Windows
    machines, you can use the Task Scheduler app.

    .. note::
        To run this operation in ArcGIS Enterprise, you must log in with
        an Enterprise account. You cannot execute notebooks using the
        ArcGIS Notebook Server primary site administrator account.

    .. note::
        ArcGIS Online has additional parameters, as noted in the parameter
        table below.

    You can specify parameters to be used in the notebook at execution
    time. If you've specified one or more parameters, they'll be
    inserted into the notebook as a new cell. This cell will be placed
    at the beginning of the notebook, unless you have added the tag
    parameters to a cell.

    ====================    ====================================================================
    **Parameter**            **Description**
    --------------------    --------------------------------------------------------------------
    item                    Required :class:`~arcgis.gis.Item`. Opens an existing portal item.
    --------------------    --------------------------------------------------------------------
    update_portal_item      Optional Boolean. Specifies whether you want to update the
                            notebook's portal item after execution. The default is true. You may
                            want to specify true when the notebook you're executing contains
                            information that needs to be updated, such as a workflow that
                            collects the most recent version of a dataset. It may not be
                            important to update the portal item if the notebook won't store any
                            new information after executing, such as an administrative notebook
                            that emails reminders to inactive users.
    --------------------    --------------------------------------------------------------------
    parameters              Optional List. An optional array of parameters to add to the
                            notebook for this execution. The parameters will be inserted as a
                            new cell directly after the cell you have tagged ``parameters``.
                            Separate parameters with a comma. Use the format "x":1 when
                            defining parameters with numbers, and "y":"text" when defining
                            parameters with text strings.
    --------------------    --------------------------------------------------------------------
    save_parameters         Optional Boolean.  Specifies whether the notebook parameters cell
                            should be saved in the notebook for future use. The default is
                            false.
    --------------------    --------------------------------------------------------------------
    timeout                 Optional Int. The number of minutes to run the instance before timeout. This is only available on ArcGIS Online.
    --------------------    --------------------------------------------------------------------
    future                  Optional boolean. If True, a Job object will be returned and the process
                            will not wait for the task to complete. The default is False, which means wait for results.
    ====================    ====================================================================

    :returns: Dict else If ``future = True``, then the result is
              a `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html>`_ object.
              Call ``result()`` to get the response

    .. code-block:: python

        #Usage example

        >>> from arcgis.gis import GIS
        >>> from arcgis.notebook import execute_notebook

        >>> gis = GIS(profile="your_org_profile")

        >>> nb_item = gis.content.get("ac7b7792913b4b3c9b22da4e2c42f986")
        >>> execute_notebook(nb_item)

    """
    if gis is None:
        from arcgis import env

        gis = env.active_gis or item._gis
        assert gis

    mgrs = gis.notebook_server
    if len(mgrs) > 0:
        if gis._portal.is_arcgisonline:
            instance_type = None
            mgr = gis.notebook_server[0]
            assert isinstance(mgr, AGOLNotebookManager)
            return mgr.notebooksmanager.execute_notebook(
                item=item,
                update_portal_item=update_portal_item,
                parameters=parameters,
                save_parameters=save_parameters,
                instance_type=instance_type,
                timeout=timeout,
                future=future,
            )
        else:
            mgr = gis.notebook_server[server_index].notebooks
            assert isinstance(mgr, NotebookManager)
            return mgr.execute_notebook(
                item=item,
                update_portal_item=update_portal_item,
                parameters=parameters,
                save_parameters=save_parameters,
                future=future,
            )

    return

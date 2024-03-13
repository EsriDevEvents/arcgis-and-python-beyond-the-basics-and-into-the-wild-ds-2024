"""
The arcgis.workflowmanager module contains classes and functions for working with a Workflow Manager installation.
Workflow diagrams, job templates and jobs can be created, modified, or deleted. Information such as location,
assignment, history and attachments for individual jobs can be accessed. Additionally, information about the various
roles, users, groups, and searches can be view, modified or created.
"""

from ._workflow_manager import WorkflowManager
from ._workflow_manager import WorkflowManagerAdmin
from ._workflow_manager import JobManager, JobLocation, JobTemplate
from ._workflow_manager import Job
from ._workflow_manager import JobDiagram
from ._workflow_manager import Group
from ._workflow_manager import SavedSearchesManager
from ._workflow_manager import WMRole
from ._workflow_manager import LookUpTable

""" The store module defines functions for interacting with the Workforce backend.
"""
from . import assignments
from . import assignment_types
from . import assignment_types_v2
from . import attachments
from . import dispatchers
from . import workers
from . import tracks
from . import projects

from .assignment_types import (
    get_assignment_type,
    get_assignment_types,
    add_assignment_types,
    update_assignment_types,
    delete_assignment_types,
    add_assignment_type,
    update_assignment_type,
)

from .assignment_types_v2 import (
    get_assignment_type_v2,
    get_assignment_types_v2,
    add_assignment_types_v2,
    update_assignment_types_v2,
    delete_assignment_types_v2,
    add_assignment_type_v2,
    update_assignment_type_v2,
)

from .assignments import (
    get_assignment,
    get_assignments,
    query_assignments,
    add_assignments,
    update_assignments,
    delete_assignments,
    add_assignment,
    update_assignment,
)

from .attachments import get_attachments, add_attachment, delete_attachments

from .dispatchers import (
    get_dispatcher,
    get_dispatchers,
    query_dispatchers,
    add_dispatchers,
    update_dispatchers,
    delete_dispatchers,
    add_dispatcher,
    update_dispatcher,
)

from .integrations import (
    get_integration,
    query_integrations,
    add_integration,
    add_integrations,
    update_integration,
    update_integrations,
    delete_integrations,
)

from .projects import get_project, create_project

from .tracks import (
    get_track,
    get_tracks,
    query_tracks,
    delete_tracks,
    add_tracks,
    add_track,
    update_tracks,
    update_track,
)

from .workers import (
    get_worker,
    get_workers,
    query_workers,
    add_workers,
    update_workers,
    delete_workers,
    add_worker,
    update_worker,
)

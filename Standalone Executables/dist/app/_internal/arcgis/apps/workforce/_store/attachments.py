""" Defines store functions for working with Attachments.
"""

from ... import workforce


def get_attachments(assignment):
    """Gets all Attachments for the assignment.
    :param assignment: An assignment that exists on the server.
    :returns: list of Attachments.
    """
    feature_layer = assignment.project.assignments_layer
    attachment_infos = feature_layer.attachments.get_list(assignment.object_id)
    return [workforce.Attachment(assignment, attinfo) for attinfo in attachment_infos]


def add_attachment(assignment, file_path):
    """Adds an Attachment to the assignment.
    :param assignment: the assignment to add the attachment to
    :param file_path: local path of file to upload.
    :raises ServerError: Indicates that the server rejected the attachment upload.
    """
    feature_layer = assignment.project.assignments_layer
    response = feature_layer.attachments.add(assignment.object_id, file_path)
    if not response["addAttachmentResult"]["success"]:
        raise workforce.ServerError([response["addAttachmentResult"]["error"]])


def delete_attachments(assignment, attachments):
    """Removes the attachments from the assignment.
    :param assignment: An assignment that exists on the server.
    :param attachments: The attachments to remove.
    :raises ServerError: Indicates that the server rejected the attachment removal.
    """
    if attachments:
        feature_layer = assignment.project.assignments_layer
        attachment_ids = ",".join([str(attachment.id) for attachment in attachments])
        response = feature_layer.attachments.delete(
            assignment.object_id, attachment_ids
        )
        errors = [
            result.error
            for result in response["deleteAttachmentResults"]
            if not result["success"]
        ]
        if errors:
            raise workforce.ServerError(errors)

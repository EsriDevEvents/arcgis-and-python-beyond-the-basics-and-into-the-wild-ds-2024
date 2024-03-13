""" Defines the Attachment object.
"""

import os

from .model import Model


class Attachment(Model):
    """
    Represents a file attachment for an Assignment

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    assignment             Required :class:`~arcgis.apps.workforce.Assignment`. The assignment object that this attachment belongs
                           to
    ------------------     --------------------------------------------------------------------
    attachment_info        Required :class:`dict`. The attachment info dictionary representing the
                           attachment.
    ==================     ====================================================================

    """

    def __init__(self, assignment, attachment_info):
        self.assignment = assignment
        self._attachment_info = attachment_info

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<Attachment {} on {}>".format(self.id, repr(self.assignment))

    @property
    def id(self):
        """Gets the attachment id"""
        return self._attachment_info["id"]

    @property
    def global_id(self):
        """Gets the attachment global id"""
        if "globalId" in self._attachment_info:
            return self._attachment_info["globalId"]
        elif "globalid" in self._attachment_info:
            return self._attachment_info["globalid"]

    @property
    def name(self):
        """Gets the attachment name"""
        return self._attachment_info["name"]

    @property
    def size(self):
        """Gets the attachment size"""
        return self._attachment_info["size"]

    @property
    def content_type(self):
        """Gets the attachment content type"""
        return self._attachment_info["contentType"]

    @property
    def project(self):
        """Gets the project that the attachment belongs to"""
        return self.assignment.project

    @property
    def attachment_info(self):
        """Gets the attachment info of the attachment"""
        return self._attachment_info

    def download(self, out_folder=None):
        """Downloads the attachment to the specified path.  If the path is omitted, the Attachment
        will be saved to the current working directory, using the name property as the filename.
        :param out_folder: The folder in which the attachment should be saved.  Defaults to the
        current working directory.

        :return: The absolute path to the downloaded file.
        """
        if not out_folder:
            out_folder = os.getcwd()
        paths = self.project.assignments_layer.attachments.download(
            self.assignment.object_id, self.id, out_folder
        )
        if len(paths) == 1:
            return paths[0]
        else:
            return paths

from __future__ import annotations
from enum import Enum
from typing import Optional, Union
import uuid
from arcgis.auth.tools import LazyLoader

arcgis = LazyLoader("arcgis")
urllib3 = LazyLoader("urllib3")
requests = LazyLoader("requests")
mimetypes = LazyLoader("mimetypes")
pil_image = LazyLoader("PIL.Image")
os = LazyLoader("os")
_io = LazyLoader("io")
_parse = LazyLoader("urllib.parse")


class TextStyles(Enum):
    """
    Represents the Supported Text Styles Type Enumerations.
    Example: Text(text="foo", style=TextStyles.HEADING)
    """

    PARAGRAPH = "paragraph"
    LARGEPARAGRAPH = "large-paragraph"
    BULLETLIST = "bullet-list"
    NUMBERLIST = "numbered-list"
    HEADING = "h2"
    SUBHEADING = "h3"
    QUOTE = "quote"


class Scales(Enum):
    """
    Scale is a unitless way of describing how any distance on the map translates
    to a real-world distance. For example, a map at a 1:24,000 scale communicates that 1 unit
    on the screen represents 24,000 of the same unit in the real world.
    So one inch on the screen represents 24,000 inches in the real world.
    """

    WORLD = {"scale": 147914382, "zoom": 2}
    CONTINENT = {"scale": 50000000, "zoom": 3}
    COUNTRIESLARGE = {"scale": 25000000, "zoom": 4}
    COUNTRIESSMALL = {"scale": 12000000, "zoom": 5}
    STATES = {"scale": 6000000, "zoom": 6}
    PROVINCES = {"scale": 6000000, "zoom": 6}
    STATE = {"scale": 3000000, "zoom": 7}
    PROVINCE = {"scale": 3000000, "zoom": 7}
    COUNTIES = {"scale": 1500000, "zoom": 8}
    COUNTY = {"scale": 750000, "zoom": 9}
    METROPOLITAN = {"scale": 320000, "zoom": 10}
    CITIES = {"scale": 160000, "zoom": 11}
    CITY = {"scale": 80000, "zoom": 12}
    TOWN = {"scale": 40000, "zoom": 13}
    NEIGHBORHOOD = {"scale": 2000, "zoom": 14}
    STREETS = {"scale": 10000, "zoom": 15}
    STREET = {"scale": 5000, "zoom": 16}
    BUILDINGS = {"scale": 2500, "zoom": 17}
    BUILDING = {"scale": 1250, "zoom": 18}
    SMALLBUILDING = {"scale": 800, "zoom": 19}
    ROOMS = {"scale": 400, "zoom": 20}
    ROOM = {"scale": 100, "zoom": 22}


###############################################################################################################
class Separator:
    """
    Class representing a `separator`. You can use this class to edit and remove separators from a storymap.
    """

    def __init__(self, **kwargs) -> None:
        # Can be created from scratch or already exist in story
        # Separator is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "separator"
        self.node = kwargs.pop("node_id", "n-" + uuid.uuid4().hex[0:6])

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Image"

    # ----------------------------------------------------------------------
    def _add_separator(self, story=None):
        # Assign the story
        self._story = story

        # Create separator nodes.
        self._story._properties["nodes"][self.node] = {
            "type": "separator",
        }

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)


###############################################################################################################
class Image:
    """
    Class representing an `image` from a url or file.

    .. warning::
        Image must be smaller than 10 MB to avoid having issues when saving or publishing.

    .. note::
        Once you create an Image instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    path                    Required String. The file path or url to the image that will be added.
    ==================      ====================================================================
    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        # Image is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "image"
        # Keep track if URL since different representation style in story dictionary
        self._url = False
        self.node = kwargs.pop("node_id", None)
        # If node exists in story, then create from resources and node dictionary provided.
        # If node doesn't already exist, create a new instance.
        self._existing = self._check_node()
        if self._existing is True:
            # Get the resource node id
            self.resource_node = self._story._properties["nodes"][self.node]["data"][
                "image"
            ]
            if (
                self._story._properties["resources"][self.resource_node]["data"][
                    "provider"
                ]
                == "uri"
            ):
                # Indicate that the image comes from a url
                self._url = True
            if self._url is True:
                # Path differs whether from file path or url originally
                self._path = self._story._properties["resources"][self.resource_node][
                    "data"
                ]["src"]
            else:
                self._path = self._story._properties["resources"][self.resource_node][
                    "data"
                ]["resourceId"]
        else:
            # Create a new instance of Image
            self._path = path
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self.resource_node = "r-" + uuid.uuid4().hex[0:6]

            # Determine if url or file path
            if _parse.urlparse(self._path).scheme == "https":
                self._url = True

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Image"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties for the Image.

        :return:
            A dictionary depicting the node dictionary and resource
            dictionary for the image.
            If nothing is returned, make sure your content has been added
            to the story.
        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
                "resource_dict": self._story._properties["resources"][
                    self.resource_node
                ],
            }

    # ----------------------------------------------------------------------
    @property
    def image(self):
        """
        Get/Set the image property.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        image               String. The new image path or url for the Image.
        ==================  ========================================

        :return:
            The image that is being used.
        """
        if self._existing is True:
            if self._url is False:
                return self._story._properties["resources"][self.resource_node]["data"][
                    "resourceId"
                ]
            else:
                return self._story._properties["resources"][self.resource_node]["data"][
                    "src"
                ]

    # ----------------------------------------------------------------------
    @image.setter
    def image(self, path):
        if self._existing is True:
            self._update_image(path)
            return self.image

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the image.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Image.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption
            return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the image.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Image.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
            return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set display for image.

        Values: `small` | `wide` | `full` | `float`
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["config"]["size"]

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["config"]["size"] = display
            return self.display

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_image(self, caption=None, alt_text=None, display=None, story=None):
        # Assign the story
        self._story = story
        self._existing = True
        # Make an add resource call if not url
        if self._url is False:
            self._story._add_resource(self._path)

        # Create image nodes. This is similar for file path and url
        self._story._properties["nodes"][self.node] = {
            "type": "image",
            "data": {
                "image": self.resource_node,
                "caption": "" if caption is None else caption,
                "alt": "" if alt_text is None else alt_text,
            },
            "config": {"size": "" if display is None else display},
        }

        # Create resource node. Different if file path or url
        if self._url is False:
            # Get image properties and create the resourceId that corresponds to the resource added
            im = pil_image.open(self._path)
            w, h = im.size
            self._story._properties["resources"][self.resource_node] = {
                "type": "image",
                "data": {
                    "resourceId": os.path.basename(os.path.normpath(self._path)),
                    "provider": "item-resource",
                    "height": h,
                    "width": w,
                },
            }
        else:
            # Get image properties and assign the image src
            data = requests.get(self._path).content
            im = pil_image.open(_io.BytesIO(data))
            w, h = im.size
            self._story._properties["resources"][self.resource_node] = {
                "type": "image",
                "data": {
                    "src": self._path,
                    "provider": "uri",
                    "height": h,
                    "width": w,
                },
            }

    # ----------------------------------------------------------------------
    def _update_image(self, new_image):
        # Check if new_image is url or path
        if _parse.urlparse(new_image).scheme == "https":
            # New image is a Url
            self._url = True
            # Update the height and width for the image
            data = requests.get(new_image).content
            im = pil_image.open(_io.BytesIO(data))
            w, h = im.size
            self._story._properties["resources"][self.resource_node]["data"][
                "height"
            ] = h
            self._story._properties["resources"][self.resource_node]["data"][
                "width"
            ] = w

            # Update resource dictionary
            # Do not need to make a resource
            self._story._properties["resources"][self.resource_node]["data"][
                "src"
            ] = new_image
            # Delete if the image was previously a file path
            if (
                "resouceId"
                in self._story._properties["resources"][self.resource_node]["data"]
            ):
                del self._story._properties["resources"][self.resource_node]["data"][
                    "resourceId"
                ]
            # Update provider
            self._story._properties["resources"][self.resource_node]["data"][
                "provider"
            ] = "uri"
        else:
            # Update the height and width for the image
            self._url = False
            im = pil_image.open(new_image)
            w, h = im.size
            self._story._properties["resources"][self.resource_node]["data"][
                "height"
            ] = h
            self._story._properties["resources"][self.resource_node]["data"][
                "width"
            ] = w

            # Update resource dictionary
            resource_id = (
                self._story._properties["resources"][self.resource_node]["data"][
                    "resourceId"
                ]
                if "resourceId"
                in self._story._properties["resources"][self.resource_node]["data"]
                else None
            )
            # Update where file path is held
            self._story._properties["resources"][self.resource_node]["data"][
                "resourceId"
            ] = os.path.basename(os.path.normpath(new_image))
            # Delete path if item was previously a url
            if (
                "src"
                in self._story._properties["resources"][self.resource_node]["data"]
            ):
                del self._story._properties["resources"][self.resource_node]["data"][
                    "src"
                ]
            # Update provider
            self._story._properties["resources"][self.resource_node]["data"][
                "provider"
            ] = "item-resource"
            # Update the resource by removing old and adding new
            if resource_id:
                self._story._remove_resource(resource_id)
            self._story._add_resource(new_image)
        # Set new path
        self._path = new_image

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Node is not in the story if no story or node id is present
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Video:
    """
    Class representing a `video` from a url or file

    .. note::
        Once you create a Video instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    path                    Required String. The file path or embed url to the video that will
                            be added.

                            .. note::
                                URL must be an embed url.
                                Example: "https://www.youtube.com/embed/G6b7Kgvd0iA"

    ==================      ====================================================================
    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        # Video is not an immersive node
        # Get properties if provided
        self._story = kwargs.pop("story", None)
        self._type = "video"
        # Hold whether video is url, this will impact the dictionary structure
        self._url = False
        self.node = kwargs.pop("node_id", None)
        # Check if node already in story, else create new instance
        self._existing = self._check_node()
        if self._existing is True:
            # If node is type video then video came from file path
            if self._story._properties["nodes"][self.node]["type"] == "video":
                self.resource_node = self._story._properties["nodes"][self.node][
                    "data"
                ]["video"]
                self._path = self._story._properties["resources"][self.resource_node][
                    "data"
                ]["resourceId"]
            else:
                # Node is of embedType: video and video came from url
                self.resource_node = None
                self._path = self._story._properties["nodes"][self.node]["data"]["url"]
                self._url = True
        else:
            # Create new instance of Video
            self._path = path
            self.node = "n-" + uuid.uuid4().hex[0:6]
            if _parse.urlparse(path).scheme == "https":
                self._url = True
                self.resource_node = None
            else:
                self.resource_node = "r-" + uuid.uuid4().hex[0:6]

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Video"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Video"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties for the Video.

        :return:
            A dictionary depicting the node dictionary and resource
            dictionary for the video.
            If nothing is returned, make sure the content is part of the story.

        .. note::
            To change various properties of the Video use the other property setters.
        """
        if self._existing is True:
            vid_dict = {
                "node_dict": self._story._properties["nodes"][self.node],
            }
            if self.resource_node:
                vid_dict["resource_dict"] = (
                    self._story._properties["resources"][self.resource_node],
                )
            return vid_dict

    # ----------------------------------------------------------------------
    @property
    def video(self):
        """
        Get/Set the video property.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        video               String. The new video path for the Video.
        ==================  ========================================

        :return:
            The video that is being used.
        """
        if self._existing is True:
            if self.resource_node:
                # If resouce node exists it means the video comes from a file path
                return self._story._properties["resources"][self.resource_node]["data"][
                    "resourceId"
                ]
            else:
                # No resource node means the video is of type embed and embedType: video
                return self._story._properties["nodes"][self.node]["data"]["url"]

    # ----------------------------------------------------------------------
    @video.setter
    def video(self, path):
        if self._existing is True:
            self._update_video(path)
            return self.video

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the video.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Video.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption
            return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the video.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Video.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
            return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set display for the video.

        Values: `small` | `wide` | `full` | `float`

        .. note::
            Cannot change display when video is created from a url
        """
        if self._existing is True:
            if self._url is True:
                return self._story._properties["nodes"][self.node]["data"]["display"]
            else:
                return self._story._properties["nodes"][self.node]["config"]["size"]

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._existing is True:
            if self._url is True:
                self._story._properties["nodes"][self.node]["data"]["display"] = display
        return self.display

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_video(
        self,
        caption=None,
        alt_text=None,
        display=None,
        story=None,
        node_id=None,
        resource_node=None,
    ):
        # Add the story to the node
        self._story = story
        self._existing = True
        if node_id:
            # If node already exists (updating node)
            self.node = node_id
        if resource_node:
            # If node already exists (updating node)
            self.resource_node = resource_node
        if self._url is False:
            # Make an add resource call since it is a file path
            self._story._add_resource(self._path)

            # Create video nodes for file path
            self._story._properties["nodes"][self.node] = {
                "type": "video",
                "data": {
                    "video": self.resource_node,
                    "caption": "" if caption is None else caption,
                    "alt": "" if alt_text is None else alt_text,
                },
                "config": {
                    "size": display,
                },
            }

            # Create resource node for file path
            self._story._properties["resources"][self.resource_node] = {
                "type": "video",
                "data": {
                    "resourceId": os.path.basename(os.path.normpath(self._path)),
                    "provider": "item-resource",
                },
            }
        else:
            # Path is a url so node will be type embed and embedType: video
            # No resource call or resource node is made
            self._story._properties["nodes"][self.node] = {
                "type": "embed",
                "data": {
                    "url": self._path,
                    "embedType": "video",
                    "caption": "" if caption is None else caption,
                    "alt": "" if alt_text is None else alt_text,
                    "display": "inline",
                    "aspectRatio": 1.778,
                    "addedAsEmbedCode": True,
                },
            }

    # ----------------------------------------------------------------------
    def _update_video(self, new_video):
        # Node structure depends if new_video is file path or url
        # Changes are made and add video call is done since easier than restructuring
        self._path = new_video
        if self.resource_node:
            # If resource node present, remove resource from item.
            resource_id = self._story._properties["resources"][self.resource_node][
                "data"
            ]["resourceId"]
            self._story._remove_resource(resource_id)
            # Remove the resource node since should not exist for url. Will be added back if file path
            del self._story._properties["resources"][self.resource_node]
        if _parse.urlparse(new_video).scheme == "https":
            # New video is a url
            self._url = True
            self.resource_node = None
            # Update the node by making add video call with correct parameters
            self._add_video(
                caption=self.caption,
                alt_text=self.alt_text,
                story=self._story,
                node_id=self.node,
            )
        else:
            # If the node was not a file path before, need to create resource id
            if self.resource_node is None:
                self.resource_node = "r-" + uuid.uuid4().hex[0:6]
            # display depends on self._url so get it before
            display = self.display
            self._url = False
            # Update the node by making add video call with correct parameters
            self._add_video(
                caption=self.caption,
                alt_text=self.alt_text,
                display=display,
                story=self._story,
                node_id=self.node,
                resource_node=self.resource_node,
            )

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Audio:
    """
    This class represents content that is of type `audio`. It can be created from
    a file path and added to the story.

    .. note::
        Once you create an Audio instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    path                    Required String. The file path to the audio that will be added.
    ==================      ====================================================================

    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        # Audio is not an immersive node
        if _parse.urlparse(path).scheme == "https":
            # Audio cannot be added by Url at this time.
            raise ValueError(
                "To add an audio from an embedded url, use the Embed content class."
            )
        # Assing audio node properties
        self._story = kwargs.pop("story", None)
        self._type = "audio"
        self.node = kwargs.pop("node_id", None)
        # If node does not exist yet, create new instance
        self._existing = self._check_node()
        if self._existing is True:
            # Get existing resouce node
            self.resource_node = self._story._properties["nodes"][self.node]["data"][
                "audio"
            ]
            # Get existing audio path
            self._path = self._story._properties["resources"][self.resource_node][
                "data"
            ]["resourceId"]
        else:
            # Create a new instance
            self._path = path
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self.resource_node = "r-" + uuid.uuid4().hex[0:6]

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Audio"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Audio"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties for the Audio.

        :return:
            A dictionary depicting the node dictionary and resource
            dictionary for the audio.

            If nothing is returned, make sure the content is part of the story.

        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
                "resource_dict": self._story._properties["resources"][
                    self.resource_node
                ],
            }

    # ----------------------------------------------------------------------
    @property
    def audio(self):
        """
        Get/Set the audio path.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        audio               String. The new audio path for the Audio.
        ==================  ========================================

        :return:
            The audio that is being used.
        """
        if self._existing is True:
            return self._story._properties["resources"][self.resource_node]["data"][
                "resourceId"
            ]

    # ----------------------------------------------------------------------
    @audio.setter
    def audio(self, path):
        if _parse.urlparse(path).scheme == "https":
            raise ValueError(
                "To add an audio from an embedded url, use the Embed content class. Update audio with file path only."
            )
        if self._existing is True:
            self._update_audio(path)
            return self.audio

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the audio.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Audio.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption
            return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the audio.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Audio.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
            return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set display for audio.

        Values: `small` | `wide` | float`
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["config"]["size"]

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._check_node() is True:
            self._story._properties["nodes"][self.node]["config"]["size"] = display
            return self.display

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_audio(
        self,
        caption=None,
        alt_text=None,
        display=None,
        story=None,
    ):
        self._story = story
        self._existing = True
        # Make an add resource call
        self._story._add_resource(self._path)
        # Create image nodes
        self._story._properties["nodes"][self.node] = {
            "type": "audio",
            "data": {
                "audio": self.resource_node,
                "caption": "" if caption is None else caption,
                "alt": "" if alt_text is None else alt_text,
            },
            "config": {"size": display},
        }

        # Create resource node
        self._story._properties["resources"][self.resource_node] = {
            "type": "audio",
            "data": {
                "resourceId": os.path.basename(os.path.normpath(self._path)),
                "provider": "item-resource",
            },
        }

    # ----------------------------------------------------------------------
    def _update_audio(self, new_audio):
        # Assign new path
        self._path = new_audio

        # Assign new resouce id, get old one to delete resource
        resource_id = self._story._properties["resources"][self.resource_node]["data"][
            "resourceId"
        ]
        self._story._properties["resources"][self.resource_node]["data"][
            "resourceId"
        ] = os.path.basename(os.path.normpath(self._path))

        # Add new resource and remove old one
        self._story._add_resource(self._path)
        self._story._remove_resource(resource_id)

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Embed:
    """
    Class representing a `webpage` or `embedded audio`.
    Embed will show as a card in the story.

    .. note::
        Once you create an Embed instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    path                    Required String. The url that will be added as a webpage, video, or
                            audio embed into the story.
    ==================      ====================================================================
    """

    def __init__(self, path: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        # Embed is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "embed"
        self.node = kwargs.pop("node_id", None)
        # If node doesn't already exist, create new instance
        self._existing = self._check_node()
        if self._existing is True:
            # Get the link path
            self._path = self._story._properties["nodes"][self.node]["data"]["url"]
        else:
            # Create new instance, notice no resource node is needed for embed
            self._path = path
            self.node = "n-" + uuid.uuid4().hex[0:6]

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties for the Embed.

        .. note::
            To change various properties of the Embed use the other property setters.

        :return:
            A dictionary depicting the node dictionary for the embed.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
            }

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Embed"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Embed"

    # ----------------------------------------------------------------------
    @property
    def link(self):
        """
        Get/Set the link property.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        link                String. The new url for the Embed.
        ==================  ========================================

        :return:
            The embed that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["url"]

    # ----------------------------------------------------------------------
    @link.setter
    def link(self, path):
        if self._existing is True:
            self._update_link(path)
            return self.link

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the webpage.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Embed.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption
            return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the embed.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Embed.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
            return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set display for embed.

        Values: `card` | `inline`
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["display"]

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._existing:
            self._story._properties["nodes"][self.node]["data"]["display"] = display
            return self.display

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_link(self, caption=None, alt_text=None, display="card", story=None):
        self._story = story
        self._existing = True
        sections = _parse.urlparse(self._path)
        # Create embed node, no resource node needed
        self._story._properties["nodes"][self.node] = {
            "type": "embed",
            "data": {
                "url": self._path,
                "embedType": "link",
                "title": sections.netloc,
                "description": "" if caption is None else caption,
                "providerUrl": sections.netloc,
                "alt": "" if alt_text is None else alt_text,
                "display": display,
            },
        }

    # ----------------------------------------------------------------------
    def _update_link(self, new_link):
        # parse new url
        sections = _parse.urlparse(new_link)
        # set new path
        self._path = new_link
        # update dictionary properties
        self._story._properties["nodes"][self.node]["data"]["url"] = self._path
        self._story._properties["nodes"][self.node]["data"]["title"] = sections.netloc
        self._story._properties["nodes"][self.node]["data"][
            "providerUrl"
        ] = sections.netloc

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Map:
    """
    Class representing a `webmap` or `webscene` for the story

    .. note::
        Once you create a Map instance you must add it to the story to be able to edit it further.

    =================       ====================================================================
    **Parameter**            **Description**
    -----------------       --------------------------------------------------------------------
    item                    An Item of type :class:`~arcgis.mapping.WebMap` or
                            :class:`~arcgis.mapping.WebScene` or a String representing the item
                            id to add to the story map.
    =================       ====================================================================
    """

    def __init__(self, item: Optional[arcgis.gis.Item] = None, **kwargs):
        # Can be created from scratch or already exist in story
        # Map is not an immersive node
        self._story = kwargs.pop("story", None)
        self.node = kwargs.pop("node_id", None)
        # Check if node exists else create new instance
        self._existing = self._check_node()
        if self._existing:
            # Gather all exisiting properties needed
            self.resource_node = self._story._properties["nodes"][self.node]["data"][
                "map"
            ]
            # The item id is in the resource node
            self._path = self.resource_node[2::]
            rdata = self._story._properties["resources"][self.resource_node]["data"]
            ndata = self._story._properties["nodes"][self.node]["data"]
            # map layers
            if "mapLayers" in ndata:
                self._map_layers = ndata["mapLayers"]
            elif "mapLayers" in rdata:
                self._map_layers = rdata["mapLayers"]
            else:
                self._map_layers = None

            # extent
            if "extent" in ndata:
                self._extent = ndata["extent"]
            elif "extent" in rdata:
                self._extent = rdata["extent"]
            else:
                self._extent = {}

            # center
            if "center" in ndata:
                self._center = ndata["center"]
            elif "center" in rdata:
                self._center = rdata["center"]
            else:
                self._center = None

            # viewpoint
            if "viewpoint" in ndata:
                self._viewpoint = ndata["viewpoint"]
            elif "viewpoint" in rdata:
                self._viewpoint = rdata["viewpoint"]
            else:
                self._viewpoint = {"rotation": 0, "scale": -1, "targetGeometry": {}}

            # zoom
            if "zoom" in ndata:
                self._zoom = ndata["zoom"]
            elif "zoom" in rdata:
                self._zoom = rdata["zoom"]
            else:
                self._zoom = 2

            # only in rdata
            self._type = rdata["itemType"]
            if self._type == "Web Scene":
                self._lighting_date = self._story._properties["resources"][
                    self.resource_node
                ]["data"]["lightingDate"]
        else:
            # Create new instance
            if isinstance(item, str):
                # If string id get the item
                item = arcgis.env.active_gis.content.get(item)
            # Create map object to extract properties
            if isinstance(item, arcgis.gis.Item):
                if item.type == "Web Map":
                    map_item = arcgis.mapping.WebMap(item)
                elif item.type == "Web Scene":
                    map_item = arcgis.mapping.WebScene(item)
                else:
                    raise ValueError("Item must be of Type Web Map or Web Scene")
            # Assign properties
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self.resource_node = "r-" + item.id
            self._path = item
            self._type = item.type
            if item.type == "Web Map":
                self._extent = map_item._mapview.extent
                if map_item._mapview.center is None:
                    x_center = (self._extent["xmin"] + self._extent["xmax"]) / 2
                    y_center = (self._extent["ymin"] + self._extent["ymax"]) / 2
                    self._center = {
                        "spatialReference": map_item.definition.spatialReference,
                        "x": x_center,
                        "y": y_center,
                    }
                else:
                    self._center = map_item._mapview.center
                self._zoom = map_item._mapview.zoom if map_item.zoom is not False else 2
                self._viewpoint = {
                    "rotation": map_item._mapview.rotation,
                    "scale": map_item._mapview.scale,
                    "targetGeometry": self._center,
                }

                layers = []
                # Create layer dictionary:
                for layer in map_item.layers:
                    layer_props = {}
                    layer_props["id"] = layer["id"]
                    layer_props["title"] = layer["title"]
                    if "visibility" in layer:
                        layer_props["visible"] = layer["visibility"]
                    elif "layer_visibility" in map_item:
                        layer_props["visible"] = map_item["layer_visibility"]
                    layers.append(layer_props)
                self._map_layers = layers
            # Add properties for Web Scene
            elif item.type == "Web Scene":
                layers = []
                # Create layer dictionary:
                for layer in map_item["operationalLayers"]:
                    layer_props = {}
                    layer_props["id"] = layer["id"]
                    layer_props["title"] = layer["title"]
                    if "visibility" in layer:
                        layer_props["visible"] = layer["visibility"]
                    else:
                        layer_props["visible"] = False
                    layers.append(layer_props)
                self._map_layers = layers
                self._extent = None
                self._center = map_item["initialState"]["viewpoint"]["camera"][
                    "position"
                ]
                self._zoom = 2
                self._viewpoint = map_item["initialState"]["viewpoint"]
                self._camera = map_item["initialState"]["viewpoint"]["camera"]
                self._lighting_date = map_item["initialState"]["environment"][
                    "lighting"
                ]["datetime"]

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self._type

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties for the Map.

        :return:
            A dictionary depicting the node dictionary and resource
            dictionary for the map. The resource dictionary depicts the
            original map settings. The node dictionary depicts the current map settings.
            If nothing it returned, make sure the content is part of the story.

        .. note::
            To change various properties of the Map use the other property setters.
        """
        if self._check_node() is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
                "resource_dict": self._story._properties["resources"][
                    self.resource_node
                ],
            }

    # ----------------------------------------------------------------------
    @property
    def map(self):
        """
        Get/Set the map property.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        map                 One of three choices:

                            * String being an item id for an Item of type
                            :class:`~arcgis.mapping.WebMap`
                            or :class:`~arcgis.mapping.WebScene`.

                            * An :class:`~arcgis.gis.Item` of type
                            :class:`~arcgis.mapping.WebMap`
                            or :class:`~arcgis.mapping.WebScene`.
        ==================  ========================================

        .. note::
            Only replace a Map with a new map of same type. Cannot replace a
            2D map with 3D.

        :return:
            The item id for the map that is being used.
        """
        if self._existing is True:
            map_id = self._story._properties["resources"][self.resource_node]["data"][
                "itemId"
            ]
            return self._story._gis.content.get(map_id)

    # ----------------------------------------------------------------------
    @map.setter
    def map(self, map):
        if self._existing is True:
            self._update_map(map)
            return self.map

    # ----------------------------------------------------------------------
    def set_viewpoint(self, extent: dict = None, scale: Scales = None):
        """
        Set the extent and/or scale for the map in the story.

        If you have an extent to use from a bookmark,
        find this extent by using the `bookmarks` property in
        the :class:`~arcgis.mapping.WebMap` Class.
        The `map` property on this class will return the Web Map
        Item being used. By passing this item into
        the :class:`~arcgis.mapping.WebMap` Class you can retrieve a list of all
        bookmarks and their extents with the `bookmarks` property.

        To see the current viewpoint call the `properties` property on the Map
        node.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        extent              Optional dictionary representing the extent of
                            the map. This will update the extent, center and viewpoint
                            accordingly.

                            Example:
                                | {'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
                                | 'xmin': -609354.6306080809,
                                | 'ymin': 2885721.2797636474,
                                | 'xmax': 6068184.160383142,
                                | 'ymax': 6642754.094035632}
        ------------------  ----------------------------------------
        scale               Optional Scales enum class value or dict with 'scale' and 'zoom' keys.

                            Scale is a unitless way of describing how any distance on the map translates
                            to a real-world distance. For example, a map at a 1:24,000 scale communicates that 1 unit
                            on the screen represents 24,000 of the same unit in the real world.
                            So one inch on the screen represents 24,000 inches in the real world.
        ==================  ========================================

        :return: The current viewpoint dictionary
        """
        rdata_dict = self._story._properties["resources"][self.resource_node]["data"]
        if "viewpoint" not in self._story._properties["nodes"][self.node]["data"]:
            try:
                self._story._properties["nodes"][self.node]["data"][
                    "viewpoint"
                ] = rdata_dict["viewpoint"]
            except:
                self._story._properties["nodes"][self.node]["data"]["viewpoint"] = {
                    "rotation": 0,
                    "scale": -1,
                    "targetGeometry": {},
                }

        change_made = False
        # set new extent if specified
        if extent:
            if isinstance(extent, dict):
                if not all(k in extent for k in ("xmin", "xmax", "ymin", "ymax")):
                    raise ValueError(
                        "Extent dictionary missing one or more of these keys: 'xmin', 'xmax', 'ymin', 'ymax'"
                    )
                if "spatialReference" not in extent:
                    try:
                        extent["spatialReference"] = self._story._properties[
                            "resources"
                        ][self.resource_node]["data"]["extent"]["spatialReference"]
                    except:
                        extent["spatialReference"] = {"wkid": 4326}

                # In order to correctly edit, the viewpoint, extent, and center must be updated.
                # update extent
                self._story._properties["nodes"][self.node]["data"]["extent"] = extent
                # update center
                center_x = (extent["xmin"] + extent["xmax"]) / 2
                center_y = (extent["ymin"] + extent["ymax"]) / 2
                self._story._properties["nodes"][self.node]["data"]["center"] = {
                    "spatialReference": extent["spatialReference"],
                    "x": center_x,
                    "y": center_y,
                }
                # update viewpoint
                self._story._properties["nodes"][self.node]["data"]["viewpoint"][
                    "targetGeometry"
                ] = self._story._properties["nodes"][self.node]["data"]["center"]

                change_made = True
        # set new scale if specified
        if scale:
            if isinstance(scale, Scales):
                self._story._properties["nodes"][self.node]["data"]["viewpoint"][
                    "scale"
                ] = scale.value["scale"]
                self._story._properties["nodes"][self.node]["data"][
                    "zoom"
                ] = scale.value["zoom"]
                change_made = True
            elif isinstance(scale, dict):
                self._story._properties["nodes"][self.node]["data"]["viewpoint"][
                    "scale"
                ] = scale["scale"]
                self._story._properties["nodes"][self.node]["data"]["zoom"] = scale[
                    "zoom"
                ]

        if change_made:
            # Once the update made, remove the original information from resources
            new_data = {
                "itemId": rdata_dict["itemId"],
                "itemType": rdata_dict["itemType"],
                "type": "minimal",
            }
            self._story._properties["resources"][self.resource_node]["data"] = new_data

        return self._story._properties["nodes"][self.node]["data"]["viewpoint"]

    # ----------------------------------------------------------------------
    @property
    def show_legend(self):
        """Get/Set the showing legend toggle. True if enabled and False if disabled"""
        if self._existing is True:
            if "isShowingLegend" in self._story._properties["nodes"][self.node]["data"]:
                return self._story._properties["nodes"][self.node]["data"][
                    "isShowingLegend"
                ]
            else:
                return False

    # ----------------------------------------------------------------------
    @show_legend.setter
    def show_legend(self, value: bool):
        self._story._properties["nodes"][self.node]["data"]["isShowingLegend"] = value

    # ----------------------------------------------------------------------
    @property
    def legend_pinned(self):
        """
        Get/Set the legend pinned toggle. True if enabled and False if disabled.

        .. note::
            If set to True, make sure `show_legend` is also True. Otherwise, you will not
            see the legend pinned.
        """
        if self._existing is True:
            if "legendPinned" in self._story._properties["nodes"][self.node]["data"]:
                return self._story._properties["nodes"][self.node]["data"][
                    "legendPinned"
                ]
            else:
                return False

    # ----------------------------------------------------------------------
    @legend_pinned.setter
    def legend_pinned(self, value: bool):
        self._story._properties["nodes"][self.node]["data"]["legendPinned"] = value

    # ----------------------------------------------------------------------
    @property
    def show_search(self):
        """Get/Set the search toggle. True if enabled and False if disabled"""
        if self._existing is True:
            if "search" in self._story._properties["nodes"][self.node]["data"]:
                return self._story._properties["nodes"][self.node]["data"]["search"]
            else:
                return False

    # ----------------------------------------------------------------------
    @show_search.setter
    def show_search(self, value: bool):
        self._story._properties["nodes"][self.node]["data"]["search"] = value

    # ----------------------------------------------------------------------
    @property
    def time_slider(self):
        """Get/Set the time slider toggle. True if enabled and False if disabled"""
        if self._existing is True:
            if "time_slider" in self._story._properties["nodes"][self.node]["data"]:
                return self._story._properties["nodes"][self.node]["data"]["timeSlider"]
            else:
                return False

    # ----------------------------------------------------------------------
    @time_slider.setter
    def time_slider(self, value: bool):
        self._story._properties["nodes"][self.node]["data"]["timeSlider"] = value

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the map.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Map.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption
            return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the map.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Map.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
            return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set the display type of the map.

        Values: `standard` | `wide` | `full` | `float right` | `float left`
        """
        if self._existing is True:
            if "config" in self._story._properties["nodes"][self.node]:
                return self._story._properties["nodes"][self.node]["config"]["size"]
            else:
                return None

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._existing is True:
            if "float" in display.lower():
                self._story._properties["nodes"][self.node]["config"]["size"] = "float"
                if "right" in display.lower():
                    self._story._properties["nodes"][self.node]["config"][
                        "floatAlignment"
                    ] = "end"
                else:
                    self._story._properties["nodes"][self.node]["config"][
                        "floatAlignment"
                    ] = "start"
            else:
                self._story._properties["nodes"][self.node]["config"][
                    "size"
                ] = display.lower()
                self._story._properties["nodes"][self.node]["config"].pop(
                    "floatAlignment", None
                )
            return self.display

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_map(self, caption=None, alt_text=None, display=None, story=None):
        self._story = story
        self._existing = True
        # Create webmap nodes
        # This represents the map as seen in the story
        self._story._properties["nodes"][self.node] = {
            "type": "webmap",
            "data": {
                "map": self.resource_node,
                "caption": "" if caption is None else caption,
                "alt": "" if alt_text is None else alt_text,
            },
            "config": {"size": display},
        }
        # Create resource node
        # This represents the original map item and it's properties
        self._story._properties["resources"][self.resource_node] = {
            "type": "webmap",
            "data": {
                "extent": self._extent,
                "center": self._center,
                "zoom": self._zoom,
                "mapLayers": self._map_layers,
                "viewpoint": self._viewpoint,
                "itemId": self._path.id,
                "itemType": self._type,
                "type": "minimal",
            },
        }

        # Add for Web Scene
        if self._type == "Web Scene":
            self._story._properties["resources"][self.resource_node]["data"][
                "lightingDate"
            ] = self._lighting_date
            self._story._properties["resources"][self.resource_node]["data"][
                "camera"
            ] = self._camera
            self._story._properties["nodes"][self.node]["data"][
                "lightingDate"
            ] = self._lighting_date
            self._story._properties["nodes"][self.node]["data"]["camera"] = self._camera

    # ----------------------------------------------------------------------
    def _update_map(self, map):
        new_map = Map(map)
        # Check for error.
        if (
            new_map._type
            != self._story._properties["resources"][self.resource_node]["data"][
                "itemType"
            ]
        ):
            raise ValueError("New Map must be of same type as the exisiting map.")

        # Get all the old properties but update with new map where needed

        # remove old resource node
        self._story._properties["resources"][
            new_map.resource_node
        ] = self._story._properties["resources"].pop(self.resource_node)
        # assign new resource node
        self.resource_node = new_map.resource_node
        # set the new item id in the story resources dictionary for this resource
        self._story._properties["resources"][new_map.resource_node]["data"][
            "itemId"
        ] = new_map._path.id
        # set the new map layers in the story resources dict for this resource
        self._story._properties["resources"][new_map.resource_node]["data"][
            "mapLayers"
        ] = new_map._map_layers
        # Update path to resource node in the node dictionary
        self._story._properties["nodes"][self.node]["data"][
            "map"
        ] = new_map.resource_node

        # Extra necessary updates when it is a Web Scene (3D Map)
        if self._type == "Web Scene":
            self._story._properties["resources"][self.resource_node]["data"][
                "lightingDate"
            ] = new_map._lighting_date
            self._story._properties["resources"][self.resource_node]["data"][
                "camera"
            ] = new_map._camera
            self._story._properties["nodes"][self.node]["data"][
                "lightingDate"
            ] = new_map._lighting_date
            self._story._properties["nodes"][self.node]["data"][
                "camera"
            ] = new_map._camera

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Text:
    """
    Class representing a `text` and a style of text.

    .. note::
        Once you create a Text instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    text                    Required String. The text that will be shown in the story.

                            .. code-block:: python

                                # Usage Example for paragraph:

                                >>> text = Text('''Paragraph with <strong>bold</strong>, <em>italic</em>
                                                and <a href=\"https://www.google.com\" rel=\"noopener noreferrer\"
                                                target=\"_blank\">hyperlink</a> and a <span
                                                class=\"sm-text-color-080\">custom color</span>''')

                                # Usage Example for numbered list:

                                >>> text = Text("<li>List Item1</li> <li>List Item2</li> <li>List Item3</li>")

    ------------------      --------------------------------------------------------------------
    style                   Optional TextStyles type. There are 7 different styles of text that can be
                            added to a story.

                            Values: PARAGRAPH | LARGEPARAGRAPH | NUMBERLIST | BULLETLIST |
                            HEADING | SUBHEADING | QUOTE
    ------------------      --------------------------------------------------------------------
    custom_color            Optional String. The hex color value without the #.
                            Only available when type is either 'paragraph', 'bullet-list', or
                            'numbered-list'.


                            Ex: custom_color = "080"
    ==================      ====================================================================


    Properties of the different text types:

    ===================     ====================================================================
    **Type**                **Text**
    -------------------     --------------------------------------------------------------------
    paragraph               String can contain the following tags for text formatting:
                            <strong>, <em>, <a href="{link}" rel="noopener noreferer" target="_blank"
                            and a class attribute to indicate color formatting:
                            class=sm-text-color-{values} attribute in the <strong> | <em> | <a> | <span> tags

                            Values: `themeColor1` | `themeColor2` | `themeColor3` | `customTextColors`
    -------------------     --------------------------------------------------------------------
    large-paragraph         String can contain the following tags for text formatting:
                            <strong>, <em>, <a href="{link}" rel="noopener noreferer" target="_blank"
                            and a class attribute to indicate color formatting:
                            class=sm-text-color-{values} attribute in the <strong> | <em> | <a> | <span> tags

                            Values: `themeColor1` | `themeColor2` | `themeColor3` | `customTextColors`
    -------------------     --------------------------------------------------------------------
    heading                 String can only contain <em> tag
    -------------------     --------------------------------------------------------------------
    subheading              String can only contain <em> tag
    -------------------     --------------------------------------------------------------------
    bullet-list             String can contain the following tags for text formatting:
                            <strong>, <em>, <a href="{link}" rel="noopener noreferer" target="_blank"
                            and a class attribute to indicate color formatting:
                            class=sm-text-color-{values} attribute in the <strong> | <em> | <a> | <span> tags

                            Values: `themeColor1` | `themeColor2` | `themeColor3` | `customTextColors`
    -------------------     --------------------------------------------------------------------
    numbered-list           String can contain the following tags for text formatting:
                            <strong>, <em>, <a href="{link}" rel="noopener noreferer" target="_blank"
                            and a class attribute to indicate color formatting:
                            class=sm-text-color-{values} attribute in the <strong> | <em> | <a> | <span> tags

                            Values: `themeColor1` | `themeColor2` | `themeColor3` | `customTextColors`
    -------------------     --------------------------------------------------------------------
    quote                   String can only contain <strong> and <em> tags
    ===================     ====================================================================

    """

    def __init__(
        self,
        text: Optional[str] = None,
        style: TextStyles = TextStyles.PARAGRAPH,
        color: str = None,
        **kwargs,
    ):
        # Can be created from scratch or already exist in story
        # Text is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "text"
        self.node = kwargs.pop("node_id", None)
        # Check if node exists in story else create new instance.
        self._existing = self._check_node()
        if self._existing is True:
            self._text = self._story._properties["nodes"][self.node]["data"]["text"]
            self._style = self._story._properties["nodes"][self.node]["data"]["type"]
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._text = text
            if isinstance(style, TextStyles):
                self._style = style.value

            # Color only applies certain styles
            if self._style in [
                "paragraph",
                "large-paragraph",
                "bullet-list",
                "numbered-list",
            ]:
                self._color = color
            else:
                self._color = None

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Text"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Text"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get the properties for the text.

        :return:
            The Text dictionary for the node.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
            }

    # ----------------------------------------------------------------------
    @property
    def text(self):
        """
        Get/Set the text itself for the text node.

        ==================  ==================================================
        **Parameter**        **Description**
        ------------------  --------------------------------------------------
        text                Optional String. The new text to be displayed.
        ==================  ==================================================

        :return:
            The text for the node.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["text"]

    # ----------------------------------------------------------------------
    @text.setter
    def text(self, text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["text"] = text
            return self.text

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_text(self, story=None):
        self._story = story
        self._existing = True
        self._story._properties["nodes"][self.node] = {
            "type": "text",
            "data": {
                "type": self._style,
                "text": self._text,
            },
        }
        if self._color is not None:
            self._story._properties["nodes"][self.node]["data"]["customTextColors"] = [
                self._color
            ]

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Check if node exists
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Button:
    """
    Class representing a `button`.

    .. note::
        Once you create a Button instance you must add it to the story to be able to edit it further.

    ==================      ====================================================================
    **Parameter**            **Description**
    ------------------      --------------------------------------------------------------------
    link                    Required String. When user clicks on button, they will be brought to
                            the link.
    ------------------      --------------------------------------------------------------------
    text                    Required String. The text that shows on the button.
    ==================      ====================================================================

    """

    def __init__(
        self, link: Optional[str] = None, text: Optional[str] = None, **kwargs
    ):
        # Can be created from scratch or already exist in story
        # Button is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "button"
        self.node = kwargs.pop("node_id", None)
        # Check if node exists else create new instance
        self._existing = self._check_node()
        if self._existing is True:
            self._link = self._story._properties["nodes"][self.node]["data"]["link"]
            self._text = self._story._properties["nodes"][self.node]["data"]["text"]
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._link = link
            self._text = text

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Button"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Button"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get the properties for the button.

        :return:
            The Button dictionary for the node.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return {"node_dict": self._story._properties["nodes"][self.node]}

    # ----------------------------------------------------------------------
    @property
    def text(self):
        """
        Get/Set the text for the button.

        ==================  ==================================================
        **Parameter**        **Description**
        ------------------  --------------------------------------------------
        text                Optional String. The new text to be displayed.
        ==================  ==================================================

        :return:
            The text for the node.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["text"]

    # ----------------------------------------------------------------------
    @text.setter
    def text(self, text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["text"] = text
            return self.text

    # ----------------------------------------------------------------------
    @property
    def link(self):
        """
        Get/Set the link for the button.

        ==================  ==================================================
        **Parameter**        **Description**
        ------------------  --------------------------------------------------
        link                Optional String. The new path for the button.
        ==================  ==================================================

        :return:
            The link being used.
            If nothing is returned, make sure the content is part of the story.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["link"]

    # ----------------------------------------------------------------------
    @link.setter
    def link(self, link):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["link"] = link
            return self.link

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _add_button(self, story):
        self._story = story
        self._existing = True
        self._story._properties["nodes"][self.node] = {
            "type": "button",
            "data": {"text": self._text, "link": self._link},
        }

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Check if node exists
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Gallery:
    """
    Class representing an `image gallery`

    To begin with a new gallery, simply call the class. Once added to the story,
    you can add up to 12 images.

    .. note::
        Once you create a Gallery instance you must add it to the story to be able to edit it further.

    .. code-block:: python

        # Images to add to the gallery.
        >>> image1 = Image(<url or path>)
        >>> image2 = Image(<url or path>)
        >>> image3 = Image(<url or path>)

        # Create a gallery and add to story before adding images to it.
        >>> gallery = Gallery()
        >>> my_story.add(gallery)
        >>> gallery.add_images([image1, image2, image3])
    """

    def __init__(self, **kwargs):
        # Can be created from scratch or already exist in story
        # Gallery is not an immersive node
        self._story = kwargs.pop("story", None)
        self._type = "gallery"
        self.node = kwargs.pop("node_id", None)
        # Check if node exists, else create new empty instance
        self._existing = self._check_node()
        if self._existing is True:
            self._children = self._story._properties["nodes"][self.node]["children"]
        else:
            # Create new empty instance
            self._children = []
            self.node = "n-" + uuid.uuid4().hex[0:6]

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Image Gallery"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Image Gallery"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties of the Gallery object

        :return:
            A dictionary depicting the node in the story.
            If nothing is returned, make sure the gallery is part of the story.
        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
            }

    # ----------------------------------------------------------------------
    @property
    def images(self):
        """
        Get/Set list of image nodes in the image gallery. Setting the lists allows the images
        to be reordered.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        images                  List of node ids for the images in the gallery. Nodes must already be
                                in the gallery and this list will adjust the order of the images.

                                To add new images to the gallery use:
                                    Gallery.add_images(images)
                                To delete an image from a gallery use:
                                    Gallery.delete_image(node_id)
        ==================      ====================================================================

        :return:
            A list of node ids in order of image appearance in the gallery.
            If nothing is returned, make sure the gallery is part of the story.
        """
        if self._existing:
            # Update incase addition or removal was made in between last check.
            self._children = self._story._properties["nodes"][self.node]["children"]
            images = []
            for child in self._children:
                images.append(Image(story=self._story, node_id=child))
            return images
        else:
            raise Warning(
                "Image Gallery must be added to the story before adding Images."
            )

    # ----------------------------------------------------------------------
    @images.setter
    def images(self, images):
        if self._existing:
            if images != self.images:
                raise ValueError(
                    "You cannot add or remove images through this method, only rearrange them."
                )
            children = []
            for image in images:
                children.append(image.node)
            self._children = children
            self._story._properties["nodes"][self.node]["children"] = children
        return self.images

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the swipe.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Gallery.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        return self._story._properties["nodes"][self.node]["data"]["caption"]

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if isinstance(caption, str):
            self._story._properties["nodes"][self.node]["data"]["caption"] = caption
        return self.caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the swipe.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Gallery.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        return self._story._properties["nodes"][self.node]["data"]["alt"]

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text
        return self.alt_text

    # ----------------------------------------------------------------------
    @property
    def display(self):
        """
        Get/Set the display type of the Gallery.

        Values: `jigsaw` | `square-dynamic`
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["config"]["size"]

    # ----------------------------------------------------------------------
    @display.setter
    def display(self, display):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["config"]["size"] = display
            return self.display

    # ----------------------------------------------------------------------
    def add_images(self, images: list[Image]):
        """
        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        images                  Required list of images of type Image.
        ==================      ====================================================================
        """
        if self._existing:
            if len(self.images) == 12:
                raise Warning(
                    "Maximum amount of images permitted is 12. Use Gallery.delete(image_node) to remove images before adding."
                )
            if images is not None:
                for image in images:
                    if image.node in self._story._properties["nodes"]:
                        image.node = "n-" + uuid.uuid4().hex[0:6]
                    image._add_image(story=self._story)
                    self._story._properties["nodes"][self.node]["children"].append(
                        image.node
                    )
        return self.images

    # ----------------------------------------------------------------------
    def delete_image(self, image: str | Image):
        """
        The delete_image method is used to delete one image from the gallery. To see a list of images
        used in the gallery, use the :meth:`~arcgis.apps.storymap.story_content.Gallery.images` property.

        ==================      ====================================================================
        **Parameter**            **Description**
        ------------------      --------------------------------------------------------------------
        image                   Required String. The node id for the image to be removed from the gallery or the Image instance.
        ==================      ====================================================================

        :return: The current list of images in the gallery.
        """
        if isinstance(image, Image):
            image = image.node
        if image in self.images:
            # Remove from the gallery list
            self._story._properties["nodes"][self.node]["children"].remove(image)
            self._story._delete(image)
        return self.images

    # ----------------------------------------------------------------------
    def _add_gallery(self, caption=None, alt_text=None, display=None, story=None):
        self._story = story
        self._existing = True
        # Create image nodes
        self._story._properties["nodes"][self.node] = {
            "type": "gallery",
            "data": {
                "galleryLayout": display if display is not None else "jigsaw",
                "caption": "" if caption is None else caption,
                "alt": "" if alt_text is None else alt_text,
            },
            "children": self._children,
        }

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        if self._existing is True:
            return self._story._delete(self.node)
        else:
            return False

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Swipe:
    """
    Create an Swipe node.

    .. note::
        Once you create a Swipe instance you must add it to the story to be able to edit it further.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    node_id             Required String. The node id for the swipe type.
    ---------------     --------------------------------------------------------------------
    story               Required :class:`~arcgis.apps.storymap.story.StoryMap` that the swipe belongs to.
    ===============     ====================================================================

    .. code-block:: python

        >>> my_story.nodes #use to find swipe node id

        # Method 1: Use the Swipe Class
        >>> swipe = Swipe()

        # Method 2: Use the get method in story
        >>> swipe = my_story.get(node = <node_id>)

    """

    def __init__(self, **kwargs):
        self._story = kwargs.pop("story", None)
        self._type = "swipe"
        if "node" in kwargs:
            # legacy
            self.node = kwargs.pop("node", None)
        else:
            self.node = kwargs.pop("node_id", None)

        # Find the type of media that the swipe supports.
        # Both contents are of the same type so only need to look at one.
        # Check if node exists else create new instance
        self._existing = self._check_node()
        if self._existing is True:
            if "data" in self._story._properties["nodes"][self.node]:
                self._slides = self._story._properties["nodes"][self.node]["data"][
                    "contents"
                ]
                # get the media for the swipe (image or map)
                media_node = self._story._properties["nodes"][self.node]["data"][
                    "contents"
                ]["0"]

                # Find the type, this is important since swipe must only have one media type
                if media_node == "":
                    # First position is empty
                    # Check the second position
                    second_media = self._story._properties["nodes"][self.node]["data"][
                        "contents"
                    ]["1"]
                    if second_media == "":
                        # No media set yet, type is empty
                        self._media_type = ""
                    else:
                        self._media_type = self._story._properties["nodes"][
                            second_media
                        ]["type"]
                else:
                    # Use the media type of the first position
                    self._media_type = self._story._properties["nodes"][media_node][
                        "type"
                    ]

            else:
                # Empty swipe node
                self._slides = []
                self._media_type = ""
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._slides = []
            self._media_type = ""

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Swipe"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Swipe"

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        Get properties of the Swipe object

        :return:
            A dictionary depicting the node in the story.
        """
        if self._existing is True:
            return {
                "node_dict": self._story._properties["nodes"][self.node],
            }
        else:
            return None

    # ----------------------------------------------------------------------
    @property
    def caption(self):
        """
        Get/Set the caption property for the swipe.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        caption             String. The new caption for the Swipe.
        ==================  ========================================

        :return:
            The caption that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["caption"]
        else:
            return None

    # ----------------------------------------------------------------------
    @caption.setter
    def caption(self, caption):
        if self._existing is True:
            if isinstance(caption, str):
                self._story._properties["nodes"][self.node]["data"]["caption"] = caption

    # ----------------------------------------------------------------------
    @property
    def alt_text(self):
        """
        Get/Set the alternte text property for the swipe.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        alt_text            String. The new alt_text for the Swipe.
        ==================  ========================================

        :return:
            The alternate text that is being used.
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["alt"]
        else:
            return None

    # ----------------------------------------------------------------------
    @alt_text.setter
    def alt_text(self, alt_text):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["alt"] = alt_text

    # ----------------------------------------------------------------------
    def edit(
        self,
        content: Optional[Union[Image, Map]] = None,
        position: str = "right",
    ):
        """
        Edit the media content of a Swipe item. To save your edits and see them
        in the StoryMap's builder, make sure to save the story.

        Use this method to add new content if your swipe is empty. You can specify the content to add
        and which side of the swipe it should be on.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        content             Required story content of type: :class:`~arcgis.apps.storymap.story_content.Image`
                            or :class:`~arcgis.apps.storymap.story_content.Map`. Must be the same media type
                            on both panels.
        ---------------     --------------------------------------------------------------------
        position            Optional String. Either "right" or "left". Default is "right" so content
                            will be added to right panel.
        ===============     ====================================================================

        :return: True if successful
        """
        if self._existing is True:
            # Media type must be same for right and left slide.
            if isinstance(content, Image) and self._media_type == "webmap":
                raise ValueError(
                    "Media type is established as webmap. Can only accept another webmap."
                )
            if isinstance(content, Map) and self._media_type == "image":
                raise ValueError(
                    "Media type is established as image. Can only accept another image."
                )
            # Add node to story.
            self._add_item_story(content)

            if "data" not in self._story._properties["nodes"][self.node]:
                self._story._properties["nodes"][self.node]["data"] = {"contents": {}}
            # Add to content in position wanted
            if position == "left":
                self._story._properties["nodes"][self.node]["data"]["contents"][
                    "0"
                ] = content.node
            else:
                self._story._properties["nodes"][self.node]["data"]["contents"][
                    "1"
                ] = content.node
            return True
        else:
            raise KeyError(
                "The instance of Swipe must first be added to the story before you can start editing."
            )

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        if self._existing is True:
            return self._story._delete(self.node)
        else:
            return False

    # ----------------------------------------------------------------------
    def _add_swipe(self, caption=None, alt_text=None, display=None, story=None):
        self._story = story
        self._existing = True
        # Create swipe node
        self._story._properties["nodes"][self.node] = {
            "type": "swipe",
            "data": {
                "contents": {"0": "", "1": ""},
                "caption": "" if caption is None else caption,
                "alt": "" if alt_text is None else alt_text,
            },
        }
        if display:
            self._story._properties["nodes"][self.node]["config"] = {"size": display}

    # ----------------------------------------------------------------------
    def _add_item_story(self, content):
        if content and content.node in self._story._properties["nodes"]:
            content.node = "n-" + uuid.uuid4().hex[0:6]
        if isinstance(content, Image):
            content._add_image(story=self._story)
            self._media_type = "image"
        elif isinstance(content, Map):
            content._add_map(story=self._story)
            self._media_type = "webmap"

    # ----------------------------------------------------------------------
    def _check_node(self):
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Sidecar:
    """
    Create an Sidecar immersive object.

    A sidecar is composed of slides. Slides are composed of two sub structures: a narrative panel and a media panel.
    The media node can be a(n): Image, Video, Embed, Map, or Swipe.
    The narrative panel can contain mulitple types of content including Image, Video, Embed, Button, Text, Map, and more.

    .. note::
        Once you create a Sidecar instance you must add it to the story to be able to edit it further.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    style               Optional string that depicts the sidecar style.
                        Values: 'floating-panel' | 'docked-panel' | 'slideshow'
    ===============     ====================================================================

    .. code-block:: python

        >>> my_story.nodes #use to find sidecar node id

        # Method 1: Use the Sidecar Class
        >>> sidecar = Sidecar("floating-panel") # create from scratch

        # Method 2: Use the get method in story
        >>> sidecar = my_story.content_list()[3] # sidecar is fourth item in story
    """

    def __init__(self, style: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        self._story = kwargs.pop("story", None)
        self._type = "immersive"
        if "node" in kwargs:
            # legacy
            self.node = kwargs.pop("node", None)
        else:
            self.node = kwargs.pop("node_id", None)
        # Check if node exists else create new instance
        self._existing = self._check_node()
        if self._existing is True:
            self._style = self._story._properties["nodes"][self.node]["data"]["subtype"]
            self._slides = self._story._properties["nodes"][self.node]["children"]
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._style = style if style else "floating-panel"
            self._slides = []

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Sidecar"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Sidecar"

    # ----------------------------------------------------------------------
    def _add_sidecar(
        self,
        story=None,
    ):
        # Add the story to the node
        self._story = story
        self._existing = True
        # Create timeline nodes
        self._story._properties["nodes"][self.node] = {
            "type": "immersive",
            "data": {
                "type": "sidecar",
                "subtype": self._style,
                "narrativePanelPosition": "start",
                "narrativePanelSize": "medium",
            },
            "children": [],
        }

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        List all slides and their children for a Sidecar node.

        :return:
            A list where the first item is the node id for the sidecar. Next
            items are slides with the dictionary their children.
        """
        sidecar_tree = []
        for slide in self._slides:
            narrative_panel = self._story._properties["nodes"][slide]["children"][0]
            children = (
                self._story._properties["nodes"][narrative_panel]["children"]
                if "children" in self._story._properties["nodes"][narrative_panel]
                else ""
            )
            narrative_children = []
            for child in children:
                info = self._story._properties["nodes"][child]
                narrative_children.append({info["type"]: child})

            # there will always be a narrative panel node but not always a media node
            if len(self._story._properties["nodes"][slide]["children"]) == 2:
                media_item = self._story._properties["nodes"][slide]["children"][1]
                media_type = self._story._properties["nodes"][media_item]["type"]
            else:
                media_item = ""
                media_type = ""

            # construct tree like structure
            sidecar_tree.append(
                {
                    slide: {
                        "narrative_panel": {
                            "panel": narrative_panel,
                            "children": narrative_children,
                        },
                        "media": {media_type: media_item},
                    }
                }
            )
        return sidecar_tree

    # ----------------------------------------------------------------------
    @property
    def content_list(self):
        """
        Get a list of all the content within the sidecar in order of appearance.
        The content will be displayed in the following order:
        A list of the content in slide 1, a list of the content in slide 2, etc.
        Each sub-list will contain content found in the narrative panel, if any, and the media content, if any.
        """
        contents = []
        # get the values from the nodes list and return only these
        sidecar_dict = self.properties
        for slide in sidecar_dict:
            content = []
            # get the entire slide dict
            slide_dict = list(slide.values())[0]
            if (
                "narrative_panel" in slide_dict
                and "children" in slide_dict["narrative_panel"]
                and len(slide_dict["narrative_panel"]["children"]) > 0
            ):
                # Get the content that are children of the narrative panel
                children = slide_dict["narrative_panel"]["children"]
                for child in children:
                    # Get each class from the node value
                    content.append(self.get(list(child.values())[0]))
            if "media" in slide_dict and (
                slide_dict["media"] is not None or slide_dict["media"] != {}
            ):
                # Get the media content for the slide
                media = list(slide_dict["media"].values())[0]

                if media == None or media == "":
                    pass
                else:
                    # Get the class using the node value
                    content.append(self.get(media))
            contents.append(content)
        return contents

    # ----------------------------------------------------------------------
    def edit(
        self,
        content: Union[Image, Video, Map, Embed],
        slide_number: int,
    ):
        """
        Edit method can be used to edit the **type** of media in a slide of the Sidecar.
        This is done by specifying the slide number and the media content to be added.
        The media can only be of type: Image, Video, Map, or Embed.

        .. note::
            This method should not be used to edit the narrative panel of the Sidecar. To better edit both
            the media and the narrative panel, it is recommended to use the :func:`~Sidecar.get` method
            in the Sidecar class. The `get` method can be used to change media if the content is of the same
            type as what is currently present and preserve the node_id.


        ==================      =======================================================================
        **Parameter**            **Description**
        ------------------      -----------------------------------------------------------------------
        content                 Required item that is a story content item.
                                Item type for the media node can be: :class:`~arcgis.apps.storymap.story_content.Image`,
                                :class:`~arcgis.apps.storymap.story_content.Video`, :class:`~arcgis.apps.storymap.story_content.Map`
                                :class:`~arcgis.apps.storymap.story_content.Embed`, :class:`~arcgis.apps.storymap.story_content.Swipe`
        ------------------      -----------------------------------------------------------------------
        slide_number            Required Integer. The slide that will be edited. First slide is 1.
        ==================      =======================================================================

        .. code-block:: python

            # Get sidecar from story and see the properties
            sc = story.get(<sidecar_node_id>)
            sc.properties
            >> returns a dictionary structure of the sidecar

            # If a slide 2 contains a map and you want to change it to an image
            im = Image(<img_url_or_path>)
            sc.edit(im, 2)
            sc.properties
            >> notice slide 2 now has an image

            # If I want to update the image then 2 methods:
            # OPTION 1
            im2 = Image(<img_url_or_path>)
            sc.edit(im2, 2)

            # OPTION 2 (only applicable if content is of same type as existing)
            im2 = sc.get(im.node_id)
            im2.image = <img_url_or_path>

        """
        # Find media child
        slide = self.properties[slide_number - 1]
        slide_node = list(slide.keys())[0]
        media_node = list(slide[slide_node]["media"].values())[0]

        # Add to node properties
        self._add_item_story(content)

        if media_node:
            self._story._delete(media_node)
        self._story._properties["nodes"][slide_node]["children"].insert(1, content.node)

    # ----------------------------------------------------------------------
    def get(self, node_id: str):
        """
        The get method is used to get the node that will be edited. Use `sidecar.properties` to
        find all nodes associated with the sidecar.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        node_id             Required String. The node id for the content that will be returned.
        ===============     ====================================================================

        :return: An class instance of the node type.

        .. code-block:: python

            # Find the nodes associated with the sidecar
            sc = story.get(<sidecar_node_id>)
            sc.properties
            >> returns a dictionary structure of the sidecar

            # Get a node associated with the sidecar, in this example an image, and change the image
            im = sc.get(<node_id>)
            im.image = <new_image_path>

            # Save the story to see changes applied in Story Map builder
            story.save()

        """
        return self._story._assign_node_class(node_id)

    # ----------------------------------------------------------------------
    def add_action(
        self,
        slide_number: int,
        text: str,
        viewpoint: dict,
        extent: dict | None = None,
        map_layers: list[dict] | None = None,
    ):
        """
        Add a map action button to a slide. You can specify the data of the action.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        slide_number        Required Integer. The slide that the map action will be added to. First slide is 1.
        ---------------     --------------------------------------------------------------------
        text                Required String. The map action button text
        ---------------     --------------------------------------------------------------------
        viewpoint           Required Dictionary. The viewpoint to be set. The minimum keys to include are
                            an x and y center point in the target geometry.

                            Example:
                                viewpoint = {
                                    "rotation": 0,
                                    "scale": 18055.954822,
                                    "targetGeometry": {
                                        "spatialReference": {
                                            "latestWkid": 3857,
                                            "wkid": 102100
                                        },
                                        "x": -8723429.856341356,
                                        "y": 4019095.847955684
                                    }
                                }
        ---------------     --------------------------------------------------------------------
        extent              Optional Dictionary. The extent of the map that will be shown when
                            the action button is used.

                            Example:
                                extent = {
                                    "spatialReference": {
                                        "latestWkid": 3857,
                                        "wkid": 102100
                                    },
                                    "xmin": -8839182.968379805,
                                    "ymin": 3907027.5240857545,
                                    "xmax": -8824335.075635428,
                                    "ymax": 3915378.269425899
                                }
        ---------------     --------------------------------------------------------------------
        map_layers          Optional list of dictionaries. Each dictionary represents a map layer
                            and the parameters set on the map layer.

                            Example:
                                map_layers = [
                                    {
                                        "id": "18511776c33-layer-2",
                                        "title": "USA Forest Type",
                                        "visible": true
                                    }
                                ]
        ===============     ====================================================================

        :return: The node id for the action that was added to the slide
        """
        # create node for the action
        node = "n-" + uuid.uuid4().hex[0:6]

        # find the target map
        slide_node = self._slides[slide_number - 1]
        slide_dict = self.properties[slide_number][slide_node]
        if "media" not in slide_dict:
            raise ValueError(
                "The slide needs a webmap or expressmap for the map action to be created."
            )

        # get the map type
        map_type = list(slide_dict["media"].keys())[0]
        if map_type not in ["expressmap", "webmap"]:
            raise ValueError(
                "The slide needs a webmap or expressmap for the map action to be created."
            )
        map_node = slide_dict["media"][map_type]

        # compose the action dict
        action_dict = {
            "origin": node,
            "trigger": "ActionButton_Apply",
            "target": map_node,
            "event": "ExpressMap_UpdateData"
            if map_type == "expressmap"
            else "WebMap_UpdateData",
            "data": {},
        }
        if extent and not viewpoint:
            action_dict["data"]["extent"] = extent
            x_center = (extent["xmin"] + extent["xmax"]) / 2
            y_center = (extent["ymin"] + extent["ymax"]) / 2
            viewpoint = {
                "rotation": 0,
                "targetGeometry": {
                    "spatialReference": extent["spatialReference"]
                    if "spatialReference" in extent
                    else {"latestWkid": 3857, "wkid": 102100},
                    "x": x_center,
                    "y": y_center,
                },
            }
        if map_layers:
            action_dict["data"]["mapLayers"] = map_layers
        if viewpoint:
            action_dict["data"]["viewpoint"] = viewpoint

        # add to actions list in story properties
        if "actions" in self._story._properties:
            self._story._properties["actions"].append(action_dict)
        else:
            self._story._properties["actions"] = [action_dict]

        # compose the node dict in story properties
        self._story._properties["nodes"][node] = {
            "type": "action-button",
            "data": {"text": text},
            "config": {"size": "wide"},
        }

        # add to the narrative panel
        narrative_panel_node = slide_dict["narrative_panel"]["panel"]
        self._story._properties["nodes"][narrative_panel_node]["children"].append(node)

        return node

    # ----------------------------------------------------------------------
    def add_slide(
        self,
        contents: list | None = None,
        media: Image | Video | Map | Embed | None = None,
        slide_number: int = None,
    ):
        """
        Add a slide to the sidecar. You are able to specify the position of the slide, the
        content of the narrative panel and the media of the slide.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        contents                    Optional list of story content item(s). The instances of story content that
                                    will be added to the narrative panel such as Text, Image, Embed, etc.
        -----------------------     --------------------------------------------------------------------
        media                       Optional item that is a story content item.
                                    Item type for the media node can be: Image, Video, Map, Embed, or Swipe.
        -----------------------     --------------------------------------------------------------------
        slide_number                Optional Integer. The position at which the new slide will be.
                                    If none is provided then it will be added as the last slide.

                                    First slide is 1.
        =======================     ====================================================================

        .. code-block:: python

            # Get sidecar from story and see the properties
            sc = story.get(<sidecar_node_id>)
            sc.properties
            >> returns a dictionary structure of the sidecar

            # create the content we will add to narrative_panel_nodes parameter
            im = Image(<img_url_or_path>)
            txt = Text("Hello World")
            embed = Embed(<url>)
            narrative_nodes = [im, txt, embed]

            mmap = Map(<item_id webmap>)

            # Add new slide with the content:
            sc.add_slide(narrative_nodes, mmap, 4)
            >> New slide added with the content at position 4
        """
        # Loop to:
        # 1. Add the content to the story if not already added
        # 2. Add the node ids to list to pass as children later
        if self._existing:
            np_children = []
            if contents:
                for content in contents:
                    self._add_item_story(content)
                    np_children.append(content.node)
            if media:
                self._add_item_story(media)

            # For reference on some styles, grab first slide to go off of
            if len(self._slides) > 0:
                first_slide = self._story.properties["nodes"][self._slides[0]]
                first_np = self._story.properties["nodes"][first_slide["children"][0]]
                data = first_np["data"]  # keep same settings as other slide
            else:
                if self._style == "slideshow":
                    data = {"position": "start-top", "panelStyle": "themed"}
                else:
                    data = {
                        "position": "start",
                        "size": "small",
                        "panelStyle": "themed",
                    }

            # Create narrative panel node
            np_node = "n-" + uuid.uuid4().hex[0:6]
            np_def = {
                "type": "immersive-narrative-panel",
                "data": data,
                "children": np_children,
            }
            self._story._properties["nodes"][np_node] = np_def

            # Create slide node and add the other nodes to it
            slide_node = "n-" + uuid.uuid4().hex[0:6]
            slide_def = {
                "type": "immersive-slide",
                "data": {"transition": "fade"},
                "children": [np_node],  # First listed node is the Narrative Panel
            }
            # If no media given then put a background color instead
            if media:
                slide_def["children"].append(media.node)
            else:
                slide_def["data"]["backgroundColor"] = "#FFFFFF"
            self._story._properties["nodes"][slide_node] = slide_def

            # Add slide node to sidecar node children at position indicated or last.
            if slide_number is None:
                # If no slide number then insert slide last
                slide_number = len(self._slides) + 1
            else:
                # Correct for the indexing (user puts position 1, index is 0)
                slide_number = slide_number - 1
            self._story._properties["nodes"][self.node]["children"].insert(
                slide_number, slide_node
            )
            # Update slide definition for the class to relect new list
            self._slides = self._story._properties["nodes"][self.node]["children"]
            return {"New Slide": slide_node}
        else:
            return Exception(
                "The sidecar must first be added to a story before editing."
            )

    # ----------------------------------------------------------------------
    def remove_slide(self, slide: str):
        """
        Remove a slide from the sidecar.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        slide               Required String. The node id for the slide that will be removed.
        ===============     ====================================================================
        """
        # Remove slide and all associated children.
        self._remove_associated(slide)
        self._story._properties["nodes"][self.node]["children"].remove(slide)
        self._story._delete(slide)
        self._slides = self._story._properties["nodes"][self.node]["children"]
        return True

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _remove_associated(self, slide):
        # Get narrative panel, always first child of the slide
        narrative_panel = self._story._properties["nodes"][slide]["children"][0]
        # Delete the children of the narrative panel
        if "children" in self._story._properties["nodes"][narrative_panel]:
            children = self._story._properties["nodes"][narrative_panel]["children"]
            for child in children:
                self._story._delete(child)
        # Delete the narrative panel itself
        self._story._delete(narrative_panel)

        # Remove media item and resource node if one exists
        if len(self._story._properties["nodes"][slide]["children"]) >= 1:
            media_item = self._story._properties["nodes"][slide]["children"][0]
            self._story._delete(media_item)

    # ----------------------------------------------------------------------
    def _add_item_story(self, content):
        if content and content.node in self._story._properties["nodes"]:
            content.node = "n-" + uuid.uuid4().hex[0:6]
        if isinstance(content, Image):
            content._add_image(display="wide", story=self._story)
        elif isinstance(content, Video):
            content._add_video(display="wide", story=self._story)
        elif isinstance(content, Embed):
            content._add_link(display="card", story=self._story)
        elif isinstance(content, Map):
            content._add_map(display="wide", story=self._story)
        elif isinstance(content, Text):
            content._add_text(story=self._story)
        elif isinstance(content, Button):
            content._add_button(story=self._story)
        elif isinstance(content, Audio):
            content._add_audio(display="wide", story=self._story)
        elif isinstance(content, Timeline):
            content._add_timeline(story=self._story)
        elif isinstance(content, Swipe):
            content._add_swipe(story=self._story)

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Node is not in the story if no story or node id is present
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class Timeline:
    """
    Create a Timeline object from a pre-existing `timeline` node.

    A timeline is composed of events.
    Events are composed of maximum three nodes: an image, a sub-heading text, and a paragraph text.

    .. note::
        Once you create a Timeline instance you must add it to the story to be able to edit it further.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    style               Required string, the style type of the timeline. If the timeline will be
                        added to a Sidecar, then only `waterfall` and `single-sided` are allowed.


                        Values: 'waterfall' | 'single-side' | 'condensed'
    ===============     ====================================================================

    .. code-block:: python

        >>> my_story.nodes #use to find timeline node id

        # Method 1: Use the Timeline Class
        >>> timeline = Timeline(my_story, <node_id>)

        # Method 2: Use the get method in story
        >>> timeline = my_story.get(node = <node_id>)
    """

    def __init__(self, style: Optional[str] = None, **kwargs):
        # Can be created from scratch or already exist in story
        self._story = kwargs.pop("story", None)
        self._type = "timeline"
        if "node" in kwargs:
            # legacy
            self.node = kwargs.pop("node", None)
        else:
            self.node = kwargs.pop("node_id", None)
        # Check if node exists else create new instance
        self._existing = self._check_node()
        if self._existing is True:
            self._style = self._story._properties["nodes"][self.node]["data"]["type"]
            self._events = self._story._properties["nodes"][self.node]["children"]
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._style = style if style else "waterfall"
            self._events = []

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Timeline"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Timeline"

    # ----------------------------------------------------------------------
    def _add_timeline(
        self,
        story=None,
    ):
        # Add the story to the node
        self._story = story
        # Create timeline nodes
        self._story._properties["nodes"][self.node] = {
            "type": "timeline",
            "data": {
                "type": self._style,
            },
            "children": [],
        }
        self._existing = True

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """
        List all events and their children

        :return:
            A list where the first item is the node id for the timeline. Next
            items are dictionary of events and their children.
        """
        timeline = {self.node: {}}
        for event in self._events:
            timeline[self.node][event] = {}
            if "children" in self._story._properties["nodes"][event]:
                for child in self._story._properties["nodes"][event]["children"]:
                    node_type = self._story._properties["nodes"][child]["type"]
                    if node_type == "text":
                        node_type = self._story._properties["nodes"][child]["data"][
                            "type"
                        ]
                        if node_type == "h3":
                            node_type = "subheading"
                    timeline[self.node][event][node_type] = child
            else:
                node_type = self._story._properties["nodes"][event]["type"]
                timeline[self.node][event] = node_type
        return timeline

    # ----------------------------------------------------------------------
    @property
    def style(self) -> str:
        """
        Get/Set the style of the timeline

        Values: `waterfall` | `single-slide` | `condensed`
        """
        if self._existing is True:
            return self._story._properties["nodes"][self.node]["data"]["type"]
        else:
            return self._style

    # ----------------------------------------------------------------------
    @style.setter
    def style(self, style):
        if self._existing is True:
            self._story._properties["nodes"][self.node]["data"]["type"] = style
            self._style = style

    # ----------------------------------------------------------------------
    def edit(
        self,
        content: Union[Image, Text],
        event: int,
    ):
        """
        Edit event text or image content. To add a new event use the `add_event` method.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        content             Required content to replace current content.
                            Item type can be :class:`~arcgis.apps.storymap.story_content.Image` or :class:`~arcgis.apps.storymap.story_content.Text` .

                            Text can only be of style TextStyles.SUBHEADING or TextStyles.PARAGRAPH
        ---------------     --------------------------------------------------------------------
        event               Required Integer. The event that will be edited. First event is 1.
        ===============     ====================================================================
        """
        # Find children nodes
        event = self._events[event - 1]

        # Get position of new item, if None: needs to be added in.
        position = self._find_position_content(content, event)

        # Check to see if content has been added to node properties
        if content.node not in self._story._properties["nodes"]:
            self._add_item_story(content)

        # Insert new content
        if isinstance(content, Text):
            # Can either be the heading or subheading of the timeline.
            # Need to either replace old or add new if not already existing.
            if position is not None:
                old_text_node = self._story._properties["nodes"][event]["children"].pop(
                    position
                )
                self._story._delete(old_text_node)
                self._story._properties["nodes"][event]["children"].insert(
                    position, content.node
                )
            else:
                self._story._properties["nodes"][event]["children"].append(content.node)
        elif isinstance(content, Image):
            # Remove current image content and add new content if image already present
            if position is not None:
                old_image_node = self._story._properties["nodes"][event][
                    "children"
                ].pop(position)
                self._story._delete(old_image_node)
                self._story._properties["nodes"][event]["children"].insert(
                    position, content.node
                )
            else:
                # Image was not currently present so simply add
                self._story._properties["nodes"][event]["children"].append(content.node)

    # ----------------------------------------------------------------------
    def add_event(
        self, contents: list[Image | Text] | None = None, position: int | None = None
    ) -> bool:
        """
        Add event or spacer to the timeline.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        contents            Optional item list that will be in the event. Need to be passed in
                            by order of appearance.
                            Item type can be :class:`~arcgis.apps.storymap.story_content.Image` or :class:`~arcgis.apps.storymap.story_content.Text` .

                            Text can only be of style TextStyles.SUBHEADING or TextStyles.PARAGRAPH

                            .. note::
                                To create timeline spacer, do not pass in any value for this parameter.
        ---------------     --------------------------------------------------------------------
        position            Optional Integer. The position at which the even will be added. First event is 1.
                            If None, then the event will be added to the end.
        ===============     ====================================================================

        """
        if self._existing is True:
            # Check if able to add event (20 max)
            if len(self._events) == 20:
                raise ValueError(
                    "There is a maximum of 20 events allowed per timeline. To remove an event use the `remove_event` method."
                )

            event_node = "n-" + uuid.uuid4().hex[0:6]
            if position:
                self._story._properties["nodes"][self.node]["children"].insert(
                    position - 1, event_node
                )
            else:
                self._story._properties["nodes"][self.node]["children"].append(
                    event_node
                )
            if contents:
                contents_ids = []
                for content in contents:
                    # Check to see if content has been added to node properties
                    if content.node not in self._story._properties["nodes"]:
                        self._add_item_story(content)
                    contents_ids.append(content.node)
                self._story._properties["nodes"][event_node] = {
                    "type": "timeline-event",
                    "children": contents_ids,
                }
            else:
                self._story._properties["nodes"][event_node] = {
                    "type": "timeline-spacer"
                }

            # update self._events
            self._events = self._story._properties["nodes"][self.node]["children"]

            return True
        else:
            return Exception("The node must be part of a story before editing")

    # ----------------------------------------------------------------------
    def remove_event(self, event: str):
        """
        Remove an event from the timeline.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        event               Required String. The node id for the timeline event that will be removed.
        ===============     ====================================================================
        """
        self._remove_associated(event)
        self._story._properties["nodes"][self.node]["children"].remove(event)
        self._story._delete(event)
        return True

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the node

        :return: True if successful.
        """
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _remove_associated(self, event):
        # Remove narrative panel and text associated
        if "children" in self._story._properties["nodes"][event]:
            children = self._story._properties["nodes"][event]["children"]
            for child in children:
                self._story._delete(child)
            self._story._delete(event)

    # ----------------------------------------------------------------------
    def _find_position_content(self, content, event_node):
        # Find the position in which to insert the new content
        if isinstance(content, Text):
            content_type = "text"
            subtype = content._style
        elif isinstance(content, Image):
            content_type = "image"

        # Find the position of the node that corresponds to the content being added
        # If a user does not previously have a type of content, the position is None.
        for child in self._story._properties["nodes"][event_node]["children"]:
            if (
                self._story._properties["nodes"][child]["type"] == content_type
                and content_type == "image"
            ):
                position = self._story._properties["nodes"][event_node][
                    "children"
                ].index(child)
                return position
            elif (
                self._story._properties["nodes"][child]["type"] == content_type
                and self._story._properties["nodes"][child]["data"]["type"] == subtype
            ):
                position = self._story._properties["nodes"][event_node][
                    "children"
                ].index(child)
                return position
            else:
                # Content type doesn't exist yet and will need to be added in.
                position = None
        return position

    # ----------------------------------------------------------------------
    def _add_item_story(self, content):
        if content.node in self._story._properties["nodes"]:
            content.node = "n-" + uuid.uuid4().hex[0:6]
        if isinstance(content, Image):
            content._add_image(story=self._story)
        elif isinstance(content, Text):
            content._add_text(story=self._story)

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Node is not in the story if no story or node id is present
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class MapTour:
    """
    Create a MapTour object from a pre-existing `maptour` node.

    .. note::
        Once you create a MapTour instance you must add it to the story to be able to edit it further.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    node_id             Required String. The node id for the map tour type.
    ---------------     --------------------------------------------------------------------
    story               Required :class:`~arcgis.apps.storymap.story.StoryMap` that the map tour belongs to.
    ===============     ====================================================================

    .. code-block:: python

        >>> my_story.nodes #use to find map tour node id

        # Method 1: Use the MapTour Class
        >>> maptour = MapTour(my_story, <node_id>)

        # Method 2: Use the get method in story
        >>> maptour = my_story.get(node = <node_id>)
    """

    def __init__(self, **kwargs):
        # Content must already exist in the story
        # Map Tour is not an immersive node
        self._story = kwargs.pop("story", None)
        self.node = kwargs.pop("node_id", None)
        self._existing = self._check_node()

        if self._existing:
            self.map = self._story._properties["nodes"][self.node]["data"]["map"]
            if self._story._properties["nodes"][self.node]["type"] != "tour":
                raise Exception("This node is not of type tour.")
            self._type = self._story._properties["nodes"][self.node]["data"]["type"]
            self._subtype = self._story._properties["nodes"][self.node]["data"][
                "subtype"
            ]
            self._places = self._story._properties["nodes"][self.node]["data"]["places"]
        else:
            raise ValueError(
                "You cannot create a Map Tour from scratch at this time. Please use an existing Map Tour."
            )

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Map Tour"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Map Tour"

    # ----------------------------------------------------------------------
    @property
    def _children(self) -> list:
        """private method to gather all children of a map tour from places data"""
        children = [self.map]
        for place in self.places:
            if "children" in place and place["contents"]:
                for content in place["contents"]:
                    children.append(content)
            if "media" in place and place["media"]:
                children.append(place["media"])
            if "title" in place and place["title"]:
                children.append(place["title"])
        return children

    # ----------------------------------------------------------------------
    @property
    def style(self):
        """Get the type and subtype of the map tour"""
        return (
            self._story._properties["nodes"][self.node]["data"]["type"]
            + " - "
            + self._story._properties["nodes"][self.node]["data"]["subtype"]
        )

    # ----------------------------------------------------------------------
    @property
    def places(self):
        """
        List all places on the map
        """
        return self._story._properties["nodes"][self.node]["data"]["places"]

    # ----------------------------------------------------------------------
    def get(self, node_id: str):
        """
        The get method is used to get the node that will be edited. Use `maptour.properties` to
        find all nodes associated with the sidecar.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        node_id             Required String. The node id for the content that will be returned.
        ===============     ====================================================================

        :return: An class instance of the node type.

        .. code-block:: python

            # Find the nodes associated with the map tour
            mt = story.get(<maptour_node_id>)
            mt.places
            >> returns places of the map tour

            # Get a node associated with the map tour, in this example an image, and change the image
            im = mt.get(<node_id>)
            im.image = <new_image_path>

            # Save the story to see changes applied in Story Map builder
            story.save()

        """
        return self._story._assign_node_class(node_id)

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Node is not in the story if no story or node id is present
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################
class MapAction:
    """
    Within the sidecar block, there are stationary media panels and scrolling narrative panels works hand in hand
    to deliver an immersive experience. If the media panel consists of a web map or web scene, the map actions
    functionality allows authors to include options for further interactivity.
    Simply put, map actions are buttons that change something on the map or scene when toggled.
    These buttons can be configured to modify the map extent, the visibility of different layers etc., and this can be
    useful to include additional details without deviating from the primary narrative.

    There are two main types: Inline text map actions and map action blocks in sidecar.

    To create a map action you must use the `create_action` method found in the sidecar.

    ===============     ====================================================================
    **Parameter**        **Description**
    ---------------     --------------------------------------------------------------------
    node_id             Required String. The node id for the map tour type.
    ---------------     --------------------------------------------------------------------
    story               Required :class:`~arcgis.apps.storymap.story.StoryMap` that the map tour belongs to.
    ===============     ====================================================================

    """

    def __init__(self, **kwargs) -> None:
        node = kwargs.pop("node_id", None)
        story = kwargs.pop("story", None)
        if node:
            self.node = node
            self._story = story
            actions = story._properties["actions"]
            for action in actions:
                if action["origin"] == node:
                    self.target = action["target"]
                    self.properties = action
        else:
            self.node = "n-" + uuid.uuid4().hex[0:6]
            self._story = story

    # ----------------------------------------------------------------------
    def __str__(self) -> str:
        return "Map Action"

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        return "Map Action"

    # ----------------------------------------------------------------------
    @property
    def viewpoint(self) -> dict:
        for action in self._story._properties["actions"]:
            if action["origin"] == self.node:
                return action["data"]["viewpoint"]

    # ----------------------------------------------------------------------
    @property
    def text(self) -> str:
        """
        Get/Set the button text for a map action button.
        """
        node_dict = self._story._properties["nodes"][self.node]
        if "text" in node_dict["data"]:
            return node_dict["data"]["text"]
        return ""

    # ----------------------------------------------------------------------
    @text.setter
    def text(self, text: str) -> None:
        """"""
        if isinstance(text, str):
            self._story._properties["nodes"][self.node]["data"]["text"] = text
        else:
            raise TypeError("Text must be of type string.")

    # ----------------------------------------------------------------------
    def set_viewpoint(
        self, target_geometry: dict, scale: Scales, rotation: int | None = None
    ):
        """
        Set the extent and/or scale for the map action in the story.

        To see the current viewpoint call the `viewpoint` property on the Map Action
        node.

        ==================  ========================================
        **Parameter**        **Description**
        ------------------  ----------------------------------------
        target_geometry     Required dictionary representing the target geometry of the
                            viewpoint.

                            Example:
                                | {'spatialReference': {'latestWkid': 3857, 'wkid': 102100},
                                | 'x': -609354.6306080809,
                                | 'y': 2885721.2797636474}
        ------------------  ----------------------------------------
        scale               Required Scales enum class value or int.

                            Scale is a unitless way of describing how any distance on the map translates
                            to a real-world distance. For example, a map at a 1:24,000 scale communicates that 1 unit
                            on the screen represents 24,000 of the same unit in the real world.
                            So one inch on the screen represents 24,000 inches in the real world.
        ------------------  ----------------------------------------
        rotation            Optional float. Determine the rotation for an
                            action on a 3D map.
        ==================  ========================================

        :return: The current viewpoint dictionary
        """
        for idx, action in enumerate(self._story._properties["actions"]):
            if action["origin"] == self.node:
                if rotation is None:
                    if "viewpoint" in self._story._properties["actions"][idx]["data"]:
                        rotation = (
                            self._story._properties["actions"][idx]["data"][
                                "viewpoint"
                            ]["rotation"]
                            if "rotation"
                            in self._story._properties["actions"][idx]["data"][
                                "viewpoint"
                            ]
                            else 0
                        )
                if isinstance(scale, Scales):
                    scale = scale.value
                self._story._properties["actions"][idx]["data"]["viewpoint"] = {
                    "rotation": rotation,
                    "scale": scale,
                    "targetGeometry": target_geometry,
                }
        return self.viewpoint

    # ----------------------------------------------------------------------
    def delete(self):
        """
        Delete the map action.
        """
        for idx, action in enumerate(self._story._properties["actions"]):
            if action["origin"] == self.node:
                del self._story._properties["actions"][idx]
        return self._story._delete(self.node)

    # ----------------------------------------------------------------------
    def _check_node(self):
        # Node is not in the story if no story or node id is present
        if self._story is None:
            return False
        elif self.node is None:
            return False
        else:
            return True


###############################################################################################################

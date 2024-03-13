import arcgis


class Header(object):
    """
    Creates a dashboard header widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    title                       Optional string. Title of the header.
    -------------------------   -------------------------------------------
    subtitle                    Optional string. Subtitle of the header.
    -------------------------   -------------------------------------------
    margin                      Optional boolean. Set True to add margin to
                                header position.
    -------------------------   -------------------------------------------
    size                        Optional string. Define size of header from
                                small, medium, large.
                                Default is medium.
    -------------------------   -------------------------------------------
    logo_image_url              Optional url string. Define a logo image.
    -------------------------   -------------------------------------------
    logo_url                    Optional url string. Define a hyperlink for
                                the logo image.
    -------------------------   -------------------------------------------
    background_image_url        Optional url string. Add a background image
                                to the header.
    -------------------------   -------------------------------------------
    background_image_size       Optional string. Select size of the image.
                                Options:
                                    | fit-width
                                    | fit-height
                                    | fit-both
                                    | repeat
    -------------------------   -------------------------------------------
    background_image_position   Optional string. Define the image position
                                when using fit-width or fit-height.
                                Allowed options:
                                 | fit-height: left, center, right
                                 | fit-width: top, middle, bottom
    -------------------------   -------------------------------------------
    signout_link                Optional boolean. Parameter to save the best model
                                during training. If set to `True` the best model
                                based on validation loss will be saved during
                                training.
    -------------------------   -------------------------------------------
    menu_links                  Optional list of tuples. Each tuple contains
                                string label and string url.
    =========================   ===========================================
    """

    def __init__(
        self,
        title=None,
        subtitle=None,
        margin=True,
        size="medium",
        logo_image_url=None,
        logo_url=None,
        background_image_url=None,
        background_image_size="fit-width",
        background_image_position="left",
        signout_link=False,
        menu_links=None,
    ):
        # Header starts here.

        self._size = "medium"
        self._title = None
        self._subtitle = None
        self._margin = False
        self.type = "headerPanel"

        self.size = size
        self.title = title
        self.subtitle = subtitle
        self.margin = margin

        self._logo_image_url = logo_image_url
        self._logo_url = logo_url

        self._background_image = background_image_url
        self._background_image_placement = ("fit-height", "left")
        self.background_image_placement = (
            background_image_size,
            background_image_position,
        )

        self._sign_out_link = signout_link
        self._menu_links = []

        self.menu_links = menu_links

        self._text_color = None
        self._background_color = None
        self._selectors = []
        # Header ends here.

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard

        url = Dashboard._publish_random(self)
        return f"""<iframe src={url} width=300 height=300>"""

    @property
    def size(self):
        """
        Return size of the header. small, medium or large

        :return:
            String
        """
        return self._size

    @size.setter
    def size(self, value):
        """
        Set size of the header. Value can be small, medium or large.
        """
        self._size = value
        if value not in ["small", "medium", "large"]:
            self._size = "medium"

    @property
    def title(self):
        """
        :return: Returns title of the header
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set title of the header. A string.
        """
        if isinstance(value, str):
            self._title = value

    @property
    def subtitle(self):
        """
        :return: Subtitle of the header. A string.
        """
        return self._subtitle

    @subtitle.setter
    def subtitle(self, value):
        """
        Set subtitle of the header.
        """
        self._subtitle = value
        if not isinstance(value, str):
            self._subtitle = None

    @property
    def margin(self):
        """
        :return: Header margin, True or False
        """
        return self._margin

    @margin.setter
    def margin(self, value):
        """
        Set margin True or False.
        """
        self._margin = value
        if not isinstance(value, bool):
            self._margin = True

    @property
    def logo_image_url(self):
        """
        :return: Logo Image Url
        """
        return self._logo_image_url

    @logo_image_url.setter
    def logo_image_url(self, value):
        """
        :return: Set Logo image url.
        """
        self._logo_image_url = value

    @property
    def logo_url(self):
        """
        :return: Logo image hyperlink
        """
        return self._logo_url

    @logo_url.setter
    def logo_url(self, value):
        """
        Set Logo image hyperlink
        """
        self._logo_url = value

    @property
    def background_image_url(self):
        """
        :return: Background image url.
        """
        return self._background_image

    @background_image_url.setter
    def background_image_url(self, value):
        """
        Set background image url.
        """
        self._background_image = value

    @property
    def background_image_placement(self):
        """
        :return:
            Background image position.

            | If fit-height then left, right, center
            | If fit-width then top, bottom, middle
            | fit-both, repeat then None

        """
        return self._background_image_placement

    @background_image_placement.setter
    def background_image_placement(self, value):
        sizing = "fit-height"
        position = "left"

        size_position_mapping = {
            "fit-height": ["left", "center", "right"],
            "fit-width": ["top", "middle", "bottom"],
            "fit-both": [],
            "repeat": [],
        }

        temp_position = None
        if isinstance(value, tuple):
            if size_position_mapping.get(value[0]):
                sizing = value[0]
                temp_position = value[1]
        elif size_position_mapping.get(value):
            sizing = value

        if temp_position:
            if len(size_position_mapping.get(sizing)) == 0:
                position = None
            elif temp_position in size_position_mapping.get(sizing):
                position = temp_position
            else:
                position = size_position_mapping.get(sizing)[0]

        self._background_image_placement = (sizing, position)

    @property
    def sign_out_link(self):
        """
        :return: Sign out link enabled or disabled.
        """
        return self.sign_out_link

    @sign_out_link.setter
    def sign_out_link(self, value):
        """
        Enable or disable the signout link.
        """
        self._sign_out_link = value
        if not isinstance(value, bool):
            self._sign_out_link = True

    @property
    def menu_links(self):
        """
        :return: List of menu links.
        """
        return self._menu_links

    @menu_links.setter
    def menu_links(self, value):
        """
        Set list of tuples of menu links.
        For example: [("label", "url")]
        """
        if isinstance(value, list):
            for val in value:
                if isinstance(val, tuple) and len(val) == 2:
                    self._menu_links.append(val)
        elif isinstance(value, tuple) and len(value) == 2:
            self._menu_links.append(value)

    @property
    def text_color(self):
        """
        :return: None for default text color else HEX code.
        """
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        """
        Set HEX code for text color.
        """
        self._text_color = value
        if not isinstance(value, str):
            self._text_color = None

    @property
    def background_color(self):
        """
        :return: Background color HEX code.
        """
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        """
        Set background color HEX code.
        """
        self._background_color = value
        if not isinstance(value, str):
            self._background_color = None

    def add_selector(self, selector):
        """
        Add Number Selector, Category Selector or Date Picker widget.
        """
        self._selectors.append(selector)

    def _convert_to_json(self):
        header_panel = {
            "type": "headerPanel",
            "size": self.size,
            "showSignOutMenu": self._sign_out_link,
            "menuLinks": self._menu_links,
            "showMargin": self.margin,
            "selectors": [selector._convert_to_json() for selector in self._selectors],
        }

        if self.title:
            header_panel["title"] = self.title

        if self.subtitle:
            header_panel["subtitle"] = self.subtitle

        if self._background_image:
            header_panel["backgroundImageUrl"] = self._background_image
            header_panel["backgroundImageSizing"] = self._background_image_placement[0]
            header_panel[
                "normalBackgroundImagePlacement"
            ] = self._background_image_placement[1]
            header_panel[
                "horizontalBackgroundImagePlacement"
            ] = self._background_image_placement[1]

        if self._logo_image_url:
            header_panel["logoImageURL"] = self._logo_image_url
            header_panel["logoURL"] = self._logo_url

        if self._text_color:
            header_panel["titleTextColor"] = self._text_color

        if self._background_color:
            header_panel["backgroundColor"] = self._background_color

        return header_panel

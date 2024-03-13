class LocationTrackingError(Exception):
    """
    Location Tracking Error Class
    """

    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

from dataclasses import dataclass, asdict, field
from typing import Dict, Union, Optional, Any, ClassVar
import pytz


@dataclass(frozen=True)
class RunInterval:
    """
    Set the run interval for the feed.

    ===============     ================================================================================================
    **Parameter**        **Description**
    ---------------     ------------------------------------------------------------------------------------------------
    cron_expression     String. Cron expression that specifies the run interval. You can use the cron generator at the
                        following link to generate a cron expression: `Cron Expression Generator & Explainer
                        <https://www.freeformatter.com/cron-expression-generator-quartz.html>`_.

                        The default is every one minute, represented by the following expression:

                            "``0 * * ? * * *``"
    ---------------     ------------------------------------------------------------------------------------------------
    timezone            String. Run interval timezone to use. The default is: "America/Los_Angeles"

                        .. note::
                            To learn more about time zones, see
                            `List of tz database time zones <https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>`_ page on Wikipedia.
    ===============     ================================================================================================

    :return: `True` if the operation is a success

    .. code-block:: python

        # Usage Example

        feed.run_interval = RunInterval(
            cron_expression="0 * * ? * * *", timezone="America/Los_Angeles",
        )

        # Seconds value must be between 10 and 59
        # Minutes value must be between 1 and 59
        # Hours value must be between 1 and 23

    """

    cron_expression: str
    timezone: str = field(default="America/Los_Angeles")

    def __post_init__(self):
        if self.cron_expression in (None, ""):
            raise ValueError("Cron expression cannot be empty or None")
        elif self.timezone not in pytz.all_timezones:
            raise ValueError("Invalid time zone string")

    def _build(self):
        return {
            "recurrence": {
                "expression": self.cron_expression,
                "timeZone": self.timezone,
            }
        }

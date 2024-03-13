from typing import Dict
from abc import ABCMeta, abstractmethod


class Task(metaclass=ABCMeta):
    @abstractmethod
    def start(self) -> Dict:
        """Starts a task instance"""
        raise NotImplemented()

    @abstractmethod
    def stop(self) -> Dict:
        """Stops a task instance"""
        raise NotImplemented()

    @abstractmethod
    def status(self) -> Dict:
        """Fetch current status of a task instance"""
        raise NotImplemented()

    @abstractmethod
    def metrics(self) -> Dict:
        """Fetch current status of a task instance"""
        raise NotImplemented()

    @abstractmethod
    def delete(self) -> bool:
        """Delete a task instance"""
        raise NotImplemented()

from asyncio import Task
import datetime
from typing import Dict


class User:
    def __init__(self, user_id: int):
        self.id = user_id
        # Map of task objects with their assigned timestamp
        self.map_task: Dict[int, Task] = {} # type: ignore

    def add_task(self, time: int, task: Task):
        self.map_task[time] = task

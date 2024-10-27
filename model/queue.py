from typing import Optional
from model.task import Task


class Queue:
    def __init__(self):
        self.tasks = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def get_next_task(self) -> Optional[Task]:
        return self.tasks.pop(0) if self.tasks else None # FIFO QUEUE

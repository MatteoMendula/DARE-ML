from threading import Lock
from typing import Optional
from model.task import Task

class Queue:
    def __init__(self):
        self.tasks = []
        self.lock = Lock()  # Initialize a lock for the task list

    def add_task(self, task: Task):
        # Lock this section to avoid concurrent modification
        with self.lock:
            self.tasks.append(task)
            # print(f"Task {task.id} added to the queue.")  # Optional logging for debugging

    def get_next_task(self) -> Optional[Task]:
        # Lock this section to safely access and modify the list
        with self.lock:
            if self.tasks:
                task = self.tasks.pop(0)
                # print(f"Task {task.id} removed from the queue.")  # Optional logging for debugging
                return task
            return None  # Return None if queue is empty

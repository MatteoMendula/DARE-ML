
from model.queue import Queue


class Policy:
    def __init__(self, policy_type: str, task_queue: Queue):
        self.policy_type = policy_type
        self.task_queue = task_queue

    def get_next_task(self):
        """Return the next task based on the selected policy."""
        if self.policy_type == "fifo":
            return self.get_fifo_task()
        elif self.policy_type == "shortest_job":
            return self.get_shortest_job_task()
        else:
            raise ValueError("Unknown policy type")

    def get_fifo_task(self):
        """FIFO policy: get the first task in the queue."""
        return self.task_queue.get_next_task()

    def get_shortest_job_task(self):
        """Shortest Job policy: get the task with the shortest training time."""
        # Find the task with the shortest training time without removing it
        if not self.task_queue.tasks:
            return None
        shortest_task = min(self.task_queue.tasks, key=lambda task: task.training_time)
        self.task_queue.tasks.remove(shortest_task)  # Remove this task from the queue
        return shortest_task


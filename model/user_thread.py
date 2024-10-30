import time
from colorama import Fore, Style, init
import threading
from model.user import User

# Initialize colorama
init(autoreset=True)

class UserThread(threading.Thread):
    """Thread for each user that waits and adds tasks at specified times."""
    def __init__(self, user, task_queue, task_list):
        super().__init__()
        self.user: User = user
        self.task_queue = task_queue
        self.task_list = task_list

    def run(self):
        while self.user.map_task:
            # Get the earliest task request time
            time_of_asking = min(self.user.map_task.keys())
            task = self.user.map_task.pop(time_of_asking)  # Retrieve and remove the task from map_task

            # Wait until the specified time to request the task
            time.sleep(time_of_asking)
            print(Fore.CYAN + f"User {self.user.id} is requesting {task.id} at time {time_of_asking}\n")
            task.arrival_time = time.time()
            
            # Add the task to the queue
            self.task_queue.add_task(task)


import threading
import time
from colorama import Fore, Style, init
from model.scheduler import Scheduler
from model.task import Task

# Initialize colorama
init(autoreset=True)

class TaskThread(threading.Thread):
    def __init__(self, task: Task, scheduler: Scheduler):
        super().__init__()
        self.task = task
        self.scheduler = scheduler
    
    def run(self):
        # Wait for the duration of the task's training time
        time.sleep(self.task.training_time)  # Simulate training time
        
        # Release the GPU after the task is done
        self.scheduler.release_gpu(self.task.assigned_gpu.id)
        print(Fore.MAGENTA + f"Task {self.task.id} completed and GPU {self.task.assigned_gpu.id} released.\n")

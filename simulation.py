import argparse
import random
import time
from threading import Thread
from colorama import Fore, init
from model.gpu import GPU
from model.policy import Policy
from model.scheduler import Scheduler
from model.task_thread import TaskThread
from model.user import User
from model.queue import Queue
from model.task import Task
from model.user_thread import UserThread

from plots.gantt_executions import generate_gantt_gantt_executions
from plots.gantt_arrival_ending_time import generate_gantt_arrival_ending_time

# Initialize colorama
init(autoreset=True)

MAX_NUM_TASKS = 10
MAX_NUM_TASK_ID = 1000
MIN_TIME = 1
MAX_TIME = 3

def main(args):
    random.seed(args.seed)
    
    # Create GPUs
    gpus = [GPU(gpu_id=i+1, memory_size=memory) for i, memory in enumerate(args.gpus)]
    scheduler = Scheduler(gpus=gpus)
    task_queue = Queue()

    # Define model properties
    model_properties = {
        "google/flan-t5-base": {"training_time": 10, "memory_required": 24},
        "google/flan-t5-small": {"training_time": 5, "memory_required": 11},
        "lucadiliello/bart-small": {"training_time": 1, "memory_required": 11}
    }

    # Create users and their tasks
    users = []
    user_threads = []
    all_tasks = []
    task_arrival_times = {}  # Dictionary to store the arrival times of each task
    task_records = []  # List to store task allocations for Gantt charts

    for user_id in range(1, args.number_of_users + 1):
        user = User(user_id=user_id)
        num_tasks = random.randint(1, MAX_NUM_TASKS)
        user_tasks = []
        for t in range(num_tasks):
            task_id = f"task_{t}_of_user_{user_id}"
            model_name = random.choice(list(model_properties.keys()))
            training_time = model_properties[model_name]["training_time"]
            memory_required = model_properties[model_name]["memory_required"]
            task = Task(
                task_id=task_id,
                model_name=model_name,
                training_time=training_time,
                memory_required=memory_required,
                user_id=user_id
            )
            # Assign a time for when this task will be requested by the user
            time_of_asking_the_task = random.randint(MIN_TIME, MAX_TIME)
            user.add_task(time_of_asking_the_task, task)
            user_tasks.append((time_of_asking_the_task, task))
            all_tasks.append((time_of_asking_the_task, task))

        # Create a thread for each user with their tasks
        user_thread = UserThread(user=user, task_queue=task_queue, task_list=user_tasks)
        user_threads.append(user_thread)

    # Initialize Policy
    policy = Policy(policy_type=args.policy, task_queue=task_queue)

    # Start user threads to request tasks
    for u in user_threads:
        u.start()

    # Process tasks as they arrive in the task queue
    threads = []
    while any(thread.is_alive() for thread in user_threads) or task_queue.tasks:
        # Get the next task based on the scheduling policy
        current_task = policy.get_next_task()
        if current_task:
            if scheduler.assign_task_to_gpu(current_task):
                # Track task start and end time for Gantt chart
                start_time = time.time()
                end_time = start_time + current_task.training_time

                # Add task details to records for Gantt chart
                task_records.append((current_task.assigned_gpu.id, current_task.arrival_time, start_time, end_time, current_task.training_time, current_task.user_id))

                # Start a thread for the task
                thread = TaskThread(current_task, scheduler)
                thread.start()
                threads.append(thread)
            else:
                # Re-add the task to the queue if no GPU is available
                task_queue.add_task(current_task)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Wait for all user threads to complete
    for user_thread in user_threads:
        user_thread.join()

    print(task_records)
    # Generate Gantt charts
    generate_gantt_gantt_executions(task_records, gpus=gpus)
    generate_gantt_arrival_ending_time(task_records=task_records, gpus=gpus)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a GPU Scheduler Simulation for ML Tasks")
    parser.add_argument(
        '--gpus',
        nargs='+',
        type=int,
        default=[24, 24, 24, 11, 11, 11, 11],
        help="List of GPU memory sizes in GB (default: 3 GPUs of 24GB and 4 GPUs of 11GB)"
    )
    parser.add_argument('--number_of_users', type=int, required=True, help="Number of users to create")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for generating tasks")
    parser.add_argument('--policy', type=str, default='fifo', help="Task scheduling policy (fifo or shortest_job)")

    args = parser.parse_args()
    main(args)

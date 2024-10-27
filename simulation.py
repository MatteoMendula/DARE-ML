import argparse
import random
import matplotlib.pyplot as plt
from colorama import Fore, init
from model.gpu import GPU
from model.policy import Policy
from model.scheduler import Scheduler
from model.task_thread import TaskThread
from model.user import User
from model.queue import Queue
from model.task import Task
import time

from plots.gantt_executions import generate_gantt_chart

# Initialize colorama
init(autoreset=True)

MAX_NUM_TASKS = 10
MAX_NUM_TASK_ID = 1000
MAX_TIME = 10000000
MIN_TIME = 10

def main(args):
    random.seed(args.seed)
    
    # Create GPUs
    gpus = [GPU(gpu_id=i+1, memory_size=memory) for i, memory in enumerate(args.gpus)]
    scheduler = Scheduler(gpus=gpus)
    task_queue = Queue()

    # Define model properties
    model_properties = {
        "google/flan-t5-base": {"training_time": 0.06 * 60, "memory_required": 24},
        "google/flan-t5-small": {"training_time": 0.006 * 60, "memory_required": 11},
        "lucadiliello/bart-small": {"training_time": 0.0006 * 60, "memory_required": 11}
    }

    # Create users and their tasks
    users = []
    all_tasks = []
    for user_id in range(1, args.number_of_users + 1):
        user = User(user_id=user_id)
        num_tasks = random.randint(1, MAX_NUM_TASKS)
        for t in range(num_tasks):
            task_id = f"task_{t}_of_user_{user_id}"
            model_name = random.choice(list(model_properties.keys()))
            training_time = model_properties[model_name]["training_time"]
            memory_required = model_properties[model_name]["memory_required"]
            task = Task(
                task_id=task_id,
                model_name=model_name,
                training_time=training_time,
                memory_required=memory_required
            )
            # Assign a time for when this task will be requested by the user
            time_of_asking_the_task = random.randint(MIN_TIME, MAX_TIME)
            user.add_task(time_of_asking_the_task, task)
            all_tasks.append((time_of_asking_the_task, task))
        users.append(user)

    # Sort all tasks by `time_of_asking_the_task`
    all_tasks.sort(key=lambda x: x[0])

    # Initialize Policy
    policy = Policy(policy_type=args.policy, task_queue=task_queue)

    # Simulation clock and task processing
    threads = []
    task_index = 0
    task_records = []  # List to store task allocations for Gantt chart

    while task_index < len(all_tasks) or task_queue.tasks:
        # Add tasks to the queue that are due at the current time
        while task_index < len(all_tasks):
            _, task = all_tasks[task_index]
            task_queue.add_task(task)
            task_index += 1

        # Get the next task based on the scheduling policy
        current_task = policy.get_next_task()
        if current_task:
            if scheduler.assign_task_to_gpu(current_task):
                # Track task start and end time for Gantt chart
                start_time = time.time()
                end_time = start_time + current_task.training_time
                task_records.append((current_task.assigned_gpu.id, current_task.id, start_time, end_time))

                
                # Start a thread for the task
                thread = TaskThread(current_task, scheduler)
                thread.start()
                threads.append(thread)
            else:
                # Re-add the task to the queue if no GPU is available
                task_queue.add_task(current_task)
        # else:
        #     print(Fore.YELLOW + "No tasks ready for processing at the moment.")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Generate Gantt chart
    generate_gantt_chart(task_records, gpus)



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

import argparse
import random
import time
from colorama import init
from model.gpu import GPU
from model.policy import Policy
from model.scheduler import Scheduler
from model.task_thread import TaskThread
from model.user import User
from model.queue import Queue
from model.task import Task
from model.user_thread import UserThread
import csv
import json  # Import json to handle saving dictionaries

# Initialize colorama
init(autoreset=True)

def main(args):
    max_num_tasks = args.tasks
    min_time = args.min_time  # Set from command-line arguments
    max_time = args.max_time  # Set from command-line arguments

    # Dictionary to store random numbers for reproducibility
    random_numbers = {}

    # Create GPUs
    gpus = [GPU(gpu_id=i + 1, memory_size=memory) for i, memory in enumerate(args.gpus)]
    scheduler = Scheduler(gpus=gpus)
    task_queue = Queue()

    # Scaling factor to adjust training times down to smallest unit possible
    smallest_time = 0.02 # Smallest desired training time (e.g., set to 1 unit for fastest simulation)
    original_smallest_time = 1154.37700009346  # Smallest original time in each policy
    scaling_factor = smallest_time / original_smallest_time

    # Define model properties
    if args.policy_dare:
        model_properties = {
            "lucadiliello/bart-small": {"training_time": 2228.8450000286102 * scaling_factor, "memory_required": 11},
            "google/flan-t5-base": {"training_time": 1584.233999967575 * scaling_factor, "memory_required": 24},
            "google/flan-t5-small": {"training_time": 1154.37700009346 * scaling_factor, "memory_required": 11},
        }
    else:
        model_properties = {
            "lucadiliello/bart-small": {"training_time": 47806.575000047684 * scaling_factor, "memory_required": 11},
            "google/flan-t5-base": {"training_time": 141987.64184201873 * scaling_factor, "memory_required": 24},
            "google/flan-t5-small": {"training_time": 61417.08800005913 * scaling_factor, "memory_required": 11},
        }

    SESSION_DURATION = 14 * 60 * 60 * scaling_factor

    # Create users and their tasks
    user_threads = []
    task_records = []  # List to store task allocations for Gantt charts

    # Use built-in random generator
    random.seed(10)

    for user_id in range(1, args.users + 1):
        user = User(user_id=user_id)
        num_tasks = random.randint(1, max_num_tasks)  # Get random number of tasks
        random_numbers[f'user_{user_id}_num_tasks'] = num_tasks  # Store the number of tasks

        for t in range(num_tasks):
            task_id = f"task_{t}_of_user_{user_id}"
            model_name = random.choice(list(model_properties.keys()))
            # Store model name
            random_numbers[f'task_{task_id}_model_name'] = model_name
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
            time_of_asking_the_task = random.uniform(min_time, max_time)
            random_numbers[f'task_{task_id}_request_time'] = time_of_asking_the_task  # Store the task request time
            user.add_task(time_of_asking_the_task, task)

        # Create a thread for each user with their tasks
        user_thread = UserThread(user=user, task_queue=task_queue)
        user_threads.append(user_thread)

    # Initialize Policy
    policy = Policy(policy_type=args.scheduling_type, task_queue=task_queue)

    # Start user threads to request tasks
    for u in user_threads:
        u.start()

    # Process tasks as they arrive in the task queue
    threads = []
    while any(thread.is_alive() for thread in user_threads) or len(task_queue.tasks)>0:
        # Get the next task based on the scheduling policy
        current_task = policy.get_next_task()
        if current_task:
            if scheduler.assign_task_to_gpu(current_task):
                # Track task start and end time for analysis
                start_time = time.time()
                end_time = start_time + current_task.training_time

                # Add task details to records for analysis
                task_records.append({
                    "GPU_ID": current_task.assigned_gpu.id,
                    "Arrival_Time": current_task.arrival_time,
                    "Start_Time": start_time,
                    "End_Time": end_time,
                    "Training_Time": current_task.training_time,
                    "User_ID": current_task.user_id,
                    "Model_Name": current_task.model_name,
                })

                # Start a thread for the task
                thread = TaskThread(current_task, scheduler, SESSION_DURATION if args.session else 0)
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

    # Generate a filename based on input parameters
    filename = f"results/task_records_users_{args.users}_tasks_{args.tasks}_seed_{10}_scheduling_{args.scheduling_type}_range_{args.min_time}_{args.max_time}_dare_{str(args.policy_dare)}.csv"
    
    # Save task_records as a CSV file
    with open(filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["GPU_ID", "Arrival_Time", "Start_Time", "End_Time", "Training_Time", "User_ID", "Model_Name"])
        writer.writeheader()  # Write header
        writer.writerows(task_records)  # Write each record

    # Save random numbers to a file as a dictionary
    random_filename = f"results/random_numbers_users_{args.users}_tasks_{args.tasks}_seed_{10}.json"
    with open(random_filename, "w") as random_file:
        json.dump(random_numbers, random_file, indent=4)  # Save the dictionary as a JSON file


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
    parser.add_argument('--users', type=int, required=True, default=3, help="Number of users to create")
    parser.add_argument('--tasks', type=int, required=True, default=10, help="Max number of tasks each user wants to submit")
    parser.add_argument('--scheduling-type', type=str, default='fifo', help="Task scheduling policy (fifo or shortest_job)")
    parser.add_argument('--min-time', type=float, required=True, default=1, help="Minimum task request time interval")
    parser.add_argument('--max-time', type=float, required=True, default=2, help="Maximum task request time interval")
    parser.add_argument('--policy-dare', type=bool, default=False, help="Use Dare policy or not")
    parser.add_argument('--session', type=bool, default=False, help="Use Dare policy or not")

    args = parser.parse_args()
    main(args)

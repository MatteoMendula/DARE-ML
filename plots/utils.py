import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  # Import path effects for text outline
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

smallest_time = 0.2 # Smallest desired training time (e.g., set to 1 unit for fastest simulation)
original_smallest_time = 1154.37700009346  # Smallest original time in each policy
scaling_factor = smallest_time / original_smallest_time

def load_task_records(file_path):
    """Load task records from a CSV file."""
    try:
        task_records = pd.read_csv(file_path)
        return task_records
    except Exception as e:
        print(f"Error loading task records: {e}")
        return None
    
def generate_gantt_arrival_ending_time(task_records):

    # check if task_records["Task_Retrain"] contains -1 values
    is_retrain = -1 not in list(task_records["Task_Retrain"].unique())

    # Extract unique GPU IDs from the task records DataFrame
    gpus = task_records['GPU_ID'].unique()
    print(type(gpus))  # Should be a numpy array or similar

    wait_color = '#2C7865'  
    burst_color = '#FF9800'   

    min_start_time = task_records["Start_Time"].min()
    min_arrival_time = task_records["Arrival_Time"].min()

    fig, axs = plt.subplots(4, 2, figsize=(15, 10))  # Adjust figsize if needed

    # Iterate over GPUs to create individual plots
    for g in gpus:
        gpu_index = g - 1
        r = gpu_index // 2
        c = gpu_index % 2

        # Filter tasks for the current GPU
        gpu_tasks = task_records[task_records['GPU_ID'] == g]
        tot_training_time = gpu_tasks['Training_Time'].sum()
        print(f"GPU {g} Tasks: {len(gpu_tasks)}, Total Training Time: {tot_training_time}, Unique Users: {gpu_tasks['User_ID'].nunique()}")

        for i, row in gpu_tasks.iterrows():
            arrival = row['Arrival_Time'] - min_arrival_time
            start = row['Start_Time'] - min_start_time
            training = row['Training_Time']
            user_id = row['User_ID']

            wait_time = start - arrival  # Calculate wait time

            # Plot wait time segment
            if wait_time > 0:  # Ensure wait time is positive
                axs[r, c].barh(y=i, width=wait_time, left=arrival, color=wait_color, edgecolor='black', align="center")

            # Plot burst time segment (training time)
            axs[r, c].barh(y=i, width=training, left=start, color=burst_color, edgecolor='black', align="center")

            # Center the user ID label on the burst time bar
            mid_point = start + training / 2
            if not is_retrain:
                text = axs[r, c].text(mid_point, i, f"User {user_id}", va='center', ha='center', color='black', fontsize=8)

                # Apply white outline around the text
                text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

        
        # Configure chart
        axs[r, c].set_yticks([])  # Optionally set Y ticks based on tasks if needed
        axs[r, c].set_xlabel("Time[h]")
        axs[r, c].set_title(f"GPU {g}")

    # Add legend outside the loop, only once at the bottom right
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=wait_color, edgecolor='black'),
        plt.Rectangle((0, 0), 1, 1, facecolor=burst_color, edgecolor='black')
    ]

    if not is_retrain:
        fig.legend(handles, ["Wait Time", "Burst Time", "User ID"], title="Task Segments", loc='lower right', 
           bbox_to_anchor=(0.75, 0.1), fontsize=12, title_fontsize=12)

    # Hide any extra subplot, e.g., the bottom right plot in a 4x2 grid when there are only 7 GPUs
    if len(gpus) < 8:
        axs[3, 1].axis('off')  # Turn off the bottom-right subplot
    
    plt.tight_layout()  # Adjust layout to prevent clipping of elements

    fig.suptitle("Gantt Chart of Task Wait and Burst Time", fontsize=16, x=0.5, y=1.05)
    plt.show()  # Display all subplots at once

# Assuming you have a DataFrame `task_records` already defined
# generate_gantt_arrival_ending_time(task_records)

def get_gpus(task_records):
    """Get unique GPU IDs from task records."""
    gpu_ids = []  # Initialize an empty list to store unique GPU IDs

    # Iterate through each record in the task_records DataFrame
    for _, record in task_records.iterrows():
        gpu_id = record['GPU_ID']

        if gpu_id not in gpu_ids:
            gpu_ids.append(gpu_id)

    return gpu_ids

def generate_gantt_gantt_executions(task_records):
    gpus = get_gpus(task_records)
    fig, ax = plt.subplots(figsize=(20, 6))

    model_names = [
        "google/flan-t5-base",
        "google/flan-t5-small",
        "lucadiliello/bart-small",
    ]

    colors = [
        "#A04747",  # Red for longest duration
        "#D8A25E",  # Orange for below medium
        "#EEDF7A",  # Yellow for shortest duration
    ]

    hatchs = [
        ".",
        "x",
        "o"
    ]

    def get_color(model_name):
        if model_name in model_names:
            if model_name == model_names[0]:
                return (colors[0], hatchs[0])
            elif model_name == model_names[1]:
                return (colors[1], hatchs[1])
            elif model_name == model_names[2]:
                return (colors[2], hatchs[2])
            
    min_start_time = task_records['Start_Time'].min()

    # Plot each task on the Gantt chart
    for _, row in task_records.iterrows():
        gpu_id = row['GPU_ID']
        start = row['Start_Time'] - min_start_time
        duration = row['Training_Time']
        name = row['Model_Name']
        color, hatch = get_color(name)

        ax.barh(gpu_id, duration, left=start, color=color, edgecolor='black', hatch=hatch)

    # Configure chart
    ax.set_yticks(gpus)
    ax.set_yticklabels([f"GPU {gpu}" for gpu in gpus])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart of Task Allocation on GPUs")

    # Create custom legend with hatches
    legend_handles = []
    for color, hatch, label in zip(colors, hatchs, model_names):
        legend_handles.append(Patch(facecolor=color, edgecolor='black', label=label, hatch=hatch))

    # Create the legend
    ax.legend(handles=legend_handles, title="Job types", loc='upper right', frameon=True)

    plt.show()

# Example of how to call the function with dummy data
# Uncomment the line below when you have your DataFrame ready
# generate_gantt_gantt_executions(task_records)

def calculate_total_waiting_time(task_records, user_id = None):
    """Calculate the total waiting time across all tasks in a DataFrame."""

    df = task_records

    if user_id != None:
        df = task_records[task_records["User_ID"] == user_id]

    tot_times = []
    checked_ids = []

    for i, row in df.iterrows():
        if row["Task_Retrain"] == -1:
            tot_times += [row["Start_Time"] - row["Arrival_Time"]]
        else:

            if row["Task_Id"] in checked_ids:
                continue

            last_retraing = df[df["Task_Id"] == row["Task_Id"]]
            last_retraing = last_retraing.sort_values(by=['Task_Retrain'])


            last_start_time = last_retraing["Start_Time"].tail(1)
            last_arrival_time = last_retraing["Arrival_Time"].tail(1)

            tot_times += [last_start_time - last_arrival_time]
            checked_ids += [row["Task_Id"]]


    mean_tot_times = np.array(tot_times).mean()
    std_tot_times = np.array(tot_times).std()
    sum_tot_times = np.array(tot_times).sum()

    return (mean_tot_times, std_tot_times, sum_tot_times)


    # # Calculate waiting times
    # task_records['Waiting_Time'] = task_records['Start_Time'] - task_records['Arrival_Time']
    
    # # # Sum the waiting times
    # total_waiting_time = task_records['Waiting_Time'].sum()
    # return total_waiting_time

def plot_waiting_times(task_records):
    fig, ax = plt.subplots(figsize=(10, 6))

    user_ids = list(task_records["User_ID"].unique())
    bar_width = 0.8
    
    # Scale x positions to increase space between clusters
    x = np.arange(len(user_ids)) * 2.5

    for idx, u_id in enumerate(user_ids):
        user_metrics = calculate_total_waiting_time(task_records=task_records, user_id=u_id)

        # Bar plot for average waiting times
        ax.bar(x[idx] - bar_width/2, user_metrics[0], color='#00224D', alpha=1, label='Average Waiting Time' if idx == 0 else "")

        # Bar plot for total waiting times on top
        ax.bar(x[idx] + bar_width/2, user_metrics[2], color='#A0153E', alpha=1, label='Total Waiting Time' if idx == 0 else "")

    ax.set_xlabel('User ID')
    ax.set_ylabel('Waiting Time (seconds)')
    ax.set_title('Waiting Times per User')
    ax.set_xticks(x)
    ax.set_xticklabels(user_ids)
    ax.legend()
    ax.grid(axis='y')
    fig.tight_layout()
    plt.show()

def calc_tot_energy_from_df(df, profiling = False):
    BASE_MEAN_mW = 378.8556695697469

    SMALL_MEAN_mW = 93.54626298542978

    BART_MEAN_mW = 93.95008267753535

    # check if Task_Retrain column contains -1 values
    is_retrain = -1 not in list(df["Task_Retrain"].unique())

    df_base = df[df["Model_Name"] == "google/flan-t5-base"]
    df_small = df[df["Model_Name"] == "google/flan-t5-small"]
    df_bart = df[df["Model_Name"] == "lucadiliello/bart-small"]

    tot_base_training_time = df_base["Training_Time"].sum()
    tot_small_training_time = df_small["Training_Time"].sum()
    tot_bart_training_time = df_bart["Training_Time"].sum()

    tot_base_energy = (BASE_MEAN_mW * tot_base_training_time) / 3600
    tot_small_energy = (SMALL_MEAN_mW * tot_small_training_time) / 3600
    tot_bart_energy = (BART_MEAN_mW * tot_bart_training_time) / 3600

    if is_retrain and profiling:
        tot_base_energy = tot_base_energy / 92.70503491796818
        tot_small_energy = tot_small_energy / 21.795236303013805
        tot_bart_energy = tot_bart_energy / 53.371237007832384

    return (tot_base_energy, tot_small_energy, tot_bart_energy)

def parse_seconds_to_hours(seconds):
    hours = seconds / 3600
    return hours

def gpus_usage(df, profiling = False):

    is_retrain = -1 not in list(df["Task_Retrain"].unique())

    gpus_usage = []
    for gpu_id in range(1, 8):
        gpu_df = df[df["GPU_ID"] == gpu_id]
        gpu_usage_base = gpu_df[df["Model_Name"] == "google/flan-t5-base"]["Training_Time"].sum()
        gpu_usage_small = gpu_df[df["Model_Name"] == "google/flan-t5-small"]["Training_Time"].sum()
        gpu_usage_bart = gpu_df[df["Model_Name"] == "lucadiliello/bart-small"]["Training_Time"].sum()

        if is_retrain and profiling:
            gpu_usage_base = gpu_usage_base / 92.70503491796818
            gpu_usage_small = gpu_usage_small / 21.795236303013805
            gpu_usage_bart = gpu_usage_bart / 53.371237007832384

        gpus_usage.append(gpu_usage_base + gpu_usage_small + gpu_usage_bart)

    
    return gpus_usage
        
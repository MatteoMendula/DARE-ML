import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  # Import path effects for text outline
from plots import analysis


def generate_gantt_arrival_ending_time(task_records):
    gpus=analysis.get_gpus

    wait_color = 'green'  
    burst_color = 'red'   

    # Iterate over GPUs to create individual plots
    for g in gpus:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Flags to track if wait and burst labels are added to legend
        wait_label_added = False
        burst_label_added = False

        # Filter tasks for the current GPU
        gpu_tasks = [(gpu, arrival, start, end, training, user_id) for gpu, arrival, start, end, training, user_id in task_records if gpu == g.id]
        
        for i, (gpu, arrival, start, end, training, user_id) in enumerate(gpu_tasks):
            wait_time = start - arrival  # Calculate wait time

            # Plot wait time segment
            if wait_time > 0:  # Ensure wait time is positive
                ax.barh(y=i, width=wait_time, left=arrival, color=wait_color, edgecolor='black', align="center")

            # Plot burst time segment (training time)
            ax.barh(y=i, width=training, left=start, color=burst_color, edgecolor='black',align="center")

            # Center the user ID label on the burst time bar
            mid_point = start + training / 2
            text = ax.text(mid_point, start, f"User {user_id}", va='center', ha='center', color='black', fontsize=8)

            # Apply white outline around the text
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])
        
        # Configure chart
        ax.set_yticks([])  # Plot tasks based on their start time
        ax.set_yticklabels([])  # Y-axis label as task start
        ax.set_xlabel("Time")
        ax.set_title(f"Gantt Chart of Task Wait and Burst Times for GPU {g.id}")

        # Add legend for wait and burst time
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=wait_color, edgecolor='black'),
            plt.Rectangle((0, 0), 1, 1, facecolor=burst_color, edgecolor='black')
        ]
        ax.legend(handles, ["Wait Time", "Burst Time"], title="Task Segments", loc='upper right')


        # plt.tight_layout() # gives error
        plt.show()

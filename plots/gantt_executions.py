import matplotlib.pyplot as plt
import numpy as np

def generate_gantt_chart(task_records, gpus):
    print(task_records)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate durations
    durations = [end - start for _, _, start, end in task_records]
    max_duration = max(durations)
    min_duration = min(durations)

    # Define thresholds for colors
    medium_duration = (max_duration + min_duration) / 2

    # Define a color assignment function
    def get_color(duration):
        if duration == max_duration:
            return (1.0, 0, 0)  # Red for longest duration
        elif duration == min_duration:
            return (1.0, 1.0, 0)  # Yellow for shortest duration
        elif duration > min_duration and duration < medium_duration:
            return (1.0, 0.5, 0)  # Orange for medium duration (above min and below medium)
        else:
            return (1.0, 0.75, 0.25)  # Light orange for medium duration (above medium)

    # Plot each task on the Gantt chart
    for gpu_id, task_id, start, end in task_records:
        duration = end - start
        color = get_color(duration)
        ax.barh(gpu_id, duration, left=start, color=color,
                edgecolor='black', label=task_id)

    # Configure the chart
    ax.set_yticks([gpu.id for gpu in gpus])
    ax.set_yticklabels([f"GPU {gpu.id}" for gpu in gpus])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart of Task Allocation on GPUs")

    # Create a custom legend for color meanings
    legend_labels = ['Short Duration (Yellow)', 'Medium Duration (Orange)', 'Long Duration (Red)']
    legend_colors = [(1.0, 1.0, 0), (1.0, 0.5, 0), (1.0, 0, 0)]  # Colors for each label
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(handles, legend_labels, title="Color Meaning", loc='upper right')

    plt.show()

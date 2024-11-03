import matplotlib.pyplot as plt
from plots import analysis

def generate_gantt_gantt_executions(task_records):
    gpus=analysis.get_gpus

    fig, ax = plt.subplots(figsize=(20, 6))

    # Determine min and max durations for color scaling
    durations = [end - start for gpu, arrival, start, end, training, user_id in task_records]
    max_duration = max(durations)
    min_duration = min(durations)
    medium_duration = (max_duration + min_duration) / 2

    # Define color based on task duration
    def get_color(duration):
        if duration == max_duration:
            return (1.0, 0, 0)  # Red for longest duration
        elif duration == min_duration:
            return (1.0, 1.0, 0)  # Yellow for shortest duration
        elif duration > min_duration and duration < medium_duration:
            return (1.0, 0.5, 0)  # Orange for below medium
        else:
            return (1.0, 0.75, 0.25)  # Light orange for above medium

    # Plot each task on the Gantt chart
    for gpu_id, arrival, start, end, training, user_id in task_records:
        duration = end - start
        color = get_color(duration)
        ax.barh(gpu_id, duration, left=start, color=color, edgecolor='black')

    # Configure chart
    ax.set_yticks([gpu.id for gpu in gpus])
    ax.set_yticklabels([f"GPU {gpu.id}" for gpu in gpus])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart of Task Allocation on GPUs")

    # Create custom legend for duration color coding
    legend_labels = ['Short Duration (Yellow)', 'Medium Duration (Orange)', 'Long Duration (Red)']
    legend_colors = [(1.0, 1.0, 0), (1.0, 0.5, 0), (1.0, 0, 0)]  # Colors for each label
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(handles, legend_labels, title="Color Meaning", loc='upper right')

    plt.show()

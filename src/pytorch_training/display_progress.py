import matplotlib.pyplot as plt
from math import ceil

def plot_history(history, split_list, metric_list, max_cols = 2, fp_history=None):
    plot_width, plot_height = 3, 2
    # Calculate number of plots
    num_metrics = len(metric_list)
    if num_metrics <= max_cols:
        num_rows = 1
        num_cols = num_metrics
    else:
        num_rows = ceil(num_metrics/max_cols)
        num_cols = max_cols
    # Get epoch vector
    num_epochs = len(history[split_list[0]][metric_list[0]])
    epochs = [i for i in range(num_epochs)]
    # Make axes
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols*plot_width, num_rows*plot_height), dpi=300)
    if num_rows > 1:
        axes = axes.flatten()
    if len(metric_list) == 1:
        axes = [axes]
    # Plot metrics
    for i, metric in enumerate(metric_list):
        for split in split_list:
            axes[i].plot(epochs, history[split][metric], label=split.capitalize() if i == 0 else None)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_xlabel("Epochs")
    # Remove additional plots
    if num_rows*num_cols > num_metrics:
        for j in range(i+1,num_rows*num_cols):
            axes[j].axis('off')
    # Add legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(split_list))
    plt.tight_layout()
    
    # Save plots or show plot
    if fp_history:
        plt.savefig(fp_history, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

def print_split_epoch_metrics(splitname, metric_dict):
    print(f"- {splitname.capitalize()}: ", end="")
    last_metrics_idx = len(metric_dict)-1
    for i, (metric, val) in enumerate(metric_dict.items()):
        print(f"{metric.capitalize()}: {val:.5f}", end=", " if i != last_metrics_idx else "\n")
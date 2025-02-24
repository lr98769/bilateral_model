import matplotlib.pyplot as plt

def plot_history(history, metrics_list, title=None):
    number_of_plots = len(metrics_list)
    subplot_width, subplot_height = 4, 2
    fig, axes = plt.subplots(1, number_of_plots, figsize=(subplot_width*number_of_plots, subplot_height), dpi=128)
    for i, metric in enumerate(metrics_list):
        if number_of_plots > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.plot(history.history[f'{metric}'])
        ax.plot(history.history[f'val_{metric}'])
        ax.set_ylabel(f'{metric}'.capitalize())
        ax.set_xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='center left', bbox_to_anchor=(1, 0.5))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    
def display_tuning_df(tuning_df):
    def highlight_sentiment(row):
        if row["best_hyperparameter"]:
            return ['background-color: #cafae6'] * len(row)
        else:
            return [''] * len(row)
    return tuning_df.style.apply(highlight_sentiment, axis=1)
import matplotlib.pyplot as plt

from methods.utils.accuracy import accuracy

def accuracy_vs_CPU(data):
    colors = ["#ad1fb4", "#0e62ff", "#2ca02c"] 
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^']
    linewidth = 0.5
    markersize = 2

    for i, (label, (processing_time, accuracy)) in enumerate(data.items()):
        ax.plot(
            processing_time,
            accuracy,
            marker=markers[i % len(markers)],
            color=colors[i],
            linewidth=linewidth,
            markersize=markersize,
            label=label
        )

    ax.set_title("Accuracy vs CPU Time", fontsize=22, fontweight='bold')
    ax.set_xlabel("CPU Time (seconds)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)

    ax.grid(False)
    
    ax.tick_params(colors='black')

    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()


def tracking(function, x_init, oracle, **kwargs):
    result = function(x_init, oracle, **kwargs)

    history = result["history"]
    times = result["times"]

    accs = [accuracy(oracle.predictors, oracle.predictions, x)
        for x in history]

    return times, accs

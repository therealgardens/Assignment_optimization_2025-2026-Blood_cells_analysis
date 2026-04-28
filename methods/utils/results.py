import matplotlib.pyplot as plt

def accuracy_vs_CPU(data):
    colors = ["#ad1fb4", "#0e62ff", "#2ca02c"] 
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^']
    linewidth = 2
    markersize = 2

    for i, (label, (processing_time, accuracy)) in enumerate(data.items()):
        ax.plot(
            processing_time,
            accuracy,
            # marker=markers[i % len(markers)],
            color=colors[i],
            linewidth=linewidth,
            markersize=markersize,
            label=label
        )

    ax.set_xlabel("CPU Time (seconds)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    ax.grid(False)
    
    ax.tick_params(colors='black')

    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

def distance_vs_iterations(data):
    colors = ["#ad1fb4", "#0e62ff", "#2ca02c"] 
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ['o', 's', '^']
    linewidth = 0.5
    markersize = 2

    for i, (label, (distances)) in enumerate(data.items()):
        ax.plot(
            list(range(len(distances))),
            distances,
            marker=markers[i % len(markers)],
            color=colors[i],
            linewidth=linewidth,
            markersize=markersize,
            label=label
        )

    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Distance to Solutions", fontsize=14)

    ax.grid(False)
    
    ax.tick_params(colors='black')

    ax.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

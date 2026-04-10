import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_benchmark_results(df, filename="benchmark.png", title="Benchmark Results"):
    """
    Generate bar chart comparing clique sizes found by different methods.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(x='Instance', y='Size', hue='Method', data=df, palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)
        
    plt.title(title)
    plt.ylabel('Clique Size')
    plt.xlabel('Graph Instance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved to {filename}")
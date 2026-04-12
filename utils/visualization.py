import matplotlib.pyplot as plt
import numpy as np

def plot_opinion_distribution(opinions, step, title="Opinion Distribution"):
    """
    Plots a histogram of opinions to visualize polarization clusters.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(opinions, bins=20, range=(-1, 1), color='skyblue', edgecolor='black')
    plt.title(f"{title} - Step {step}")
    plt.xlabel("Opinion Value")
    plt.ylabel("Number of Agents")
    plt.xlim(-1.1, 1.1)
    plt.grid(axis='y', alpha=0.5)
    plt.show() # If running in a notebook, this renders immediately
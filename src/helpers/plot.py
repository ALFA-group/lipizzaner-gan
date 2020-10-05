"""
When an experiment is completed, this file can be used to obtain all the plots
corresponding to that experiment. To achieve this, all values like
f(Generator(x))=..., f(Discriminator(x))=... and score=... in the log file
corresponding to a client are copied to a dedicated file such as generator.txt,
discriminator.txt and fid.txt respectively (as indicated in the code below)
following which the required plots are obtained
"""
import time
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


n_iterations = 200
time_start = int(time.time() * 1000)


def read_output_from_file(file_name, arr):
    """
    Reads files containing only individual g/d loss or FID/IS scores from a file
    Args:
        file_name: file from which entries corresponding to g/d loss or FID/IS
        scores are read
        arr: the array into which these entries are written to
    Returns:
        the array containing all the entries pertaining to the experiment
    """
    with open(file_name) as f:
        for line in f:
            arr.append([float(x) for x in line.split()])


def plot_loss():
    x = [i for i in range(1, n_iterations + 1)]

    f_generator = []
    f_discriminator = []
    # Insert Correct Path to generator.txt
    read_output_from_file("generator.txt", f_generator)
    # Insert Correct Path to discriminator.txt
    read_output_from_file("discriminator.txt", f_discriminator)

    # Plot generator loss
    plt.plot(x, f_generator, label="generator")
    # Plot discriminator loss
    plt.plot(x, f_discriminator, label="discriminator")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Show legend
    plt.legend()
    plt.savefig(f"../plots/plot_{time_start}.png")


def plot_fid_is():
    x = [i for i in range(1, n_iterations + 1)]

    scores = []
    # Insert Correct Path to fid.txt
    read_output_from_file("fid.txt", scores)

    plt.plot(x, scores)

    plt.xlabel("Epoch")
    # Change the ylabel to Inception Score when appropriate
    plt.ylabel("'FID Score'")
    # Change filename to start with is_ when using Inception Score
    plt.savefig(f"../plots/fid_{time_start}.png")


# Plot g/d loss
plot_loss()
# Plot FID/IS scores
plot_fid_is()

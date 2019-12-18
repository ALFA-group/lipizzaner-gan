import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time


time_start = int(time.time() * 1000)


def read_output_from_file(file_name, arr):
    with open(file_name) as f:
        for line in f:
            arr.append([float(x) for x in line.split()])


def plot_loss():
    # plt.figure(figsize=(40,40))
    x = [i for i in range(1, 201)]

    f_generator = []
    f_discriminator = []
    # Insert Correct Path to generator.txt
    read_output_from_file("generator.txt", f_generator)
    # Insert Correct Path to discriminator.txt
    read_output_from_file("discriminator.txt", f_discriminator)

    plt.plot(x, f_generator, label='generator')
    plt.plot(x, f_discriminator, label='discriminator')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'../plots/plot_{time_start}.png')


def plot_fid():
    x = [i for i in range(1, 201)]

    fids = []
    # Insert Correct Path to fid.txt
    read_output_from_file("fid.txt", fids)

    plt.plot(x, fids)

    plt.xlabel("Epoch")
    plt.ylabel("'FID Score'")
    plt.savefig(f'../plots/fid_{time_start}.png')


plot_loss()
plot_fid()

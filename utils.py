import matplotlib.pyplot as plt
import numpy as np


def concatenate_dict(main_dict, new_dict):
    for key in main_dict.keys():
        main_dict[key] += [new_dict[key]]


def plot_image(arr):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin='lower', aspect='auto', interpolation='nearest')
    fig.colorbar(im)
    return fig


def plot_lines(arr):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    for i in range(arr.shape[0]):
        ax.plot(arr[i], label='%d' % i)
    ax.legend()
    return fig


def draw(offsets, ascii_seq=None, save_file=None):
    strokes = np.concatenate(
        [offsets[:, 0:1], np.cumsum(offsets[:, 1:], axis=0)],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 3))

    stroke = []
    for eos, x, y in strokes:
        stroke.append((x, y))
        if eos == 1:
            xs, ys = zip(*stroke)
            ys = np.array(ys)
            ax.plot(xs, ys, 'k', c='blue')
            stroke = []

    if stroke:
        xs, ys = zip(*stroke)
        ys = np.array(ys)
        ax.plot(xs, ys, 'k', c='blue')
        stroke = []

    ax.set_xlim(-50, 600)
    ax.set_ylim(-40, 40)
    ax.axis('off')

    ax.set_aspect('equal')
    ax.tick_params(
        axis='off', left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labeltop=False,
        labelright=False, labelbottom=False
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)

    return fig


def draw_multiple(list_of_offsets, ascii_seq=None, save_file=None):
    list_of_strokes = []
    for offsets in list_of_offsets:
        strokes = np.concatenate(
            [offsets[:, 0:1], np.cumsum(offsets[:, 1:], axis=0)],
            axis=1
        )
        list_of_strokes.append(strokes)

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, strokes in enumerate(list_of_strokes):
        strokes[:, -1] -= 30 * i

        stroke = []
        for eos, x, y in strokes:
            stroke.append((x, y))
            if eos == 1:
                xs, ys = zip(*stroke)
                ys = np.array(ys)
                ax.plot(xs, ys, 'k', c='blue')
                stroke = []

        if stroke:
            xs, ys = zip(*stroke)
            ys = np.array(ys)
            ax.plot(xs, ys, 'k', c='blue')

    # ax.set_xlim(-50, 600)
    # ax.set_ylim(-200, 200)
    ax.axis('off')

    ax.set_aspect('equal')
    ax.tick_params(
        axis='off', left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labeltop=False,
        labelright=False, labelbottom=False
    )

    if ascii_seq is not None:
        if not isinstance(ascii_seq, str):
            ascii_seq = ''.join(list(map(chr, ascii_seq)))
        plt.title(ascii_seq)

    if save_file is not None:
        plt.savefig(save_file)

    return fig

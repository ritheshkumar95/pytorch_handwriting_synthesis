import matplotlib.pyplot as plt
import numpy as np


def draw(
        offsets, ascii_seq=None,
        save_file=None
):

    offsets[:, 1:] = offsets[:, 1:] * 10
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
            ax.plot(xs, ys, 'k', c='blue')
            stroke = []

    if stroke:
        xs, ys = zip(*stroke)
        ax.plot(xs, ys, 'k', c='blue')
        stroke = []

    ax.set_xlim(-50, 600)
    ax.set_ylim(-40, 40)

    ax.set_aspect('equal')
    ax.tick_params(
        axis='both', left=False, right=False,
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
        # print('saved to {}'.format(save_file))

    return fig

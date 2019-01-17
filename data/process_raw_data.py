import argparse
import glob
from pathlib import Path
from xml.etree.ElementTree import ElementTree
import numpy as np
from utils import align, denoise, normalize, coords_to_offsets, draw
from tqdm import tqdm
import h5py


def get_strokes(filename, max_stroke_len):
    et = ElementTree().parse(stroke_path)
    arr = []
    for stroke in et[1]:  # loop through Strokes in a StrokeSet
        for point in stroke:  # loop through points in a Stroke
            arr.append([
                int(point.attrib['x']),
                int(point.attrib['y']),
                0
            ])
        arr[-1][-1] = 1  # Stroke ends, pen up!

    coords = np.array(arr)
    coords = align(coords)
    coords = denoise(coords)
    offsets = coords_to_offsets(coords)[:max_stroke_len]
    offsets = normalize(offsets)
    return offsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', default='data.hdf5')
    parser.add_argument("--data_path", default='/Users/rithesh/iam')
    parser.add_argument("--max_stroke_len", type=int, default=1200)
    args = parser.parse_args()
    return args


args = parse_args()
f = h5py.File(args.save_file, 'w')
files = glob.glob(
    args.data_path + "/ascii/**/*.txt",
    recursive=True
)[:100]

sentences = []
strokePaths = []

for i, file in tqdm(enumerate(files)):
    root = Path(file)

    stroke_root = str(root).replace('ascii', 'lineStrokes')
    if not Path(stroke_root).parent.exists():
        print("Skipping %s" % str(root))
        continue

    data = open(root).read().splitlines()
    data = [x.strip() for x in data]
    start_idx = data.index('CSR:') + 2
    text = data[start_idx:]
    sentences += [np.string_(x) for x in text]

    for j in range(1, 1 + len(text)):
        stroke_path = str(root).replace('ascii', 'lineStrokes')
        stroke_path = stroke_path.replace('.txt', '-%02d.xml' % j)
        strokePaths.append(stroke_path)


# print(sentences)
total_string = np.string_('')
for sent in sentences:
    total_string += sent

print(total_string)
import ipdb; ipdb.set_trace()
vocab = list(set(total_string))
word2idx = {x: i for i, x in enumerate(vocab)}

idxs = [[word2idx[char] for char in line] for line in sentences]
chars = [''.join([vocab[i] for i in line]) for line in idxs]

f.attrs['vocab'] = str(vocab)
f.attrs['char2idx'] = str(word2idx)

f.create_dataset('sentences', data=sentences, dtype=h5py.special_dtype(vlen=str))
f.close()

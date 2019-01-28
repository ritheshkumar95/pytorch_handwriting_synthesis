import argparse
import glob
from pathlib import Path
from xml.etree.ElementTree import ElementTree
import numpy as np
from utils import align, denoise, normalize, coords_to_offsets
from multiprocessing import Pool
from tqdm import tqdm
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', default='data.hdf5')
    parser.add_argument("--data_path", default='/data/iam_ondb')
    parser.add_argument("--max_stroke_len", type=int, default=1200)
    args = parser.parse_args()
    return args


def get_sentences(filename):
    data = open(filename).read().splitlines()
    data = [x.strip() for x in data]
    start_idx = data.index('CSR:') + 2
    return [list(x) for x in data[start_idx:]]


def get_strokes(filename):
    et = ElementTree().parse(filename)
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
    offsets = coords_to_offsets(coords)[:args.max_stroke_len]
    offsets = normalize(offsets)
    return offsets

if __name__ == '__main__':
    args = parse_args()
    f = h5py.File(args.save_file, 'w')
    files = glob.glob(
        args.data_path + "/ascii/**/*.txt",
        recursive=True
    )

    #################################################################
    # Extracting sentences and corresponding stroke paths           #
    #################################################################
    sentences = []
    strokePaths = []

    for i, file in tqdm(enumerate(files)):
        root = Path(file)
        text = get_sentences(root)

        stroke_root = str(root).replace('ascii', 'lineStrokes')
        # Bugs in database unfortunately :(
        if not Path(stroke_root).parent.exists():
            print("Skipping %s" % str(root))
            continue

        for j, line in enumerate(text):
            stroke_path = str(root).replace('ascii', 'lineStrokes')
            stroke_path = stroke_path.replace('.txt', '-%02d.xml' % j)

            # Bugs in database unfortunately :(
            if not Path(stroke_path).exists():
                print("Skipping %s" % stroke_path)
                continue

            strokePaths.append(stroke_path)
            sentences += [line]

    ################################################
    # Create vocabulary, char2idx                  #
    ################################################
    total_string = ['']
    for sent in sentences:
        total_string += sent

    vocab = sorted(list(set(total_string)))
    char2idx = {x: i for i, x in enumerate(vocab)}

    f.attrs['vocab'] = str(vocab)
    f.attrs['char2idx'] = str(char2idx)

    print("Created vocabulary...")

    ################################################
    # Convert sentences to indices                 #
    ################################################
    idx_sentences = [[char2idx[x] for x in sent] for sent in sentences]
    lengths = [len(x) for x in idx_sentences]
    max_char_len = int(np.percentile(lengths, 95))

    chars_arr = np.zeros((len(sentences), max_char_len), dtype='int64')
    chars_mask_arr = np.zeros((len(sentences), max_char_len), dtype='float32')

    for i, sent in enumerate(idx_sentences):
        length = min(len(sent), max_char_len)
        chars_arr[i, :length] = sent[:length]
        chars_mask_arr[i, :length] = 1.

    f.create_dataset('sentences', data=sentences, dtype=h5py.special_dtype(vlen=str))
    f['chars'] = chars_arr
    f['chars_mask'] = chars_mask_arr

    ################################################
    # Extract strokes from strokePaths             #
    ################################################
    pool = Pool()
    strokes = pool.map(get_strokes, strokePaths)

    import ipdb; ipdb.set_trace()

    # strokes = [get_strokes(path) for path in strokePaths]
    strokes_arr = np.zeros((len(strokes), args.max_stroke_len, 3), dtype='float32')
    strokes_mask_arr = np.zeros((len(strokes), args.max_stroke_len), dtype='float32')

    for i, stk in enumerate(strokes):
        strokes_arr[i, :stk.shape[0]] = stk
        strokes_mask_arr[i, :stk.shape[0]] = 1.

    f['strokes'] = strokes_arr
    f['strokes_mask'] = strokes_mask_arr
    f.close()

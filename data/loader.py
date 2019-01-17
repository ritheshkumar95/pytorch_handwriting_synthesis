import glob
from pathlib import Path
from xml.etree.ElementTree import ElementTree
import numpy as np
from utils import align, denoise, normalize, coords_to_offsets, draw
from tqdm import tqdm
import pickle


class DataLoader(object):
    def __init__(self, data_path, max_stroke_len):
        self.path = Path(data_path)
        self.max_stroke_len = max_stroke_len

        if (self.path / 'processed.pkl').exists():
            self.sentences, self.lineStrokes = pickle.load(
                open(self.path / 'processed.pkl', 'rb')
            )
        else:
            self.sentences, self.lineStrokes = self.process_data()
            pickle.dump(
                [self.sentences, self.lineStrokes],
                open(self.path / 'processed.pkl', 'wb')
            )

    def process_data(self):
        files = glob.glob(str(self.path / "ascii/**/*.txt"), recursive=True)[:100]
        sentences = []
        lineStrokes = []

        for i, file in tqdm(enumerate(files)):
            root = Path(file)
            data = open(root).read().splitlines()
            data = [x.strip() for x in data]
            start_idx = data.index('CSR:') + 2
            text = data[start_idx:]
            sentences += text

            try:
                for j in range(1, 1 + len(text)):
                    stroke_path = str(root).replace('ascii', 'lineStrokes')
                    stroke_path = stroke_path.replace('.txt', '-%02d.xml' % j)
                    et = ElementTree().parse(stroke_path)
                    arr = []
                    for stroke in et[1]:  # loop through Strokes in a StrokeSet
                        for point in stroke:  # loop through points in a Stroke
                            arr.append([int(point.attrib['x']), int(point.attrib['y']), 0])
                        arr[-1][-1] = 1  # Stroke ends, pen up!

                    coords = np.array(arr)
                    coords = align(coords)
                    coords = denoise(coords)
                    offsets = coords_to_offsets(coords)[:self.max_stroke_len]
                    offsets = normalize(offsets)
                    lineStrokes.append(offsets)
            except FileNotFoundError:
                for j in range(len(text)):
                    sentences.pop()

        return sentences, lineStrokes


if __name__ == '__main__':
    loader = DataLoader('/Users/rithesh/iam', 1200)
    for i in range(100):
        draw(loader.lineStrokes[i], loader.sentences[i])
        input()
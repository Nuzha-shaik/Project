import os
import struct
from array import array

class MNISTLoader:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

    def load_data(self, image_file, label_file):
        image_path = os.path.join(self.dataset_path, image_file)
        label_path = os.path.join(self.dataset_path, label_file)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ Label file not found: {label_path}")

        with open(label_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            if magic != 2049:
                raise ValueError("❌ Invalid magic number in label file")
            labels = array('B', lbpath.read())

        with open(image_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            if magic != 2051:
                raise ValueError("❌ Invalid magic number in image file")
            image_data = array('B', imgpath.read())

        images = []
        for i in range(len(labels)):
            start = i * rows * cols
            end = start + rows * cols
            images.append(image_data[start:end])

        return images, labels

    def display_image(self, img, width=28):
        result = ''
        for i in range(len(img)):
            if i % width == 0:
                result += '\n'
            result += '@' if img[i] > 128 else '.'
        return result


if __name__ == '__main__':
    loader = MNISTLoader(dataset_path='dataset')
    try:
        images, labels = loader.load_data('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
        print("✅ Successfully loaded", len(images), "images")
        print("🔢 First label:", labels[0])
        print(loader.display_image(images[0]))
    except Exception as e:
        print(e)

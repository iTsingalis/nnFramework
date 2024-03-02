

def load_mnist(path, perc_samples=None, kind='train', random_state=2):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784).astype(float)
    # images /= 256.0
    unique_classes = np.unique(labels)
    n_samples = len(labels)

    if n_samples:

        images_ = []
        labels_ = []
        for c in unique_classes:
            images_c = images[labels == c]
            labels_c = labels[labels == c]
            num_rows, num_cols = images_c.shape

            images_.append(images_c[:int(num_rows * perc_samples)])
            labels_.append(labels_c[:int(num_rows * perc_samples)])

        images = np.vstack(images_)
        labels = np.hstack(labels_)

    return images, labels

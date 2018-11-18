from time import time
from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, decomposition)


# Prints MNIST data
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 10e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            print(shown_images)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage((np.reshape(X_images[i], (28, 28)) * 255).astype(np.uint8), cmap=plt.cm.gray_r, zoom=1),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# Fetch MNIST from data directory
mnist = MNIST('data')

# Used only 10k images to check the model
# train_images, train_labels = mnist.load_training()
test_images, test_labels = mnist.load_testing()

X_sample = test_images
y_sample = test_labels

# Filter out images from range 0 to 9
# if 10 it means all for the last loop
for digitFilter in range(11):
    X = []
    y = []
    if digitFilter == 10:
        # No Filter
        X_images = X_sample
        X = np.array(X_sample)
        y = np.array(y_sample)
    else:
        for i in range(len(y_sample)):
            if y_sample[i] == digitFilter:
                X.append(X_sample[i])
                y.append(y_sample[i])
        X_images = X
        X = np.array(X)
        y = np.array(y)

    n_samples, n_features = X.shape
    n_neighbors = 30

    # Plot images of the digits
    n_img_per_row = 20
    img = np.zeros((30 * n_img_per_row, 30 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 30 * i + 1
        for j in range(n_img_per_row):
            iy = 30 * j + 1
            img[ix:ix + 28, iy:iy + 28] = X[i * n_img_per_row + j].reshape((28, 28))

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 784-dimensional digits dataset')

    # Projection on to the first 2 principal components
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.PCA(n_components=2).fit_transform(X)
    plot_embedding(X_pca,
                   "PCA projection of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()

    # Locally linear embedding of the digits dataset
    print("Computing LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='standard')
    t0 = time()
    X_lle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_lle,
                   "LLE projection of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()

    # ICA of the digits dataset
    print("Computing ICA embedding")
    t0 = time()
    X_ica = decomposition.FastICA(n_components=2).fit_transform(X)
    plot_embedding(X_ica,
                   "ICA of projection the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()

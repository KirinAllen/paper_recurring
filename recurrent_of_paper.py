# Function for reading MNIST data
from __future__ import print_function
import gzip
import os
import urllib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy


def get_minst_images_dir():
    return r'D:\workAndstudy\dataset\mnist\train-images-idx3-ubyte.gz'


def get_minst_labels_dir():
    return r'D:\workAndstudy\dataset\mnist\train-labels-idx1-ubyte.gz'


# 逐位读取字节流中的字符
def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')  # 每次读取都让当前位置下标置零
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]  # 返回数据集中的当前下标位的值


def extract_images(filename):
    # Extract the image into a 4D uint8 numpy array [index,y,x,depth]
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:  # MINST数据集的格式中，前几位的说明中，第一位就是magic标志
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)  # 从上面读取了四个字符后面开始读起，由MINST的组织格式知道后面全是像素值
        data = numpy.frombuffer(buf, dtype=numpy.uint8)  # 将buffer中的数据以指定数据类型array化
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    # Convert class labels from scalars to one-hot vectors.
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes  # 为每个类标签生成一个10也即类别数目数组偏移量，为后面[1,0,0,0,0,0,0,0,0,0]对应标签0
    labels_one_hot = numpy.zeros((num_labels, num_classes))  # [0,1,0,0,0,0,0,0,0,0]对应标签1，以此类推
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    # Extract the labels into 1D uint8 numpy array [index].
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file:%s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (  # 当此条件为假时，抛出此断言警告
                    "images.shape: %s labels.shape: %s" % (images.shape, labels.shape)
            )
            self._num_examples = images.shape[0]

            # convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth ==1)
            #   assert images.shape[3] == 1
            images = images.reshape(images.shape[0],  # reshape后由原来的(55000,28,28,1)变成(55000,784)
                                    images.shape[1] * images.shape[2] * images.shape[3])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)  # 像素数据类型转换，为了更好地归一化
            images = numpy.multiply(images, 1.0 / 255.0)  # 归一化像素点
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_bath(self, batch_size, fake_data=False):
        # Return the next 'batch_size' examples from this data set.
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled):
        self.n_labeled = n_labeled
        self.unlabeled_ds = DataSet(images, labels)
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)  # 打乱原来的索引顺序，生成新的索引，但不改变原来索引的数据
        images = images[shuffled_indices]  # 样本中图像的顺序被随机打乱了
        labels = labels[shuffled_indices]  # 样本中的图像的标签也按同一随机顺序打乱了，为了与原图像对齐
        y = numpy.array([numpy.arange(10)[l == 1][0] for l in labels])  # 迭类似[0,1,...]的标签数据，然后让l==1的迭代位n就是原标签对应的数字
        idx = indices[y == 0][:5]  # 选取5个y中0对应的索引值
        n_classes = y.max() + 1
        n_from_each_class = n_labeled // n_classes  # py3 adapt 每一类别选出的标签个数，这里是每类选90个标签
        i_labeled = []
        for c in range(n_classes):  # 完成此步骤后就可以为0,1,2....9每个类别选出90个样本并按从小到大的顺序排列在列表当中
            i = indices[y == c][:n_from_each_class]  # 从y中选出90个y==c的标签
            i_labeled += list(i)  # 把对应该数字(类别)的样本添加到标签中
            # print(i_labeled)
        l_images = images[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)  # 将抽样好的样本送入DataSet中进一步处理，如归一化等操作
        # 以下对应于论文中基于类别抽样方法，抽三次，每次每个样本0~9抽30个，全部抽完后对应了900个样本
        i_labeled_0 = []
        for c_0 in range(n_classes):
            i_0 = indices[y == c_0][:30]
            i_labeled_0 += list(i_0)
        # print(i_labeled_0)
        l_images_0 = images[i_labeled_0]
        l_labels_0 = labels[i_labeled_0]
        self.S0 = DataSet(l_images_0, l_labels_0)

        i_labeled_1 = []
        for c_1 in range(n_classes):
            i_1 = indices[y == c_1][30:60]
            i_labeled_1 += list(i_1)
        # print(i_labeled_1)
        l_images_1 = images[i_labeled_1]
        l_labels_1 = labels[i_labeled_1]
        self.S1 = DataSet(l_images_1, l_labels_1)

        i_labeled_2 = []
        for c_2 in range(n_classes):
            i_2 = indices[y == c_2][60:90]
            i_labeled_2 += list(i_2)
        # print(i_labeled_2)
        l_images_2 = images[i_labeled_2]
        l_labels_2 = labels[i_labeled_2]
        self.S2 = DataSet(l_images_2, l_labels_2)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_bath(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_bath(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_bath(batch_size)
        return labeled_images, labels, unlabeled_images

    def next_batch_0(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            images_0, labels_0 = self.S0.next_batch(self.n_labeled)
        else:
            images_0, labels_0 = self.S0.next_batch(batch_size)
        return images_0, labels_0, unlabeled_images

    def next_batch_1(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            images_1, labels_1 = self.S1.next_batch(self.n_labeled)
        else:
            images_1, labels_1 = self.S1.next_batch(batch_size)
        return images_1, labels_1, unlabeled_images

    def next_batch_2(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            images_2, labels_2 = self.S2.next_batch(self.n_labeled)
        else:
            images_2, labels_2 = self.S2.next_batch(batch_size)
        return images_2, labels_2, unlabeled_images


def read_data_sets(train_dir, n_labeled=900, image_size=[28, 28, 1], fake_data=False, one_hot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()

    n_c = image_size[-1]
    image_size = image_size[:-1]  # 从位置0到位置-1之间的数,这里指前两个28

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        return data_sets

    TRAIN_IMAGES = r'D:\workAndstudy\dataset\mnist\train-images-idx3-ubyte.gz'
    TRAIN_LABELS = r'D:\workAndstudy\dataset\mnist\train-labels-idx1-ubyte.gz'
    TEST_IMAGES = r'D:\workAndstudy\dataset\mnist\t10k-images-idx3-ubyte.gz'
    TEST_LABELS = r'D:\workAndstudy\dataset\mnist\t10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000

    train_images = extract_images(TRAIN_IMAGES)
    train_labels = extract_labels(TRAIN_LABELS, one_hot=one_hot)
    test_images = extract_images(TEST_IMAGES)
    test_labels = extract_labels(TEST_LABELS, one_hot=one_hot)

    resized = []
    for x in train_images:
        resized.append(cv2.resize(x, tuple(image_size)))  # 原图x的shape是(28,28,1)，这里resize后变成(28,28)也就是image_size的shape
    train_images = np.array(resized)  # shape从(60000,28,28,1)变成(60000,28,28)

    train_images = np.expand_dims(train_images, -1)  # shape又变成(60000,28,28,1)
    train_images = np.repeat(train_images, n_c, axis=-1)  # shape还是(60000,28,28,1)

    resized = []
    for x in test_images:
        resized.append(cv2.resize(x, tuple(image_size)))
    test_images = np.array(resized)

    test_images = np.expand_dims(test_images, -1)
    test_images = np.repeat(test_images, n_c, axis=-1)

    validation_images = test_images[:VALIDATION_SIZE]
    validation_labels = test_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled)  # 基于类别抽样实现
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets


if __name__ == "__main__":

    # 测试
    my_minst_image_path = get_minst_images_dir()
    minst_image_datas = extract_images(my_minst_image_path)

    # 显示从字符流中读取到的图片
    for i in range(16):
        sp = plt.subplot(4, 4, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        plt.imshow(minst_image_datas[i].squeeze())
    plt.show()

    my_minst_labels_path = get_minst_labels_dir()
    minst_image_labels = extract_labels(my_minst_labels_path)

    # 打印前16个标签比对前面的图片，这是一一对应的
    print_labels = numpy.zeros((16,))
    for i in range(16):
        print_labels[i] = minst_image_labels[i]
    print(print_labels.reshape(4, 4))

    # 基于类别抽样法,打印出各个类别的抽样图片，每个样本打印16张
    data = read_data_sets('MNIST_data', n_labeled=900, one_hot=True)

    S0_images = data.train.S0.images.reshape(300, 28, 28)
    for i in range(16):
        plt.suptitle('S0_samples')
        sp = plt.subplot(4, 4, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        plt.imshow(S0_images[i * 18])
    plt.show()

    S1_images = data.train.S1.images.reshape(300, 28, 28)
    for i in range(16):
        plt.suptitle('S1_samples')
        sp = plt.subplot(4, 4, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        plt.imshow(S1_images[i * 18])
    plt.show()

    S2_images = data.train.S2.images.reshape(300, 28, 28)
    for i in range(16):
        plt.suptitle('S2_samples')
        sp = plt.subplot(4, 4, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        plt.imshow(S2_images[i * 18])
    plt.show()

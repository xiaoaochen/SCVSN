from keras import backend as K
from Parser import parameter_parser
import matplotlib.pyplot as plt
import random
import numpy as np

args = parameter_parser()


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06
    https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    """
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    num_classes = args.num_classes
    pairs = []  # 样本对
    labels = []  # 标签
    n = min([len(digit_indices[d])
             for d in range(num_classes)]) - 1  # 因为正：负 = 1：1所以n为训练集的1半
    for d in range(num_classes):  # d表示第几类 ，假设D为0，当d为1时
        for i in range(n):  # i 表示索引中的第几个
            z1, z2 = digit_indices[d][i], digit_indices[d][
                i + 1]  # z1 （0，0）,z2(0,1)  z1(1,0) z2(1,1)
            # print("d,i", d, i, d, i + 1)
            pairs += [[x[z1], x[z2]]]  # 组成相似的pairs
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes  # dn = 0
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]  # (1,0),
            # print("d_1, i_1", d, i, dn, i)
            pairs += [[x[z1], x[z2]]]  # 组成相反的Paris
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):  # numpy上的操作
    """
    Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() <= 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):  # Tensor上的操作
    """
    Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o', MarkerSize=1)
    plt.plot(history.history.get(val_metrics), '-o', MarkerSize=1)
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

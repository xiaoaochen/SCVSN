import time

from keras.layers import LSTM, Input, Flatten, Dense, Dropout, Lambda, ReLU
from sklearn.metrics import accuracy_score
from keras.callbacks import CSVLogger
from Parser import parameter_parser
from keras.models import Model
from sklearn.utils import compute_class_weight
import numpy as np
from Simase_netwrok import create_pairs, euclidean_distance, eucl_dist_output_shape, accuracy, contrastive_loss, \
    plot_train_history, compute_accuracy
from get_df import get_df
from keras.optimizers import Adamax
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical

args = parameter_parser()


def create_base_network(input_shape):
    """
    神经网络
    """
    inp = Input(input_shape)
    models = LSTM(units=300, input_shape=input_shape)(inp)
    models = ReLU()(models)
    models = (Dropout(args.dropout))(models)
    models = Dense(units=args.cell_size)(models)
    models = ReLU()(models)
    models = (Dropout(args.dropout))(models)
    models = Dense(args.num_classes, activation='relu')(models)
    return Model(inp, models)


def train():
    x_train, x_test, y_train, y_test, input_shape, labels = get_df()
    class_weight = compute_class_weight(class_weight='balanced',
                                        classes=[0, 1],
                                        y=labels)
    # create training+test positive and negative pairs
    csv_logger = CSVLogger('log.txt', append=True, separator=',')
    digit_indices = [np.where(y_train == i)[0] for i in range(2)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(2)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    # network definition
    base_network = create_base_network(input_shape)
    # plot_model(base_network, to_file='model1_structure.png', show_shapes=True)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    model.summary()
    # plot_model(model, to_file='model_structure.png', show_shapes=True)
    # train
    admax = Adamax(lr=0.002)
    model.compile(loss=contrastive_loss, optimizer=admax, metrics=[accuracy])
    print("测试集的长度为:", len(tr_pairs[:, 0]))
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]],
                        tr_y,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        class_weight=class_weight,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
                        callbacks=[csv_logger])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'accuracy', 'val_accuracy')
    plt.show()

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]], batch_size=64)

    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]], batch_size=64)
    te_acc = compute_accuracy(te_y, y_pred)
    y_pred = (y_pred.ravel() <= 0.5).astype(int)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    tn, fp, fn, tp = confusion_matrix(y_true=te_y.tolist(),
                                      y_pred=y_pred.tolist()).ravel()
    print(tn, fp, fn, tp)
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print('* False positive rate(FPR): %0.2f%%', fp / (fp + tn))
    print('* False negative rate(FNR): %0.2f%%', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('* Recall(TPR): %0.2f%%', recall)
    precision = tp / (tp + fp)
    print('* Precision: %0.2f%%', precision)
    print('* F1 score: ', (2 * precision * recall) / (precision + recall))
    with open('log.txt', mode='a') as f:
        f.write("test accuracy:" + str((tp + tn) / (tp + tn + fp + fn)) + "\n")
        f.write('False positive rate(FP): ' + str(fp / (fp + tn)) + "\n")
        f.write('False negative rate(FN): ' + str(fn / (fn + tp)) + "\n")
        f.write('Recall: ' + str(recall) + "\n")
        f.write('Precision: ' + str(precision) + "\n")
        f.write('F1 score: ' + str((2 * precision * recall) / (precision + recall)) + '\n')
        f.write("-------------------------------" + "\n")


def train_single():
    x_train, x_test, y_train, y_test, input_shape, labels = get_df()

    class_weight = compute_class_weight(class_weight='balanced',
                                        classes=[0, 1],
                                        y=labels)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    csv_logger = CSVLogger('log_single.txt', append=True, separator=',')

    # network definition
    model = create_base_network(input_shape)
    admax = Adamax(lr=0.002)
    model.compile(loss=contrastive_loss, optimizer=admax, metrics=[accuracy])
    history = model.fit(x_train,
                        y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        class_weight=class_weight,
                        validation_data=(x_test, y_test),
                        callbacks=[csv_logger])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss', 'val_loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'accuracy', 'val_accuracy')
    plt.show()

    # compute final accuracy on training and test sets
    predictions = (model.predict(x_test, batch_size=64)).round()

    tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1),
                                      np.argmax(predictions, axis=1)).ravel()

    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    print('* False positive rate(FPR): %0.2f%%', fp / (fp + tn))
    print('* False negative rate(FNR): %0.2f%%', fn / (fn + tp))
    recall = tp / (tp + fn)
    print('* Recall(TPR): %0.2f%%', recall)
    precision = tp / (tp + fp)
    print('* Precision: %0.2f%%', precision)
    print('* F1 score: ', (2 * precision * recall) / (precision + recall))
    with open('log_single.txt', mode='a') as f:
        f.write("test accuracy:" + str((tp + tn) / (tp + tn + fp + fn)) + "\n")
        f.write('False positive rate(FP): ' + str(fp / (fp + tn)) + "\n")
        f.write('False negative rate(FN): ' + str(fn / (fn + tp)) + "\n")
        f.write('Recall: ' + str(recall) + "\n")
        f.write('Precision: ' + str(precision) + "\n")
        f.write('F1 score: ' + str((2 * precision * recall) / (precision + recall)) + '\n')
        f.write("-------------------------------" + "\n")


time1 = time.time()
train()
time2 = time.time()
print("使用时间", time2 - time1)

# 使用时间 70.49699926376343

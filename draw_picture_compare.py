import csv
import matplotlib.pyplot as plt
csv_reader = csv.reader(open("dataset_siamese.CSV"))
epoch = []
accuracy = []
loss = []
accuracy_1 = []
loss_1 = []
for info in csv_reader:
    if csv_reader.line_num == 1:
        continue
    epoch.append(int(info[0]) + 1)
    accuracy.append(float(info[1]) * 100)
    loss.append(float(info[2]) * 100)
    accuracy_1.append(float(info[3]) * 100)
    loss_1.append(float(info[4]) * 100)
flag = 1
csv_reader_lstm = csv.reader(open("dataset.CSV"))
epoch_lstm = []
accuracy_lstm = []
loss_lstm = []
accuracy_1_lstm = []
loss_1_lstm = []
for info in csv_reader_lstm:
    if csv_reader_lstm.line_num == 1:
        continue
    epoch_lstm.append(int(info[0]) + 1)
    accuracy_lstm.append(float(info[1]) * 100)
    loss_lstm.append(float(info[2]) * 100)
    accuracy_1_lstm.append(float(info[3]) * 100)
    loss_1_lstm.append(float(info[4]) * 100)
flag = 1

# plt.plot(epoch, accuracy,lw=1.5, color='b', label='train acc', marker='.', markevery=1,
#                      mew=1)
plt.plot(epoch, accuracy_1,lw=1.5, color='y', label='train val-acc', marker='.', markevery=1,
                     mew=1)
# plt.plot(epoch, accuracy_lstm,lw=1.5, color='b', label='train acc', marker='.', markevery=1,
#                      mew=1)
plt.plot(epoch, accuracy_1_lstm,lw=1.5, color='b', label='train val-acc', marker='.', markevery=1,
                     mew=1)
plt.show()
# plt.plot(epoch, loss, lw=1.5, color='r', label='train loss', marker='.', markevery=1,
#                      mew=1)
# plt.plot(epoch, loss_1, lw=1.5, color='g', label='train val-loss', marker='.', markevery=1,
#                      mew=1)
# plt.show()
# plt.plot(epoch, accuracy,lw=1.5, color='b', label='train acc', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, loss, lw=1.5, color='r', label='train loss', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, accuracy_1,lw=1.5, color='y', label='train val-acc', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, loss_1, lw=1.5, color='g', label='train val-loss', marker='.', markevery=1,
#                  mew=1)
plt.xlabel("epochs", fontsize=12)
plt.ylabel("value(%)", fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.show()

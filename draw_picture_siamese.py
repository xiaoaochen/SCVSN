import csv
import matplotlib.pyplot as plt
csv_reader = csv.reader(open("dataset_siamese.CSV"))
epoch = []
accuracy = []
loss = []
accuracy_1 = []
loss_1 = []
plt.figure(figsize=(8, 4))
for info in csv_reader:
    if csv_reader.line_num == 1:
        continue
    epoch.append(int(info[0]) + 1)
    accuracy.append(float(info[1]) * 100)
    loss.append(float(info[2]) * 100)
    accuracy_1.append(float(info[3]) * 100)
    loss_1.append(float(info[4]) * 100)
# flag = 1
# if flag == 1:
#     plt.plot(epoch, accuracy,lw=1.5, color='darkorange', label='train acc', marker='.', markevery=1,
#                      mew=1)
#     plt.plot(epoch, accuracy_1,lw=1.5, color='royalblue', label='train val-acc', marker='.', markevery=1,
#                      mew=1)
# else:
#     plt.plot(epoch, loss, lw=1.5, color='r', label='train loss', marker='.', markevery=1,
#                      mew=1)
#     plt.plot(epoch, loss_1, lw=1.5, color='g', label='train val-loss', marker='.', markevery=1,
#                      mew=1)

# plt.plot(epoch, accuracy,lw=1.5, color='b', label='train acc', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, loss, lw=1.5, color='r', label='train loss', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, accuracy_1,lw=1.5, color='y', label='train val-acc', marker='.', markevery=1,
#                  mew=1)
# plt.plot(epoch, loss_1, lw=1.5, color='g', label='train val-loss', marker='.', markevery=1,
#                  mew=1)
ax1 = plt.subplot(1, 2, 1)
plt.plot(epoch, accuracy, lw=1.5, color='darkorange', label='train acc', marker='.', markevery=1,
                 mew=1)
plt.plot(epoch, accuracy_1, lw=1.5, color='royalblue', label='train val-acc', marker='.', markevery=1,
                 mew=1)
plt.xlabel("epochs", fontsize=12)
plt.ylabel("accuracy(%)", fontsize=12)
plt.legend(loc='best', fontsize=12)
ax2 = plt.subplot(1, 2, 2)
plt.plot(epoch, loss, lw=1.5, color='darkorange', label='train loss', marker='.', markevery=1,
                 mew=1)
plt.plot(epoch, loss_1, lw=1.5, color='royalblue', label='train val-loss', marker='.', markevery=1,
                 mew=1)
plt.xlabel("epochs", fontsize=12)
plt.ylabel("loss(%)", fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.show()

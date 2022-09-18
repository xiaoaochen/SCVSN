import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator, figure, plot

y1 = [
    0.9692,
    0.9517,
    0.9473,
    0.9473,
    0.9824,
    0.9649,
    0.97,
    0.9617,
    0.9605,
    0.9561,
    0.9736,
    0.978,
    0.9661,
    0.9605,
    0.96491,
    0.9649,
    0.9605,
    0.9649,
    0.9736,
    0.9736,
    0.9649,
    0.9712,
    0.9692,
    0.9736,
    0.9692,
    0.9561,
    0.9649,
    0.9473,
    0.9649,
    0.9605,
    0.9605,
    0.9675,
    0.9868,
    0.9692,
    0.9736,
    0.9629,
    0.9768,
    0.9675,
    0.9868,
    0.9692,
]
y2 = [
    0.9324,
    0.9393,
    0.9391,
    0.9642,
    0.9739,
    0.9482,
    0.948,
    0.9475,
    0.9399,
    0.9561,
    0.9736,
    0.9653,
    0.9478,
    0.9647,
    0.9565,
    0.9565,
    0.9563,
    0.9482,
    0.9487,
    0.9568,
    0.9482,
    0.9649,
    0.9404,
    0.9406,
    0.9404,
    0.9478,
    0.9482,
    0.9557,
    0.9649,
    0.9733,
    0.9563,
    0.9563,
    0.9414,
    0.9484,
    0.9487,
    0.9641,
    0.9514,
    0.9563,
    0.9414,
    0.9484,
]

y3 = [
    0.9505,
    0.9455,
    0.9432,
    0.9557,
    0.9781,
    0.9565,
    0.961,
    0.9696,
    0.9501,
    0.9561,
    0.9736,
    0.9716,
    0.9519,
    0.9626,
    0.9606,
    0.9606,
    0.9584,
    0.9565,
    0.961,
    0.9652,
    0.9565,
    0.9649,
    0.9546,
    0.9568,
    0.9546,
    0.9519,
    0.9565,
    0.9515,
    0.9649,
    0.9668,
    0.9584,
    0.9684,
    0.9635,
    0.9587,
    0.961,
    0.9534,
    0.9635,
    0.9684,
    0.9635,
    0.9587,
]
max_number = max(y1)
min_number = min(y1)
print(min_number)
print(max_number)
sum_number = 0
for x in y1:
    sum_number = x + sum_number
sum_number = sum_number - max_number - min_number
print(sum_number * 100 / 38)

# print(y3)
y_major_locator = MultipleLocator(5)

line = np.linspace(1, 40, 40)

plt.figure()

ax = plt.gca()
plt.ylabel('Value(%)')
plt.xlabel('Training times')
ax.yaxis.set_major_locator(y_major_locator)
x_0 = 2
y_0 = 0.9435 * 100
x_1 = 12
y_1 = 0.9714 * 100
x_3 = 25
y_3 = 96.01
# plt.scatter(x_0, y_0)
# plt.scatter(x_1, y_1)

# plt.annotate(r'$min:%s$' % y_0,
#              xy=(x_0, y_0),
#              xycoords='data',
#              xytext=(+30, -30),
#              textcoords='offset points',
#              fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
#
# plt.annotate(r'$max:%s$' % y_1,
#              xy=(x_1, y_1),
#              xycoords='data',
#              xytext=(+30, -30),
#              textcoords='offset points',
#              fontsize=10,
#              arrowprops=dict(arrowstyle='->',
#                              connectionstyle="arc3,rad=.2",
#                              color='blue'))
#
# plt.annotate(r'$average:%s$' % y_3,
#              xy=(x_3, y_3),
#              xycoords='data',
#              xytext=(+20, -20),
#              textcoords='offset points',
#              fontsize=10,
#              arrowprops=dict(arrowstyle='->',
#                              connectionstyle="arc3,rad=.0",
#                              color='red'))
plt.ylim(30, 100)
plt.xlim(0, 40)
y1 = [x * 100 for x in y1]
l1, = plt.plot(
    line,
    y1,
)
sum1 = 0
for x in y1:
    sum1 = sum1 + x
print("recall:" , sum1 * 100 / 40)
y2 = [x * 100 for x in y2]
l2, = plt.plot(
    line,
    y2,
)
sum2 = 0
for x in y2:
    sum2 = sum2 + x
print("Precision:", sum2 * 100 / 40)
y3 = [x * 100 for x in y3]
l3, = plt.plot(
    line,
    y3,
)
sum3 = 0
for x in y3:
    sum3 = sum3 + x
print("F1-Score:" , sum3 * 100 / 40)
plt.ylim(80, 100)
plt.legend(handles=[l1, l2, l3],
           labels=[
               'Recall average 96.64%', 'Precision average 95.25%', 'F1-Score average 95.01%',
           ],
           fontsize=11,
           loc='best')
plt.show()
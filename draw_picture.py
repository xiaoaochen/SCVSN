import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator, figure, plot

y1 = [
    0.952,
    0.9435,
    0.9492,
    0.9548,
    0.9548,
    0.959,
    0.9539,
    0.9674,
    0.9659,
    0.9594,
    0.9689,
    0.9714,
    0.9517,
    0.9627,
    0.9605,
    0.9605,
    0.9583,
    0.9561,
    0.9605,
    0.9649,
    0.9561,
    0.9649,
    0.9539,
    0.9561,
    0.9539,
    0.9617,
    0.9661,
    0.9517,
    0.9649,
    0.9671,
    0.9583,
    0.9643,
    0.9627,
    0.9583,
    0.9605,
    0.9639,
    0.9627,
    0.9643,
    0.9627,
    0.9683,
]
y2 = [
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
y3 = []
for x in range(0, 40):
    y3.append(sum_number * 100 / 38)

# print(y3)
y_major_locator = MultipleLocator(5)

x = np.linspace(1, 40, 40)

plt.figure()

ax = plt.gca()
plt.ylabel('Accuracy(%)')
plt.xlabel('Training times')
ax.yaxis.set_major_locator(y_major_locator)
x_0 = 2
y_0 = 0.9435 * 100
x_1 = 12
y_1 = 0.9714 * 100
x_3 = 25
y_3 = 96.01
plt.scatter(x_0, y_0)
plt.scatter(x_1, y_1)

plt.annotate(r'$min:%s$' % y_0,
             xy=(x_0, y_0),
             xycoords='data',
             xytext=(+30, -30),
             textcoords='offset points',
             fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

plt.annotate(r'$max:%s$' % y_1,
             xy=(x_1, y_1),
             xycoords='data',
             xytext=(+30, -30),
             textcoords='offset points',
             fontsize=10,
             arrowprops=dict(arrowstyle='->',
                             connectionstyle="arc3,rad=.2",
                             color='blue'))

plt.annotate(r'$average:%s$' % y_3,
             xy=(x_3, y_3),
             xycoords='data',
             xytext=(+20, -20),
             textcoords='offset points',
             fontsize=10,
             arrowprops=dict(arrowstyle='->',
                             connectionstyle="arc3,rad=.0",
                             color='red'))
plt.ylim(30, 100)
plt.xlim(0, 40)
y1 = [x * 100 for x in y1]
l1, = plt.plot(
    x,
    y1,
)
l2, = plt.plot(
    x,
    y3,
)
y2 = [x * 100 for x in y2]
plt.ylim(50, 100)
plt.legend(handles=[l1, l2],
           labels=[
               'Accuracy on test setting', 'Average accuracy on test setting',
           ],
           loc='best')
plt.show()
from HW2.p3.loadFittingDataP3 import getData
from HW2.linear_regression import LinReg
import matplotlib.pyplot as plt

x, y = getData(ifPlotData=False)
basis = ['poly', 'cosine0', 'cosine1'][0]
nabla = .1
M = 4
reg = LinReg(x, y, basis, nabla)
# reg.calc_opt_w(M)
reg.calc_gd_w(M, delta_thresh=.001, b=0, w_init=None)
w_x, w_y = reg.calc_w_plot(M)
plt.plot(w_x, w_y, c='r')
plt.scatter(x, y)
plt.title(basis + ' curve fit')
plt.show()


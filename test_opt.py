from gr.board import GrBoard
from gr.grq import BoardOptimizer
from gr.params import GrParams
from gr.log import GrLogger

import numpy as np
import cv2
import logging

from skopt.plots import plot_evaluations, plot_convergence

from pathlib import Path
from matplotlib import pyplot as plt

plt.set_cmap("viridis")

##
##npass = 0
##
##def f(x):
##    global npass
##    y = (np.sin(5 * x[0]) * (1 - np.tanh(x[1] ** 2)) *
##            0.05 * 0.1)
##    print(str(npass), ': f(', x[0], ', ', x[1], ') = ', y)
##    npass += 1
##
##    return y

## gp_res = gp_minimize(f, [(-2.0, 2.0)])
## print(gp_res.x)

## gbrt_res = gbrt_minimize(f, [(-2.0, 2.0)])
## print(gbrt_res.x)

## f_res = forest_minimize(f, [Real(-2.0, 2.0, name='x0'), Real(-2.0, 2.0, name='x1')])
## print(f_res.x)

## all_res = [('gp', gp_res), ('gbrt', gbrt_res), ('forest', f_res)]

## _ = plot_convergence(*all_res)
## plt.show()

##if __name__ == '__main__':
##    main()
##    cv2.destroyAllWindows()

qc = BoardOptimizer(board = GrBoard(), debug = True, echo = False)
qc.board.load_image("./img/go_board_2.png", f_process = False)
#print(qc.quality())

qc.log.logger.addHandler(logging.FileHandler("./opt.log","w"))
p_init, p_last = qc.optimize(groups = [1, 2], save = "always", max_pass = 100)

#p_init, p_last = qc.optimize(groups = [1, 2], save = "never", max_pass = 10)


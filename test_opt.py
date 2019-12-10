from gr.board import GrBoard
from gr.grq import BoardQualityChecker
from gr.params import GrParams
from gr.log import GrLogger

import numpy as np
import cv2
import logging

from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space.space import Real, Integer
from skopt.plots import plot_evaluations, plot_convergence
from skopt.utils import use_named_args

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(123)

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

##def main():
##    # Do the check
##    qc = BoardQualityChecker(debug = False)
##    g = Path("./img").glob('*.png')
##    qd = dict()
##
##    for x in g:
##        if x.is_file():
##            print("Processing", str(x))
##            q = qc.quality(str(x))
##            qd[x.name] = (q[0], q[1], qc.board.params)
##
##    # Print results and convert params to data frame
##    df = None
##    print("")
##    print("Quality check results:")
##    for n, k in enumerate(sorted(qd, key = lambda k: qd[k][0])):
##        print("{}: {}". format(k, qd[k][0]))
##        p = dict()
##        p['INDEX'] = n
##        p['NAME'] = k
##        p['QUALITY'] = qd[k][0]
##        p.update(qd[k][2].todict())
##        if df is None:
##            df = pd.DataFrame([p], index=[0])
##        else:
##            df = df.append(p, ignore_index=True)
##

##if __name__ == '__main__':
##    main()
##    cv2.destroyAllWindows()

log = GrLogger()
board = GrBoard()
board.load_image("./img/go_board_1.png", f_process = False)
p_init = board.params.todict()

qc = BoardQualityChecker(board, debug = False)
space = qc.opt_space(board.params.groups[1])

print("\n\nStaring quality {}".format(qc.quality()))
npass = 0


@use_named_args(space)
def objective(**params):
    global npass, board, qc

    npass += 1
    board.params = params
    q, p = qc.quality()
    print("Pass #{}: quality {}, {}".format(npass, q, p))
    for k in params:
        print("\t{} = {}".format(k, params[k]))
    return q

#print(space)

f_res = forest_minimize(objective, space, n_calls = 20)

print("\nResulting parameters")
for n, v in enumerate(f_res.x):
    print("\t{} = {}".format(space[n].name,v))
    board.params[space[n].name] = v

p_res = board.params.todict()

print("\nParameters diff")
for k in p_init:
    if p_init[k] != p_res[k]:
        print("\t{}: {} != {}".format(k, p_init[k], p_res[k]))

print("\nSanity check {}".format(qc.quality()))



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

qc = BoardQualityChecker(board = GrBoard(), debug = True)
qc.board.load_image("./img/go_board_2.png", f_process = False)

p_init = qc.board.params.todict()
space = qc.opt_space(groups = [1,2])

q_init = qc.quality()
print("\n\nQuality on start: {}".format(q_init[0]))
npass = 0


@use_named_args(space)
def objective(**params):
    global npass, qc

    npass += 1
    qc.board.params = params
    q, p = qc.quality()
    print("Pass #{}: quality {}, {}".format(npass, q, p))
    for k in params:
        print("\t{} = {}".format(k, params[k]))
    return q

f_res = gbrt_minimize(objective, space, n_calls = 100, n_jobs = -1)

print("\nResulting parameters")
for n, v in enumerate(f_res.x):
    print("\t{} = {}".format(space[n].name,v))
    qc.board.params[space[n].name] = v

p_res = qc.board.params.todict()

print("\nParameters diff")
for k in p_init:
    if p_init[k] != p_res[k]:
        print("\t{}: {} (prior was {})".format(k, p_res[k], p_init[k]))

q_last = qc.quality()
print("\n==> Quality check: {} (prior was {}), results: {}".format(q_last[0], q_init[0], q_last[1]))

#print("")
#print(log)

if q_init <= q_last:
    print("\nQuality is less than or equal to initial, parameters not updated")
else:
    print("\nUpdating parameters")
    print("Parameters file {} updated".format(qc.board.save_params()))


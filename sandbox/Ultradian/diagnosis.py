import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['text.usetex'] = True

def plotPhi(steps, Phi):
    plt.figure()
    plt.plot(steps, Phi*100, 'o-', color='b')
    plt.xticks(steps)
    plt.xlabel('EnKI Steps')
    plt.ylabel(r"$\|y-G(\theta)\|_{\Gamma}/\|y\|_{\Gamma} (\%)$")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('Phi.pdf')

def plotTheta(steps, mean, var, truth, ylabel, filename):
    plt.figure()
    plt.plot(steps, mean, 'o-', color='b', label='Ensemble mean')
    plt.fill_between(steps, mean -2*var, mean + 2*var,
                     color='gray', alpha=0.2, label=r'$2\sigma$')
    plt.hlines(truth, np.min(steps), np.max(steps), 'r', 'dashed', label='Truth')
    lg = plt.legend(loc=0)
    lg.draw_frame(False)
    plt.xticks(steps)
    plt.xlabel('EnKI Steps')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)

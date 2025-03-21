import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from scipy.optimize import curve_fit

class velocityFitter:
    def __init__(self, velocities, Nbins, funcs, numParams, MLEparams=None, LSQparams=None, density=True):
        self.velocities = velocities
        self.Nbins = Nbins
        self.density = density
        self.numParams = numParams
        self.funcNames = funcs

        self.funcs = [getattr(stats, f) for f in self.funcNames]

        counts, bins = np.histogram(self.velocities, self.Nbins, density=self.density)
        self.counts = counts
        self.bins = (bins[1:] + bins[:-1]) / 2

        if MLEparams is not None:
            self.MLEparams = MLEparams
        else:
            self.MLEparams = []
        
        if LSQparams is not None:
            self.LSQparams = LSQparams
        else:
            self.LSQparams = []
        
    def init_plot(self, log=False):
        fig, ax = plt.subplots(1, 1)
        ax.hist(self.velocities, bins=self.Nbins, density=True)

        ax.vlines(np.mean(self.velocities), 0, self.counts.max(), colors='r', linestyles='dashed', label='Mean')
        ax.vlines(np.median(self.velocities), 0, self.counts.max(), colors='orange', linestyles='dashed', label='Median')
        ax.legend()
        if log:
            ax.set_yscale('log')
        plt.show()
    
    def MLE_fit(self):
        self.MLEparams = []
        for f in self.funcs:
            params = f.fit(self.velocities)
            self.MLEparams.append(params)

    def LSQ_fit(self):
        self.LSQparams = []
        for i, f in enumerate(self.funcs):
            p0 = np.ones(self.numParams[i])
            params, _ = curve_fit(f.pdf, self.bins, self.counts, p0=p0)
            self.LSQparams.append(params)

    def function_comparer(self, method='MLE'):
        def r_squared(ydata, xdata, params, f):
            residuals = ydata- f(xdata, *params)
            ss_res = np.sum(residuals**2)

            ss_tot = np.sum((ydata-np.mean(ydata))**2)

            r_squared = 1 - (ss_res / ss_tot)
            return r_squared

        def residuals(ydata, xdata, params, f):
            return np.sum((ydata - f(xdata, *params))**2)

        def AIC(xdata, params, f):
            k = len(params)
            logLik = np.sum(f(xdata, *params))
            return 2*k - 2*logLik

        x = self.bins
        y = self.counts
        funcNames = self.funcNames

        if method == 'MLE':
            results = self.MLEparams
        elif method == 'LSQ':        
            results = self.LSQparams
        else:
            raise ValueError("Invalid method, must be 'MLE' or 'LSQ'")

        compare_vals = np.zeros((len(results), 3))

        for i, f in enumerate(self.funcs):
            compare_vals[i, 0] = r_squared(y, x, results[i], f.pdf)
            compare_vals[i, 1] = residuals(y, x, results[i], f.pdf)
            compare_vals[i, 2] =  AIC(x, results[i], f.logpdf)
        
        funcNames = funcNames[np.argsort(compare_vals[:, 2])]
        compare_vals = compare_vals[np.argsort(compare_vals[:, 2])]
        print(f'{method} results:')
        for i, [r2, res, AIC_val] in enumerate(compare_vals):
            print(f"{funcNames[i]}: r2 = {r2:.4f}, S = {res:.6f}, AIC = {AIC_val:.3f}")

    def plot_distributions(self, log=False):
        num_funcs = len(self.funcNames)
        rows = np.ceil(num_funcs / 2)  # Determine the number of rows needed

        fig, axs = plt.subplots(rows, 2, figsize=(15, rows * 5))  # Create 2xN grid
        axs = axs.flatten()  # Flatten for easy indexing

        sample_vT = np.linspace(self.velocities.min(), self.velocities.max(), 1000)

        for i, ax in enumerate(axs):
            if i < num_funcs:
                func = self.funcs[i]
                
                ax.plot(sample_vT, func.pdf(sample_vT, *self.LSQparams[i]), label=self.funcNames[i], color='yellow')
                ax.plot(sample_vT, func.pdf(sample_vT, *self.MLEparams[i]), label='MLE ' + self.funcNames[i], color='magenta', lw=3, alpha=0.7)
                ax.hist(self.velocities, bins=self.Nbins, density=False, alpha=0.5, label='Data', color='blue')
                ax.legend(loc='lower right')

                if log:
                    ax.set_yscale('log')
            else:
                ax.axis('off')  # Hide extra subplots if the number of functions is odd

        plt.tight_layout()
        plt.show()
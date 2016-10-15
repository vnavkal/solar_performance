from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf

d = pd.read_csv('data/datasci_takehome.csv', index_col=('month', 'system'))
d['pr'] = d['correctedkwh'] / d['expectedkwh']
def energy_density_estimator(month):
    avg_fraction_of_year_elapsed = (month - .5) / 12
    winter_solstice_fraction_of_year = 21 / 365
    return -np.cos(2 * np.pi * avg_fraction_of_year_elapsed - winter_solstice_fraction_of_year)

d['energydensityestimator'] = energy_density_estimator(d.index.get_level_values('month').values)

def plot_kwh():
    fig, axes = plt.subplots(3, 1, sharex='col')
    d[d['vintage'] == 'A'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[0])
    d[d['vintage'] == 'B'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[1])
    d[d['vintage'] == 'C'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[2])
    plt.show()

vintage_dummies = pd.get_dummies(d['vintage'], drop_first=True)
month_dummies = pd.get_dummies(d.reset_index()['month'], drop_first=True)
d = pd.concat((d, vintage_dummies, month_dummies), axis=1, ignore_index=True)

# model = smf.GLM.from_formula(formula='pr ~ A + B + energydensityestimator', family=sm.families.Poisson(), data=d)
model = smf.GLM.from_formula(formula='pr ~ energydensityestimator', family=sm.families.Poisson(), data=d)
result = model.fit()

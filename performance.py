from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf

d = pd.read_csv('data/datasci_takehome.csv')
d['pr'] = d['correctedkwh'] / d['expectedkwh']
def energy_density_estimator(month):
    avg_fraction_of_year_elapsed = (month - .5) / 12
    winter_solstice_fraction_of_year = 21 / 365
    return -np.cos(2 * np.pi * avg_fraction_of_year_elapsed - winter_solstice_fraction_of_year)

d['energydensityestimator'] = d['month'].apply(energy_density_estimator)

def plot_kwh():
    fig, axes = plt.subplots(3, 1, sharex='col')
    d[d['vintage'] == 'A'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[0])
    d[d['vintage'] == 'B'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[1])
    d[d['vintage'] == 'C'].groupby(level='month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[2])
    plt.show()

def plot_pr():
    pr_means = d.groupby('month').apply(lambda group: group.groupby('vintage')['pr'].mean())
    pr_means.plot()
    plt.show()

# vintage_dummies = pd.get_dummies(d['vintage'], drop_first=True)
# month_dummies = pd.get_dummies(d['month'], drop_first=True)
# d = pd.concat((d, vintage_dummies, month_dummies), axis=1)

model = smf.GLM.from_formula(formula='pr ~ vintage', family=sm.families.Poisson(), data=d)
# model = smf.GLM(d['pr'], d[month_dummies.columns.append(vintage_dummies.columns)], family=sm.families.Poisson())
# model = smf.GLM(d['pr'], d[vintage_dummies.columns])
# model = smf.GLM.from_formula(formula='pr ~ vintage', family=sm.families.Gaussian(), data=d[np.logical_or(d['vintage']=='B', d['vintage']=='C')])
# model = smf.GLM.from_formula(formula='pr ~ vintage', family=sm.families.Gaussian(), data=d[np.logical_or(d['vintage']=='A', d['vintage']=='C')])
model = smf.GLM.from_formula(formula='pr ~ vintage + C(month)', family=sm.families.Gamma(), data=d)
result = model.fit()
print(result.summary())

# estimator = linear_model.LinearRegression().fit(d[vintage_dummies.columns].values, d['pr'].values)

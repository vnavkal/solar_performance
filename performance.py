from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links

d = pd.read_csv('data/datasci_takehome.csv')
d['performance_ratio'] = d['correctedkwh'] / d['expectedkwh']
all_vintages = d['vintage'].unique()

def energy_density_estimator(month):
    avg_fraction_of_year_elapsed = (month - .5) / 12
    winter_solstice_fraction_of_year = 21 / 365
    return -np.cos(2 * np.pi * avg_fraction_of_year_elapsed - winter_solstice_fraction_of_year)

d['energydensityestimator'] = d['month'].apply(energy_density_estimator)

def plot_kwh():
    fig, axes = plt.subplots(3, 1, sharex='col')
    d[d['vintage'] == 'A'].groupby('month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[0])
    d[d['vintage'] == 'B'].groupby('month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[1])
    d[d['vintage'] == 'C'].groupby('month')[['correctedkwh', 'expectedkwh']].mean().plot(ax=axes[2])
    plt.show()

def plot_pr():
    pr_means = d.groupby('month').apply(lambda group: group.groupby('vintage')['performance_ratio'].mean())
    pr_means.plot()
    plt.show()

def plot_month_distributions():
    fig, axes = plt.subplots(3, 1, sharex='col')
    for i, vintage in enumerate(all_vintages):
        d[d['vintage']==vintage]['month'].hist(bins=12, ax=axes[i], align='mid', normed=True)
    axes[i].set_xlim((1, 12))
    plt.show()

def boxplot_pr_by_vintage():
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    vintages = d['vintage'].unique()
    pr_by_vintage = [d[d['vintage']==vintage]['performance_ratio'] for vintage in vintages]
    conf_intervals = [[sem(pr_series) / 100, -sem(pr_series) / 100] for pr_series in pr_by_vintage]
    ax.boxplot(pr_by_vintage, conf_intervals=conf_intervals)
    plt.show()

# Create the boxplot
bp = ax.boxplot(data_to_plot)

# model = smf.GLM.from_formula(formula='performance_ratio ~ vintage + C(month)', family=sm.families.Gaussian(), data=d)
# model = smf.GLM.from_formula(formula='performance_ratio ~ vintage', family=sm.families.Gamma(link=links.identity), data=d)
base_model = smf.GLM.from_formula(formula='performance_ratio ~ C(month)', family=sm.families.Gamma(), data=d)
offset = base_model.fit().mu
model = smf.GLM.from_formula(formula='performance_ratio ~ vintage + C(month)', family=sm.families.Gamma(), data=d)
# model = smf.GLM.from_formula(formula='performance_ratio ~ vintage', family=sm.families.Poisson(link=links.identity), data=d)
result = model.fit()

plt.scatter(x=d['performance_ratio'].values, y=result.resid_response)
plt.show()
print(result.summary())

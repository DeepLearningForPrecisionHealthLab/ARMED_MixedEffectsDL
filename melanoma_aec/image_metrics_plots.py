import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from medl.misc import expand_results_path

strMetricsPath = 'melanoma_aec_tiedweights/base_model_v3/recon_image_metrics.csv'
# strMetricsPath = '../melanoma_autoencoder/image_attributes.csv'

dictDates = {160802: 'Day 1',
             160808: 'Day 2',
             161209: 'Day 3',
             161214: 'Day 4',
             161220: 'Day 5',
             161224: 'Day 6'}

dfMetrics = pd.read_csv(expand_results_path(strMetricsPath), index_col=0)
dfMetrics['Date'] = dfMetrics['Date'].apply(lambda x: dictDates[x])

fig, ax = plt.subplots(4, 1, figsize=(16, 13), gridspec_kw={'hspace': 0.4})

# dictMetricNames = {'Brightness': 'Mean brightness',
#                    'S.D.': 'Contrast (s.d.)',
#                    'VoL': 'Sharpness (variance-of-Laplacian)',
#                    'SNR': 'Signal-to-noise ratio'}
dictMetricNames = {'Brightness': 'Mean brightness',
                   'Contrast': 'Contrast (s.d.)',
                   'Sharpness': 'Sharpness (variance-of-Laplacian)',
                   'SNR': 'Signal-to-noise ratio'}

for i, (strMetric, strAxisLabel) in enumerate(dictMetricNames.items()):
    vmax = dfMetrics[strMetric].mean() + 5 * dfMetrics[strMetric].std()
    vmin = dfMetrics[strMetric].min()
    sns.histplot(data=dfMetrics, x=strMetric, hue='Date', ax=ax[i])
    ax[i].set_xlim(vmin, vmax)
    ax[i].set_xlabel(strAxisLabel)
    
    f, p = f_oneway(*[dfMetrics[strMetric].loc[dfMetrics['Date'] == d].values for d in dictDates.values()])
    print(strAxisLabel, f'F = {f:.03f}, p = {p:.03f}')
    
fig.show()
fig.savefig(expand_results_path(strMetricsPath).replace('csv', 'svg'))
fig.savefig(expand_results_path(strMetricsPath).replace('csv', 'png'))
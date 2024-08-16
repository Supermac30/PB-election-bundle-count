import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

scaling_factor = 1
circlesize = 12
linewidth = 4
width = 16 / scaling_factor
height = 9 / scaling_factor
ticksize = 24
fontsize = 26

def plot(data_path, pruned, normalized):
	with open(data_path, 'rb') as data_file:
	    data = pickle.load(data_file)

	x, y, *removed = zip(*(data[name] for name in data if 'mechanical' not in name))
	print(len(x))

	#plt.scatter(x, y, color='blue', marker='x')
	data_frame = pd.DataFrame({'x': x, 'y': y})

	#plt.xscale('log')
	#plt.yscale('log')


	x, y = np.array(x), np.array(y)

	# slope, intercept, r_value, p_value, std_err = linregress(x, y)
	# line_of_best_fit = intercept + slope * x
	# plt.plot(x, line_of_best_fit, color='red', label='Line of Best Fit')

	if normalized:
		values_to_plot = y / x
		ylabel = '# Bundles / m'
	else:
		ylabel = '# Bundles'
		values_to_plot = y
	print(data_path, max(values_to_plot))
	quantiles = np.percentile(values_to_plot, np.arange(0, 101, 5))
	percentiles = np.arange(0, 101, 5)

	plt.figure(figsize=(width, height))
	plt.plot(percentiles, quantiles, marker='o', linestyle='-', color='b', markersize=circlesize, linewidth=linewidth)

	if normalized:
		plt.yscale('linear')
	else:
		plt.yscale('log', base=10)
	plt.xlabel('Percentile of Instances', fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)
	plt.xticks(fontsize=ticksize)
	plt.yticks(fontsize=ticksize)
	# plt.title('Quantile Plot')


	plt.grid(True)
	# plt.legend()
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
	#ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
	if normalized and pruned:
		name = "quantiles_bundles_normalized_pruned"
	if normalized and not pruned:
		name = "quantiles_bundles_normalized"
	if not normalized and pruned:
		name = "quantiles_bundles_pruned"
	if not normalized and not pruned:
		name = "quantiles_bundles"
	plt.savefig(f"{name}.pdf", format='pdf', dpi=300, bbox_inches='tight')

plot("data_pruned.pkl", True, True)
plot("data_pruned.pkl", True, False)
plot("data.pkl", False, True)
plot("data.pkl", False, False)

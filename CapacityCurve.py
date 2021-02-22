import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats

data = pd.read_csv("Data\lamda{}.csv".format(0.45))
data


def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return (y)

agent = random.randint(0,499)
xdata = data.loc[data['AgentID']==agent, 'mi'].reset_index(drop=True)
xdata = xdata[0:300]
steps = (len(xdata))
ydata = xdata[1:]
xdata = xdata[0:len(xdata)-1]

p0 = [max(ydata), sum(xdata)/len(xdata),0.2] # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')


sns.set(rc={'figure.figsize':(8,6)})
plt.plot(xdata, sigmoid(xdata,*popt),'o',color = 'black', label = 'fit')
plt.scatter(xdata,ydata, marker ='.', alpha = '0.6', label = 'datapoints')
plt.title("Capacity curve of agent: {} for {} steps".format(agent, steps))
plt.xlabel("Money at time t")
plt.ylabel("Money at time t+1")
plt.plot([0,350], [0,350], ls = '--', color = 'red', label = '$45^{o}$ degree line')
plt.legend()
plt.savefig("CapacityCurves\lamda0.45Agent{}_300Steps.png".format(agent))
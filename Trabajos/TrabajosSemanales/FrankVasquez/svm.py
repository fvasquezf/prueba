


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC


x,y = make_blobs(n_samples=50, centers=2, random_state=0,
                 cluster_std=0.60)
plt.scatter(x[:,0],x[:,1], c=y,s=50, cmap='autumn');

xflit = np.linspace(-1,3.5)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
plt.plot([0.6],[2.1],'x',color = 'red', markeredgewidth = 2, markersize = 10)

for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xflit, m *xflit + b, '-k')
plt.xlim(-1,3.5)

xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
 yfit = m * xfit + b
 plt.plot(xfit, yfit, '-k')
 plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA',
 alpha=0.4)
plt.xlim(-1, 3.5);


#### 

model = SVC(kernel='linear', C=1E10)
model.fit(x,y)

def plot_svc_decision_function(model, ax = None,plot_support = True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    
    y,x = np.meshgrid(y,x)
    xy = np.vstack([x.ravel(),y.ravel()]).T
    p = model.decision_function(xy).reshape(x.shape)
    ax.contour(x, y, p, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                   model.support_vectors_[:,1],
                   s = 300, linewidth = 1, facecolors = 'none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(x[:,0], x [:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);               
    
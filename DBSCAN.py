import laspy
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN,OPTICS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fpath='E:/Ender/LAI/data/lidar/non_ground.las'
las=laspy.read(fpath)
points=np.vstack([las.x,las.y,las.z]).transpose()
# clustering=OPTICS(min_samples=5).fit(points)
clustering=HDBSCAN(min_samples=10,cluster_selection_method='leaf').fit(points)
print('num of noise',sum(clustering.labels_==-1))
print('num of clusters',len(set(clustering.labels_)))
import matplotlib
NUM_COLORS = len(set(clustering.labels_))

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = Axes3D(fig)

ax.set_box_aspect((np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])))  # aspect ratio is 1:1:1 in data space

colors = matplotlib.cm.rainbow(np.linspace(0, 1, NUM_COLORS))


# ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
ax.scatter(points[:,0], points[:,1], points[:,2], c=colors[clustering.labels_], s=0.1)
ax.view_init(azim=200)
plt.show()
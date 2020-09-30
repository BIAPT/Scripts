import matplotlib
matplotlib.use('Qt5Agg')
import sys
sys.path.append('../')
from helper_functions import stability_measure
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from helper_functions.General_Information import *

pdf = matplotlib.backends.backend_pdf.PdfPages("SI_SIL_33part_wholebrain_wPLI_30_10_allfrequ.pdf")

# random data
mean = np.mean(X)
std = np.std(X)

data_random= np.random.normal(mean, std, size=X.shape)
Y_ID_random = data['ID']
Y_ID = data['ID']

"""
Stability Index
"""
P=[3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
K=[2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate
Rep=20                               #number of Repetitions (Mean at the end)

[SI_M_rand, SI_SD_rand] = stability_measure.compute_stability_index(data_random, Y_ID_random, P, K, Rep)
[SI_M_Base, SI_SD_Base] = stability_measure.compute_stability_index(X, Y_ID, P, K, Rep)

fig,a =  plt.subplots(2, 2)
plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] , xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10'],
        yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0][0].imshow(np.transpose(SI_M_rand))
a[0][0].set_title('Stability Index Mean: Random')
a[0][0].set_xlabel("")
plt.colorbar(im,ax=a[0,0])

im=a[0][1].imshow(np.transpose(SI_SD_rand))
a[0][1].set_xlabel("")
a[0][1].set_title('Stability Index SD: Random')
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[0,1])

im=a[1][0].imshow(np.transpose(SI_M_Base))
a[1][0].set_title('Stability Index Mean: Baseline')
a[1][0].set_xlabel("")
im.set_clim(0.2,0.4)
plt.colorbar(im,ax=a[1,0])

im=a[1][1].imshow(np.transpose(SI_SD_Base))
a[1][1].set_title('Stability Index SD: Baseline')
a[1][1].set_xlabel("")
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[1,1])

fig.set_figheight(17)
fig.set_figwidth(10)
plt.show()

pdf.savefig(fig)

pd.DataFrame(SI_M_Base).to_pickle('SI_Base_33part_wPLI_30_10_allfr.pickle')
pd.DataFrame(SI_M_rand).to_pickle('SI_rand_33part_wPLI_30_10_allfr.pickle')


"""
Silhouette Score
"""
P=[3, 4, 5, 6, 7, 8, 9, 10]        #number of Principal components to iterate
K=[2, 3, 4, 5, 6, 7, 8, 9, 10]     #number of K-clusters to iterate

SIS_Rand = stability_measure.compute_silhouette_score(data_random, P, K)
SIS_Base = stability_measure.compute_silhouette_score(X, P, K)

fig, a = plt.subplots(1, 2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0].imshow(np.transpose(SIS_Rand),cmap='viridis_r')
a[0].set_title('Silhouette Score  : Random')
a[0].set_xlabel("")
plt.colorbar(im,ax=a[0])

im=a[1].imshow(np.transpose(SIS_Base),cmap='viridis_r')
a[1].set_title('Silhouette Score : Baseline')
a[1].set_xlabel("")
#im.set_clim(0.1,0.45)
plt.colorbar(im,ax=a[1])

fig.set_figheight(3)
fig.set_figwidth(10)
plt.show()
pdf.savefig(fig)
pdf.close()


pd.DataFrame(SIS_Base).to_pickle('SIS_Base_33part_wPLI_30_10_allfr.pickle')
pd.DataFrame(SIS_Rand).to_pickle('SIS_rand_33part_wPLI_30_10_allfr.pickle')

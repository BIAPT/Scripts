import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from import_time_data_mean import *

data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_NEW.pickle')

data12_base=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='base'))[0],:]
data12_anes=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='anes'))[0],:]
data12_reco=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='reco'))[0],:]

data20_base=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='base'))[0],:]
data20_anes=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='anes'))[0],:]
data20_reco=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='reco'))[0],:]

areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO',"Mean"]

plt.subplot(311)
plt.plot(data12_base['MEAN'])
plt.hlines((np.mean(data12_base['MEAN'])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_12 during Baseline')

plt.subplot(312)
plt.plot(np.mean(data12_anes.iloc[:,4:],axis=1))
plt.hlines(np.mean(np.mean(data12_anes.iloc[:,4:])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_12 during Anesthesia')

plt.subplot(313)
plt.plot(np.mean(data12_reco.iloc[:,4:],axis=1))
plt.hlines(np.mean(np.mean(data12_reco.iloc[:,4:])),0,290)
plt.xlabel('timestep')
plt.ylabel('wPLI')
plt.title('WSAS_12 during Recovery')

plt.subplot(211)
plt.plot(data12_base.iloc[:,4:])
#plt.hlines(np.mean(np.mean(data12_anes.iloc[:,4:])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_12 during Baseline')
#plt.legend(areas)

plt.subplot(212)
plt.plot(data12_anes.iloc[:,4:])
#plt.hlines(np.mean(np.mean(data12_anes.iloc[:,4:])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_12 during Anesthesia')
plt.legend(areas)


plt.subplot(131)
plt.imshow(mean_12_base,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS12_Baseline")
plt.subplot(132)
plt.imshow(mean_12_anes,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS12_Anesthesia")
plt.subplot(133)
plt.imshow(mean_12_reco,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS12_Recovery")
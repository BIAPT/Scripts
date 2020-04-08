import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_pickle('time_resolved_wpli_all.pickle')

data12_base=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='Base'))[0],:]
data12_anes=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='Anes'))[0],:]
data12_reco=data.iloc[np.where((data['ID']=='12') & (data['Phase']=='Reco'))[0],:]

data20_base=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='Base'))[0],:]
data20_anes=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='Anes'))[0],:]
data20_reco=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='Reco'))[0],:]

areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO']


plt.subplot(311)
plt.plot(data12_base.iloc[:,5:])
plt.subplot(312)
plt.plot(data12_anes.iloc[:,5:])
plt.subplot(313)
plt.plot(data12_reco.iloc[:,5:])
plt.legend(areas)


plt.plot(np.mean(data12_base.iloc[:,5:],axis=1))
plt.hlines(np.mean(np.mean(data12_base.iloc[:,5:])),0,2850)
plt.xlabel('timestep')
plt.ylabel('wPLI')
plt.title('WSAS_12 during Baseline')
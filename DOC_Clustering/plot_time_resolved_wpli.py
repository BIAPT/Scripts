import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from import_time_data_mean import *

data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_NEW_1320.pickle')

data13_base=data.iloc[np.where((data['ID']=='13') & (data['Phase']=='base'))[0],:]
data13_anes=data.iloc[np.where((data['ID']=='13') & (data['Phase']=='anes'))[0],:]
data13_reco=data.iloc[np.where((data['ID']=='13') & (data['Phase']=='reco'))[0],:]

data20_base=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='base'))[0],:]
data20_anes=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='anes'))[0],:]
data20_reco=data.iloc[np.where((data['ID']=='20') & (data['Phase']=='reco'))[0],:]

areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','FF','CC','PP','TT','OO','MEAN']
part=['13','20']
phases=['base','anes','reco']

full_crossing=pd.DataFrame()
full_means=pd.DataFrame()
full_sds=pd.DataFrame()

for p in part:
    for ph in phases:
        tmp_data = data.iloc[np.where((data['ID'] == p) & (data['Phase'] == ph))[0], :]

        crossing = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        means = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        sds = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        names = ['part', 'phase', 'FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT',
                 'OO', 'MEAN']
        crossing.columns=names
        means.columns=names
        sds.columns=names

        crossing['part']=p
        means['part']=p
        sds['part']=p
        crossing['phase']=ph
        means['phase']=ph
        sds['phase']=ph

        for f in areas:
            means[f]=np.mean(tmp_data[f])
            sds[f]=np.std(tmp_data[f])

            count=0
            for t in range(1,len(tmp_data)):
                time2=tmp_data[f][t]
                time1=tmp_data[f][t-1]
                if (time2 > means[f][0]+sds[f][0] and time1 <= means[f][0]+sds[f][0]):
                    count=count+1
            crossing[f]=count

        full_crossing=full_crossing.append(crossing)
        full_means=full_means.append(means)
        full_sds=full_sds.append(sds)


full_crossing.columns=names
full_means.columns=names
full_sds.columns=names

'''
#################################
###         PLOT MEANS        ###
#################################

'''
participant='20'

part_means=full_means.iloc[np.where(full_means['part']==participant)[0],:]
part_sds=full_sds.iloc[np.where(full_sds['part']==participant)[0],:]

barWidth = 0.3

# set height of bar
bars_B = part_means.iloc[0,2:]
bars_A = part_means.iloc[1,2:]
bars_R = part_means.iloc[2,2:]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,yerr=part_sds.iloc[0,2:],  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(r2, bars_A,yerr=part_sds.iloc[1,2:],  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(r3, bars_R,yerr=part_sds.iloc[2,2:],  width=barWidth,ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('mean wPLI', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
plt.legend()
plt.title('PARTICIPANT   ' + participant)


'''
#################################
###         PLOT Crossings    ###
#################################

'''
participant='13'

part_crossing=full_crossing.iloc[np.where(full_crossing['part']==participant)[0],:]

barWidth = 0.2

# set height of bar
bars_B = part_crossing.iloc[0,2:]
bars_A = part_crossing.iloc[1,2:]
bars_R = part_crossing.iloc[2,2:]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth, label='Baseline')
plt.bar(r2, bars_A,  width=barWidth, label='Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('number timepoints, wPLI crosses Mean + 1SD', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
plt.legend()
plt.title('PARTICIPANT   ' + participant)






#plt.plot(np.concatenate([data20_base['MEAN'],data20_anes['MEAN'],data20_reco['MEAN']]))
for i in range(0,len(data20_base['MEAN'])):
        plt.axvline(x=i,color='blue',alpha=0.1)
plt.axvline(x=len(data20_base['MEAN']),color='black',alpha=0.7,linewidth =5)
for i in range(len(data20_base['MEAN']),len(data20_base['MEAN'])+len(data20_anes['MEAN'])):
        plt.axvline(x=i,color='orange',alpha=0.1,linewidth =2 )
plt.axvline(x=len(data20_base['MEAN'])+len(data20_anes['MEAN']),color='black',alpha=0.7,linewidth =5)
for i in range(len(data20_base['MEAN'])+len(data20_anes['MEAN']),len(data20_base['MEAN'])+len(data20_anes['MEAN'])+len(data20_reco['MEAN'])):
        plt.axvline(x=i,color='green',alpha=0.1)
plt.plot(np.concatenate([data20_base['MEAN'],data20_anes['MEAN'],data20_reco['MEAN']]),color = 'darkslategrey')



plt.plot(data20_base['MEAN'])
plt.plot(data20_anes['MEAN'])
plt.plot(data20_reco['MEAN'])

plt.hlines((np.mean(data20_reco['FF'])),0,290)
plt.hlines((np.mean(data20_reco['FF'])+np.std(data20_reco['PC'])),0,290,linestyles='--')
plt.hlines((np.mean(data20_reco['FF'])-np.std(data20_reco['PC'])),0,290,linestyles='--')
plt.ylabel('wPLI')
plt.title('WSAS_13 Recovery PC')








plt.plot(tmp_data[f])
plt.hlines(means[f][0],0,290)
plt.hlines(means[f][0]+sds[f][0],0,290,linestyles='--')
plt.hlines(means[f][0]-sds[f][0],0,290,linestyles='--')
plt.ylabel('wPLI')
plt.title('WSAS_12 during Baseline')








































plt.subplot(312)
plt.plot(np.mean(data13_anes.iloc[:,4:],axis=1))
plt.hlines(np.mean(np.mean(data13_anes.iloc[:,4:])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_13 during Anesthesia')

plt.subplot(313)
plt.plot(np.mean(data13_reco.iloc[:,4:],axis=1))
plt.hlines(np.mean(np.mean(data13_reco.iloc[:,4:])),0,290)
plt.xlabel('timestep')
plt.ylabel('wPLI')
plt.title('WSAS_13 during Recovery')

plt.subplot(311)
plt.plot(data20_base.iloc[:,4:])
plt.subplot(312)
plt.plot(data20_anes.iloc[:,4:])
plt.legend(areas)
plt.subplot(313)
plt.plot(data20_reco.iloc[:,4:])

##
plt.ylabel('wPLI')
plt.title('WSAS_20 during Baseline')
plt.legend(['FF','CO','PO'])

plt.subplot(212)
plt.plot(data13_anes.iloc[:,4:])
#plt.hlines(np.mean(np.mean(data13_anes.iloc[:,4:])),0,290)
plt.ylabel('wPLI')
plt.title('WSAS_13 during Anesthesia')
plt.legend(areas)


plt.subplot(131)
plt.imshow(mean_13_base,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS13_Baseline")
plt.subplot(132)
plt.imshow(mean_13_anes,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS13_Anesthesia")
plt.subplot(133)
plt.imshow(mean_13_reco,cmap='jet')
plt.colorbar()
#plt.clim(0,0.15)
plt.title("WSAS13_Recovery")

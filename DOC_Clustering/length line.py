import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from import_time_data_mean import *
import seaborn as sns
import matplotlib.backends.backend_pdf
import math

data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_Left_10_1.pickle')
#data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_NEW_1320.pickle')

#data_correct_MEAN
data['MEAN']=(np.mean(data.iloc[:,4:-1],axis=1))

areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','FF','CC','PP','TT','OO','MEAN']
part=['13','20']
phases=['Base','Anes','Reco']


full_length=pd.DataFrame()
full_length_sds=pd.DataFrame()
full_means=pd.DataFrame()
full_sds=pd.DataFrame()

for p in part:
    for ph in phases:
        tmp_data = data.iloc[np.where((data['ID'] == p) & (data['Phase'] == ph))[0], :]

        linelength = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        linelength_SD = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        means = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        sds = pd.DataFrame(np.zeros((1,len(areas) + 2)))
        names = ['part', 'phase', 'FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT',
                 'OO', 'MEAN']
        linelength.columns=names
        linelength_SD.columns=names
        means.columns=names
        sds.columns=names

        linelength['part']=p
        linelength_SD['part']=p
        means['part']=p
        sds['part']=p
        linelength['phase']=ph
        linelength_SD['phase']=ph
        means['phase']=ph
        sds['phase']=ph

        for f in areas:
            means[f]=np.mean(tmp_data[f])
            sds[f]=np.std(tmp_data[f])

            counts=[]
            for t in range(1,len(tmp_data)):
                time1=tmp_data[f][t-1]
                time2=tmp_data[f][t]
                distance = max(time1,time2)-min(time1,time2)
                counts.append(distance)
            linelength[f]=np.mean(counts)
            linelength_SD[f]=np.std(counts)

        full_length=full_length.append(linelength)
        full_means=full_means.append(means)
        full_sds=full_sds.append(sds)
        full_length_sds=full_length_sds.append(linelength_SD)


full_length.columns=names
full_length_sds.columns=names
full_means.columns=names
full_sds.columns=names

'''
#################################
###         PLOT Length        ###
#################################

'''
participant='20'

part_length=full_length.iloc[np.where(full_length['part']==participant)[0],:]
part_length_sds=full_length_sds.iloc[np.where(full_length_sds['part']==participant)[0],:]

barWidth = 0.3

# set height of bar
bars_B = part_length.iloc[0,2:]
bars_A = part_length.iloc[1,2:]
bars_R = part_length.iloc[2,2:]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,yerr=part_length_sds.iloc[0,2:],  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(r2, bars_A,yerr=part_length_sds.iloc[1,2:],  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(r3, bars_R,yerr=part_length_sds.iloc[2,2:],  width=barWidth,ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('wPLI difference between two time steps', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
#plt.legend()
plt.title('PARTICIPANT   ' + participant)
plt.ylim(0,0.12)

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
#plt.legend()
plt.title('PARTICIPANT   ' + participant)
plt.ylim(0,0.40)


'''
#############################################
###         PLOT Normalized Length        ###
#############################################
'''

participant='05'
part_length=full_length.iloc[np.where(full_length['part']==participant)[0],:]
part_max=max(max(part_length.iloc[0,2:]),max(part_length.iloc[1,2:]),max(part_length.iloc[2,2:]))
part_min=min(min(part_length.iloc[0,2:]),min(part_length.iloc[1,2:]),min(part_length.iloc[2,2:]))
part_length_norm=(part_length.iloc[:,2:]-part_min)/(part_max-part_min)

part_means=full_means.iloc[np.where(full_means['part']==participant)[0],:]
part_max=max(max(part_means.iloc[0,2:]),max(part_means.iloc[1,2:]),max(part_means.iloc[2,2:]))
part_min=min(min(part_means.iloc[0,2:]),min(part_means.iloc[1,2:]),min(part_means.iloc[2,2:]))
part_means_norm=(part_means.iloc[:,2:]-part_min)/(part_max-part_min)

barWidth = 0.3

# set height of bar
bars_B = part_length_norm.iloc[0,:]
bars_A = part_length_norm.iloc[1,:]
bars_R = part_length_norm.iloc[2,:]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(r2, bars_A,  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('Region', fontweight='bold')
plt.ylabel('normalized Dynamic', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
#plt.legend()
plt.title('PARTICIPANT   ' + participant)


# set height of bar
bars_B = part_means_norm.iloc[0,:]
bars_A = part_means_norm.iloc[1,:]
bars_R = part_means_norm.iloc[2,:]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure()
# Make the plot
plt.bar(r1, bars_B,  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(r2, bars_A,  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('Normalized wPLI', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
#plt.legend()
plt.title('PARTICIPANT   ' + participant)

'''
#################################
###         PLOT Contrast     ###
#################################
'''

# set height of bar
bars_B = abs(part_means_norm.iloc[0,2:]-part_length_norm.iloc[0,2:])
bars_A = abs(part_means_norm.iloc[1,2:]-part_length_norm.iloc[1,2:])
bars_R = abs(part_means_norm.iloc[2,2:]-part_length_norm.iloc[2,2:])

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(r2, bars_A,  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('Contrast_Mean_Dynamic', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
#plt.legend()
plt.title('PARTICIPANT   ' + participant)



'''
#################################
###   PLOT MEAN Length        ###
#################################
'''

dynamic_13=full_length.iloc[np.where(full_length['part']=='13')[0],2:-1]
dynamic_20=full_length.iloc[np.where(full_length['part']=='20')[0],2:-1]
means_13=full_means.iloc[np.where(full_means['part']=='13')[0],2:-1]
means_20=full_means.iloc[np.where(full_means['part']=='20')[0],2:-1]

barWidth = 0.3
# set height of bar
bars_13 = np.mean(dynamic_13,axis=1)
bars_20 = np.mean(dynamic_20,axis=1)

# Set position of bar on X axis
r1 = np.arange(len(bars_13))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.figure()
plt.bar(r1, bars_13,yerr=np.std(dynamic_13,axis=1),  width=barWidth,color='firebrick',ecolor='lightgrey', label='chronic Patient')
plt.bar(r2, bars_20,yerr=np.std(dynamic_20,axis=1),  width=barWidth,color='lightseagreen',ecolor='lightgrey', label='recovered Patient')

# Add xticks on the middle of the group bars
plt.xlabel('experimental Phase')
plt.ylabel('wPLI difference between two time steps')
plt.xticks([r + 0.5*barWidth for r in range(len(bars_13))], ['Baseline','Anesthesia','Recovery'])
plt.legend(['chronic Patient','Recovered Patient'])
plt.title('Functional Connectivity Dynamic' )



barWidth = 0.3
# set height of bar
bars_13 = np.mean(means_13,axis=1)
bars_20 = np.mean(means_20,axis=1)

# Set position of bar on X axis
r1 = np.arange(len(bars_13))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars_13,yerr=np.std(means_13,axis=1),  width=barWidth,color='firebrick',ecolor='lightgrey', label='chronic Patient')
plt.bar(r2, bars_20,yerr=np.std(means_20,axis=1),  width=barWidth,color='lightseagreen',ecolor='lightgrey', label='recovered Patient')

# Add xticks on the middle of the group bars
plt.xlabel('experimental Phase')
plt.ylabel('wPLI')
plt.xticks([r + 0.5*barWidth for r in range(len(bars_13))], ['Baseline','Anesthesia','Recovery'])
plt.title('Functional Connectivity Mean' )



'''
#################################
###    PLOT BOTH IN 1       ###
#################################

'''

participant='13'

part_length=full_length.iloc[np.where(full_length['part']==participant)[0],:]
part_length_sds=full_length_sds.iloc[np.where(full_length_sds['part']==participant)[0],:]

part_means=full_means.iloc[np.where(full_means['part']==participant)[0],:]
part_sds=full_sds.iloc[np.where(full_sds['part']==participant)[0],:]

barWidth = 0.3

# set height of bar
bars_B = part_means.iloc[0,2:-1]
bars_A = part_means.iloc[1,2:-1]
bars_R = part_means.iloc[2,2:-1]

# Set position of bar on X axis
r1 = np.arange(len(bars_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth,ecolor='lightgrey', label='Mean_Baseline')
plt.bar(r2, bars_A,  width=barWidth,ecolor='lightgrey', label='Mean_Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,ecolor='lightgrey',label='Mean_Recovery')


bars_B = part_length.iloc[0,2:-1]
bars_A = part_length.iloc[1,2:-1]
bars_R = part_length.iloc[2,2:-1]

# Make the plot
plt.bar(r1, bars_B,  width=barWidth,color='darkblue',ecolor='lightgrey', label='Dynamic_Baseline')
plt.bar(r2, bars_A,  width=barWidth,color='indianred',ecolor='lightgrey', label='Dynamic_Anesthesia')
plt.bar(r3, bars_R,  width=barWidth,color='darkgreen',ecolor='lightgrey',label='Dynamic_Recovery')



# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('wPLI', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_A))], areas)
#plt.legend()
plt.title('PARTICIPANT   ' + participant)
plt.ylim(0,0.20)


'''
#################################
###    PLOT BOTH IN 1       ###
#################################

'''

participant='13'

part_length=full_length.iloc[np.where(full_length['part']==participant)[0],:]
part_length_sds=full_length_sds.iloc[np.where(full_length_sds['part']==participant)[0],:]

part_means=full_means.iloc[np.where(full_means['part']==participant)[0],:]
part_sds=full_sds.iloc[np.where(full_sds['part']==participant)[0],:]

barWidth = 0.3

# set height of bar
bars_B = np.mean(part_means.iloc[0,2:-1])
bars_A = np.mean(part_means.iloc[1,2:-1])
bars_R = np.mean(part_means.iloc[2,2:-1])

# Make the plot
plt.bar(1, bars_B,  width=barWidth,ecolor='lightgrey', label='Baseline')
plt.bar(1.4, bars_A,  width=barWidth,ecolor='lightgrey', label='Anesthesia')
plt.bar(1.8, bars_R,  width=barWidth,ecolor='lightgrey',label='Recovery')

bars_B = np.mean(part_length.iloc[0,2:-1])
bars_A = np.mean(part_length.iloc[1,2:-1])
bars_R = np.mean(part_length.iloc[2,2:-1])

# Make the plot
plt.bar(1, bars_B,  width=barWidth,color='darkblue',ecolor='lightgrey', label='Baseline')
plt.bar(1.4, bars_A,  width=barWidth,color='indianred',ecolor='lightgrey', label='Anesthesia')
plt.bar(1.8, bars_R,  width=barWidth,color='darkgreen',ecolor='lightgrey',label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel(' ', fontweight='bold')
plt.ylabel('wPLI', fontweight='bold')
plt.xticks([])
#plt.legend()
plt.title('PARTICIPANT   ' + participant)



'''
#################################
###        CORRELATION        ###
#################################

'''
participant='13'

part_length=full_length.iloc[np.where(full_length['part']==participant)[0],:]
part_mean=full_means.iloc[np.where(full_means['part']==participant)[0],:]

# set height of bar
means_all = np.hstack([part_mean.iloc[0,2:],part_mean.iloc[1,2:],part_mean.iloc[2,2:]])
length_all = np.hstack([part_length.iloc[0,2:],part_length.iloc[1,2:],part_length.iloc[2,2:]])

import seaborn as sns;

ax = sns.regplot(x=list(part_mean.iloc[0,2:]), y=list(part_length.iloc[0,2:]))
plt.xlabel('Connectivity Mean')
plt.ylabel('Connectivity Dynamic')
plt.title('Correlation  Base '+participant)

from scipy.stats import pearsonr
correl=pearsonr(part_mean.iloc[1,2:],part_length.iloc[1,2:])
correl

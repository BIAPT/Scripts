import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from import_time_data_mean import *
import seaborn as sns
import matplotlib.backends.backend_pdf

data=pd.read_pickle('data/WSAS_TIME_DATA_250Hz/wPLI_10_1/final_wpli_all_Left_10_1.pickle')

pdf = matplotlib.backends.backend_pdf.PdfPages("output_pca.pdf")

participants=['02','05','09','10','11','12','13','18','19','20','22']

for participant in participants:
    data_base=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Base'))[0],:]
    data_anes=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Anes'))[0],:]
    data_reco=data.iloc[np.where((data['ID']==participant) & (data['Phase']=='Reco'))[0],:]

    areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','FF','CC','PP','TT','OO','MEAN']

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(data_base.iloc[:,4:])
    X_B = pca.transform(data_base.iloc[:,4:])
    X_A = pca.transform(data_anes.iloc[:,4:])
    X_R = pca.transform(data_reco.iloc[:,4:])

    # average expression over time
    B_mean=np.mean(X_B,axis=0)
    A_mean=np.mean(X_A,axis=0)
    R_mean=np.mean(X_R,axis=0)

    difference=((B_mean+A_mean)/2)-A_mean
    selected=np.where(difference==max(difference))[0][0]

    figure = plt.figure()
    plt.plot(B_mean)
    plt.plot(A_mean)
    plt.plot(R_mean)
    plt.xlabel('Principal components')
    plt.ylabel('time-averaged expression')
    plt.legend(['Baseline', 'Anesthesia','Recovery'])
    plt.title('WSAS' + participant + 'component selected: '+ str(selected))
    pdf.savefig(figure)
    plt.close()
    #

    figure=plt.figure()
    plt.plot(pca.components_[selected])
    plt.xlabel('features')
    plt.ylabel('feature weights')
    plt.xticks(np.arange(0,16),areas)
    plt.title('WSAS' + participant + 'component selected: ' + str(selected))
    pdf.savefig(figure)
    plt.close()

pdf.close()










means_B=[]
means_A=[]
means_R=[]
participants=['05','09','10','11','12','13','18','19','20','22']


for participant in participants:
    data_base = data.iloc[np.where((data['ID'] == participant) & (data['Phase'] == 'Base'))[0], :]
    data_anes = data.iloc[np.where((data['ID'] == participant) & (data['Phase'] == 'Anes'))[0], :]
    data_reco = data.iloc[np.where((data['ID'] == participant) & (data['Phase'] == 'Reco'))[0], :]

    X_B = np.mean(pca.transform(data_base.iloc[:, 4:]))
    X_A = np.mean(pca.transform(data_anes.iloc[:, 4:]))
    X_R = np.mean(pca.transform(data_reco.iloc[:, 4:]))


    means_B.append(X_B)
    means_A.append(X_A)
    means_R.append(X_R)


barWidth = 0.2

# Set position of bar on X axis
r1 = np.arange(len(means_B))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, means_B,  width=barWidth, label='Baseline')
plt.bar(r2, means_A,  width=barWidth, label='Anesthesia')
plt.bar(r3, means_R,  width=barWidth,label='Recovery')

# Add xticks on the middle of the group bars
plt.xlabel('wPLI region', fontweight='bold')
plt.ylabel('number timepoints, wPLI crosses Mean + 1SD', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(means_A))], participants)
plt.legend()
plt.title('PARTICIPANT   ' + participant)




plt.plot(X_B[:,selected])
plt.hlines(np.mean(X_B_20[:,selected])+np.std(X_B_20[:,selected]),1,300)
plt.plot(X_A[:,selected])
plt.plot(X_R[:,selected])


timeseries_B=np.outer(X_B[:, 1],pca.components_[1])
timeseries_B= pd.DataFrame(timeseries_B)
timeseries_B.columns=areas

timeseries_A=np.outer(X_A[:, 1],pca.components_[1])
timeseries_A= pd.DataFrame(timeseries_A)
timeseries_A.columns=areas

plt.plot(timeseries_B['OO'])
plt.plot(timeseries_A['OO'])
plt.plot(X_B[:,1])

plt.plot(data_anes['OO'])
plt.plot(data_base['OO'])

plt.plot(pca.components_[1])
plt.xticks(np.arange(0,16),areas)

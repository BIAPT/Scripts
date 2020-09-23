import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import numpy as np
import matplotlib.backends.backend_pdf
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

healthy_data=pd.read_pickle('data/HEALTHY_Part_WholeBrain_dPLI_10_1_alpha.pickle')
doc_data=pd.read_pickle('data/New_Part_WholeBrain_dPLI_10_1_alpha.pickle')

data=pd.DataFrame(np.row_stack((doc_data,healthy_data)))
data.columns=healthy_data.columns

Phase=['Base']

Part = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15','S16','S17',
        'S18', 'S19', 'S20', 'S22', 'S23',
        'W03', 'W04', 'W08', 'W22', 'W28','W31', 'W34', 'W36',
        'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_heal = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
Part_nonr = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
             'S18', 'S22', 'S23', 'W04', 'W08', 'W28', 'W31', 'W34', 'W36']
Part_reco = ['S02', 'S07', 'S09', 'S19', 'S20', 'W03', 'W22']

pdf = matplotlib.backends.backend_pdf.PdfPages("All_Part_autocorr_dPLI.pdf")
areas = data.columns[4:]

corr = pd.DataFrame(np.zeros((len(Part), 201)))

c=0
for p in Part:
    p
    data_p=data.query("ID == '{}'".format(p))
    data_p_Base=data_p.query("Phase == 'Base'")[areas]

    if Part_reco.__contains__(p):
        corr.loc[c, 'Group'] = "R"
        corr.loc[c, 'ID'] = p
    elif Part_heal.__contains__(p):
        corr.loc[c, 'Group'] = "H"
        corr.loc[c, 'ID'] = p
    elif Part_nonr.__contains__(p):
        corr.loc[c, 'Group'] = "N"
        corr.loc[c, 'ID'] = p

    tmp=[]

    for i in range(len(areas)):
        #tmp.append(np.correlate(data_p[areas[i]], data_p[areas[i]], mode='full')[round(len(data_p) / 2):])
        tmp2 = plt.xcorr(data_p[areas[i]], data_p[areas[i]], usevlines=True,maxlags=200, normed=True)
        tmp.append(np.array(tmp2[1][200:]))

    tmp=pd.DataFrame(tmp)
    corr.iloc[c,0:201] = np.mean(tmp,axis=0)

    fig = plt.figure()
    for i in range(len(areas)):
        plt.xcorr(data_p[areas[i]], data_p[areas[i]], maxlags=200, normed=True, usevlines=0)
        #plt.plot(np.correlate(data_p_Base[areas[i]], data_p_Base[areas[i]],mode='full')[300:])
    plt.xlim(0, 200)
    plt.title('Part_ {}'.format(p))
    # plt.legend(areas, ncol=2)
    pdf.savefig(fig)
    plt.close()
    c += 1


heal = np.mean(corr.iloc[np.where(corr['Group']=='H')[0],:199])
reco = np.mean(corr.iloc[np.where(corr['Group']=='R')[0],:199])
nonr = np.mean(corr.iloc[np.where(corr['Group']=='N')[0],:199])

fig = plt.figure()
plt.plot(heal)
plt.plot(reco)
plt.plot(nonr)
plt.legend(['H','R','N'])
pdf.savefig(fig)
plt.close()

fig = plt.figure()
for i in Part_heal:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'r')
for i in Part_reco:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'b')
for i in Part_nonr:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'g')

pdf.savefig(fig)
plt.close()

pdf.close()
print('done')


import matplotlib
matplotlib.use('Qt5Agg')
import sys
sys.path.append('../')
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from helper_functions.General_Information import *
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

pdf = matplotlib.backends.backend_pdf.PdfPages("Autocorr_33_Part_Prog_{}_10_1_alpha.pdf".format(mode))

areas = data.columns[4:]

corr = pd.DataFrame(np.zeros((len(AllPart['Part']), 201)))

c=0
for p in AllPart['Part']:
    data_p=data.query("ID == '{}'".format(p))
    data_p_Base=data_p.query("Phase == 'Base'")[areas]

    if AllPart['Part_reco'].__contains__(p):
        corr.loc[c, 'Group'] = "Reco"
        corr.loc[c, 'ID'] = p
    elif AllPart['Part_heal'].__contains__(p):
        corr.loc[c, 'Group'] = "Heal"
        corr.loc[c, 'ID'] = p
    elif AllPart['Part_nonr'].__contains__(p):
        corr.loc[c, 'Group'] = "Nonr"
        corr.loc[c, 'ID'] = p
    elif AllPart['Part_ncmd'].__contains__(p):
        corr.loc[c, 'Group'] = "N_cmd"
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


heal = np.mean(corr.iloc[np.where(corr['Group']=='Heal')[0],:199])
reco = np.mean(corr.iloc[np.where(corr['Group']=='Reco')[0],:199])
nonr = np.mean(corr.iloc[np.where(corr['Group']=='Nonr')[0],:199])
ncmd = np.mean(corr.iloc[np.where(corr['Group']=='N_cmd')[0],:199])

fig = plt.figure()
plt.plot(heal)
plt.plot(reco)
plt.plot(nonr)
plt.plot(ncmd)
plt.legend(['Heal','Reco','Nonr','Ncmd'])
pdf.savefig(fig)
plt.close()

fig = plt.figure()
for i in AllPart['Part_heal']:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'r')
for i in AllPart['Part_reco']:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'b')
for i in AllPart['Part_nonr']:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'g')
for i in AllPart['Part_ncmd']:
    tmp = np.array(corr.iloc[np.where(corr['ID'] == i)[0],:199])
    plt.plot(tmp[0],'orange')

pdf.savefig(fig)
plt.close()

pdf.close()
print('done')


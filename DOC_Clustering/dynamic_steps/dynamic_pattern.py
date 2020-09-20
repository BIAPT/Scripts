import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import numpy as np
import matplotlib.backends.backend_pdf
import seaborn as sns

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
Part_reco=['S02', 'S07', 'S09', 'S19', 'S20', 'W03', 'W22']

pdf = matplotlib.backends.backend_pdf.PdfPages("All_Part_wholebrain_dPLI_Pattern.pdf")
areas = data.columns[4:]


def choose_maximum(data):
    max_frame = np.zeros(data.shape)

    for i in range(len(data)):
        maxarea = np.where(data.iloc[i, :] == max(data.iloc[i, :]))[0][0]
        max_frame[i, maxarea] = 1

    return max_frame

def choose_range(data):
    max_frame = np.zeros(data.shape)

    for i in range(len(data)):
        timestep = data.iloc[i, :].copy()

        for a in range(data.shape[1]):
            e = data.shape[1]-a
            maxarea = np.where(timestep == max(timestep))[0][0]
            if max(timestep) >= 0.05:
                max_frame[i, maxarea] = e
            elif max(timestep) < 0.05:
                max_frame[i, maxarea] = 0
            # set maxarea to small value
            timestep[maxarea] = -100


    return max_frame

unique = pd.DataFrame(np.zeros((len(Part), 5 )))
names = ["group","unique_max","unique_range","std_range", "std_raw"]
unique.columns = names

c=0
for p in Part:
    data_p=data.query("ID == '{}'".format(p))
    data_p_Base=data_p.query("Phase == 'Base'")[areas]
    #data_p_Anes=data_p.query("Phase == 'Anes'")[areas]
    #data_p_Reco=data_p.query("Phase == 'Reco'")[areas]

    max_Base = choose_maximum(data_p_Base)
    #max_Anes = choose_maximum(data_p_Anes)

    range_Base = choose_range(data_p_Base)
    #range_Anes = choose_range(data_p_Anes)

    #fig, ax= plt.subplots(2,1)
    #fig.suptitle('Brain_Melody Part: {}'.format(p))
    #ax[0].imshow(np.transpose(max_Base))
    #ax[1].imshow(np.transpose(max_Anes))

    fig = plt.figure(figsize=(10,3))
    plt.title('wPLI_Pattern Part: {}'.format(p))
    plt.imshow(np.transpose(max_Base))
    pdf.savefig(fig)
    plt.close()

    #fig, ax = plt.subplots(2, 1)
    #fig.suptitle('Brain_Melody Part: {}'.format(p))
    #ax[0].imshow(np.transpose(range_Base), cmap='jet')
    #ax[1].imshow(np.transpose(range_Anes), cmap='jet')

    fig = plt.figure(figsize=(10,3))
    plt.title('wPLI_Pattern Part: {}'.format(p))
    plt.imshow(np.transpose(range_Base), cmap='jet')
    plt.colorbar()
    pdf.savefig(fig)
    plt.close()

    unique_max = np.unique(max_Base, axis=0)
    unique_range = np.unique(range_Base, axis=0)

    if Part_reco.__contains__(p) :
        unique.loc[c, 'group'] = "R"
    elif Part_heal.__contains__(p) :
        unique.loc[c, 'group'] = "N"
    elif Part_nonr.__contains__(p) :
        unique.loc[c, 'group'] = "H"

    unique.loc[c, "unique_max"] = 100-(unique_max.shape[0]/ max_Base.shape[0])*100
    unique.loc[c, "unique_range"] = 100-(unique_range.shape[0]/ range_Base.shape[0])*100
    unique.loc[c, "std_range"] = np.mean(np.std(unique_range,axis=0))
    unique.loc[c, "std_raw"] = np.mean(np.std(data_p_Base,axis=0))
    c +=1
    print(str(p) + ' finished')

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="unique_max", y="group", data=unique,
            whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(x="unique_max", y="group", data=unique,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title("unique states in max")
pdf.savefig(f)
plt.close()

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="unique_range", y="group", data=unique,
            whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(x="unique_range", y="group", data=unique,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title("unique states in range")
pdf.savefig(f)
plt.close()


f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="std_range", y="group", data=unique,
            whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(x="std_range", y="group", data=unique,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title("mean std in range")
pdf.savefig(f)
plt.close()

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="std_raw", y="group", data=unique,
            whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(x="std_raw", y="group", data=unique,
              size=4, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
plt.title("mean std raw")
pdf.savefig(f)
plt.close()

pdf.close()
print('done')

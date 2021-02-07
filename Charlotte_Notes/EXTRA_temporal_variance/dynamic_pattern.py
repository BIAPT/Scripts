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

pdf = matplotlib.backends.backend_pdf.PdfPages("Dynamic_33_Part_Prog_{}_10_1_alpha.pdf".format(mode))

areas = data.columns[4:]
X = data[areas]

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


unique = pd.DataFrame()
#unique = pd.DataFrame(np.zeros((len(AllPart['Part']), (5+len(areas)))))
#names = ["group","unique_max","unique_range","std_range", "std_raw"]
#unique.iloc[:,0:5].columns = names
#unique.iloc[:,5:].columns = areas

c=0
for p in AllPart['Part']:
    data_p=data.query("ID == '{}'".format(p))[areas]

    max_data = choose_maximum(data_p)
    range_data = choose_range(data_p)

    fig = plt.figure(figsize=(10,3))
    plt.title('wPLI_Pattern Part: {}'.format(p))
    plt.imshow(np.transpose(max_data))
    pdf.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(10,3))
    plt.title('wPLI_Pattern Part: {}'.format(p))
    plt.imshow(np.transpose(range_data), cmap='jet')
    plt.colorbar()
    pdf.savefig(fig)
    plt.close()

    unique_max = np.unique(max_data, axis=0)
    unique_range = np.unique(range_data, axis=0)

    if AllPart['Part_reco'].__contains__(p) :
        unique.loc[c, 'group'] = "Reco"
    elif AllPart['Part_heal'].__contains__(p) :
        unique.loc[c, 'group'] = "Heal"
    elif AllPart['Part_nonr'].__contains__(p) :
        unique.loc[c, 'group'] = "Nonr"
    elif AllPart['Part_ncmd'].__contains__(p) :
        unique.loc[c, 'group'] = "Ncmd"

    unique.loc[c, "unique_max"] = (unique_max.shape[0]/ max_data.shape[0])*100
    unique.loc[c, "unique_range"] = (unique_range.shape[0]/ range_data.shape[0])*100
    unique.loc[c, "std_range"] = np.mean(np.std(unique_range,axis=0))
    unique.loc[c, "std_raw"] = np.mean(np.std(data_p,axis=0))
    for i in areas:
        unique.loc[c, str(i)] = np.std(data_p,axis=0)[i]
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

"""
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

"""

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

for i in areas:
    f, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x=str(i), y="group", data=unique,
                whis=[0, 100], width=.6, palette="vlag")
    sns.stripplot(x=i, y="group", data=unique,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.title("std raw {}".format(i))
    pdf.savefig(f)
    plt.close()

pdf.close()
print('done')

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np

def plot_two_curves(x1,x2,y1,y2,c1,c2,l1,l2,title,lx,ly):
    """
    :param x1: x Data condition 1
    :param x2: x Data condition 2
    :param y1: y Data condition 1
    :param y2: y Data condition 2
    :param c1: color for signal 1
    :param c2: color for signal 2
    :param l1: label for signal 1
    :param l2: label for signal 2
    :param title: figure title
    :param lx: xlabel
    :param ly: ylabel
    :return: Figure
    """

    # plot results
    f, ax = plt.subplots()
    ax.plot(x1, y1.mean(0), color=c1)
    ax.fill_between(x1, y1.mean(0) - y1.std(0), y1.mean(0) + y1.std(0), color=c1, alpha=.5)
    ax.plot(x2, y2.mean(0), color=c2)
    ax.fill_between(x2, y2.mean(0) - y2.std(0), y2.mean(0) + y2.std(0), color=c2, alpha=.5)
    ax.set(title=title, xlabel=lx,
           ylabel=ly)
    ax.legend([l1, l2])

    return f


def plot_cat_curves(psds, freqs, category, group, title, lx, ly):
    # plot results
    f, ax = plt.subplots()
    for c in np.unique(category):
        y1 = psds.iloc[np.where(category == c)[0],:]
        ax.plot(freqs, y1.mean(0))
        ax.fill_between(freqs, y1.mean(0) - y1.std(0), y1.mean(0) + y1.std(0), alpha=.5)
    ax.set(title=title, xlabel=lx,
           ylabel=ly)
    labels = []
    for c in np.unique(category):
        label = group[np.where(category == c)[0]]
        labels.append(label.iloc[0])
    ax.legend(labels)
    return f

def plot_correlation(data, x, y):
    """
    :param data: DataFrame containing data to plot
    :param x: value on x-axis (string)
    :param y: value on y-axis (string)
    :return: figure
    """
    f = plt.figure()
    corr = pearsonr(data[x], data[y])
    sns.regplot(x=x, y=y, data=data)
    plt.title("r = " + str(corr[0]) + "\n p = " + str(corr[1]))
    return f

def plot_group_correlations(data, start, category, group, pdf):
    """
    :param data: dataframe to plot
    :param start: number of column to start plotting
    :param category: groups to plot, can be outcome, list of int
    :param Names: names corresponding to numerical categories
    :param pdf: pdf to plot images in
    :return: -
    """
    labels = []
    for c in np.unique(data[category]):
        label = group[np.where(data[category] == c)[0]]
        labels.append(label.iloc[0])

    for i in data.columns[start:]:

        plt.figure()
        sns.boxplot(x = category, y = i, data=data)
        sns.stripplot(x = category, y = i, size=4, color=".3", data = data)
        plt.xticks(np.unique(data[category]), labels, rotation = 70)
        plt.title(i)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        data_DOC = data.query("scale != '-'")
        data_DOC['scale'] = data_DOC['scale'].astype(int)
        data_DOC['Age'] = data_DOC['Age'].astype(int)
        data['Age'] = data['Age'].astype(int)

        # plot CRSR or GCS scale correlation
        fig = plt.figure()
        corr = pearsonr(data_DOC["scale"], data_DOC[i])
        sns.regplot(x='scale', y=i, data=data_DOC)
        sns.scatterplot(x='scale', y=i, hue='Group',data=data_DOC)
        plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # plot Age correlation
        fig = plt.figure()
        corr = pearsonr(data["Age"], data[i])
        sns.regplot(x='Age', y=i, data=data)
        sns.scatterplot(x='Age', y=i, hue='Group',data=data)
        plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()



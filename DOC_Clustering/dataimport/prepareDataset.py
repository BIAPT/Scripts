import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
from tqdm import tqdm

def prepare_Dataset(datafile):
    #set names
    areas=['FC','FP','FO','FT','TO','TC','TP','PO','PC','CO','FF','CC','PP','TT','OO']
    names=areas

    data=pd.read_pickle(datafile)
    Y_ID=data.iloc[:,1]

    data_chro=data[(Y_ID == '13') | (Y_ID == '22') | (Y_ID == '10') | (Y_ID == '18')]
    data_reco=data[(Y_ID == '19') | (Y_ID == '20') | (Y_ID == '02') | (Y_ID == '09')]


    data_reco.insert(0, 'outcome', "1")
    data_chro.insert(0, 'outcome', "0")

    data=np.row_stack([data_reco,data_chro])
    data=pd.DataFrame(data)
    X=data.iloc[:,5:]
    X=X.astype(float)
    X.columns=names
    Y_ID=data.iloc[:,2]
    Y_St=data.iloc[:,3]
    Y_time=data.iloc[:,4]
    Y_out=data.iloc[:,0]

    data_Base=data[(Y_St == 'Base')]
    X_Base=data_Base.iloc[:,5:]
    X_Base.columns=names
    X_Base=X_Base.astype(float)
    Y_ID_Base=data_Base.iloc[:,2]
    Y_time_Base=data_Base.iloc[:,4]
    Y_out_Base=data_Base.iloc[:,0]

    data_Anes=data[(Y_St == 'Anes')]
    X_Anes=data_Anes.iloc[:,5:]
    X_Anes.columns=names
    X_Anes=X_Anes.astype(float)
    Y_ID_Anes=data_Anes.iloc[:,2]
    Y_time_Anes=data_Anes.iloc[:,4]
    Y_out_Anes=data_Anes.iloc[:,0]

    data_Reco=data[(Y_St == 'Reco')]
    X_Reco=data_Reco.iloc[:,5:]
    X_Reco.columns=names
    X_Reco=X_Reco.astype(float)
    Y_ID_Reco=data_Reco.iloc[:,2]
    Y_time_Reco=data_Reco.iloc[:,4]
    Y_out_Reco=data_Reco.iloc[:,0]


    # which values do never change
    zerostd=np.where(np.std(X)==0)[0]
    empty=np.zeros(X.shape[1])
    empty[zerostd]=1

    return [X_Base,X_Anes,X_Reco,Y_ID_Base,Y_ID_Anes,Y_ID_Reco,Y_out_Anes,Y_out_Base,Y_out_Reco ]


def calculate_Contrast_Dataset(datafile):
    # set names
    areas = ['FC', 'FP', 'FO', 'FT', 'TO', 'TC', 'TP', 'PO', 'PC', 'CO', 'FF', 'CC', 'PP', 'TT', 'OO']

    data = pd.read_pickle(datafile)
    Y_ID = data.iloc[:, 1]

    names=data.columns
    data_contrast=pd.DataFrame(columns=names)

    Part = ['13', '22', '10', '18', '19', '20', '02', '09']

    for p in tqdm(Part):
        tmp=data.iloc[np.where(data['ID']== p)[0],:]
        tmp_B=tmp.iloc[np.where(tmp['Phase']== 'Base')[0],:]
        tmp_A=tmp.iloc[np.where(tmp['Phase']== 'Anes')[0],:]
        tmp_R=tmp.iloc[np.where(tmp['Phase']== 'Reco')[0],:]

        for b in tqdm(range(len(tmp_B))):
            for a in range(len(tmp_A)):
                c=np.zeros([1,19])
                c=pd.DataFrame(c)
                c.columns=names
                c['Name']=tmp_B['Name'][b]
                c['ID']=tmp_B['ID'][b]
                c['Phase']='Base '+str(b)+'-Anes '+str(a)
                c['Time']=tmp_B['Time'][b]

                #for i in areas:
                #    c[i] = tmp_B[i][b]-tmp_A[i][a]
                c.iloc[0,4:] = tmp_B.iloc[b,4:]-tmp_A.iloc[a,4:]


                data_contrast=data_contrast.append(c)

    data_contrast.to_pickle('contrast_NEW_dPLI_all_10_1_left.pickle')




def prepare_Contrast_Dataset(contrast_datafile):
    data=pd.read_pickle(contrast_datafile)
    Y_ID = data.iloc[:, 1]

    names = data.columns

    data_contrast_chro = data[(Y_ID == '13') | (Y_ID == '22') | (Y_ID == '10') | (Y_ID == '18')]
    data_contrast_reco = data[(Y_ID == '19') | (Y_ID == '20') | (Y_ID == '02') | (Y_ID == '09')]

    data_contrast_reco.insert(0, 'outcome', "1")
    data_contrast_chro.insert(0, 'outcome', "0")

    data = np.row_stack([data_contrast_reco, data_contrast_chro])
    data = pd.DataFrame(data)

    Y_out = data.iloc[:, 0]
    data=data.iloc[:,1:]
    data.columns = names
    X = data.iloc[:, 4:]
    X = X.astype(float)
    Y_ID = data.iloc[:, 1]

    return [X,Y_out,Y_ID]


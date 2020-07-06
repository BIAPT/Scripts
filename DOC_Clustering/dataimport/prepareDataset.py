import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
from tqdm import tqdm



#datafile='data/combined_NEW_wPLI_dPLI_all_10_1_left.pickle'
def prepare_Dataset(datafile):

    #set names

    data=pd.read_pickle(datafile)

    areas = data.columns[4:]
    names = areas

    names_combined = "w_" + pd.DataFrame(areas)
    names_combined = names_combined.append('d_' + pd.DataFrame(areas))

    Y_ID=data.iloc[:,1]

    data_chro=data[(Y_ID == '13') | (Y_ID == '22') | (Y_ID == '10') | (Y_ID == '18')]
    data_reco=data[(Y_ID == '19') | (Y_ID == '20') | (Y_ID == '02') | (Y_ID == '09')]

    data_reco.insert(0, 'outcome', "1")
    data_chro.insert(0, 'outcome', "0")

    data=np.row_stack([data_reco,data_chro])
    data=pd.DataFrame(data)
    X=data.iloc[:,5:]
    X=X.astype(float)
    try:
        X.columns=names
    except:
        X.columns = names_combined
    Y_ID=data.iloc[:,2]
    Y_St=data.iloc[:,3]
    Y_time=data.iloc[:,4]
    Y_out=data.iloc[:,0]

    data_Base=data[(Y_St == 'Base')]
    X_Base=data_Base.iloc[:,5:]
    try:
        X_Base.columns = names
    except:
        X_Base.columns = names_combined
    X_Base=X_Base.astype(float)
    Y_ID_Base=data_Base.iloc[:,2]
    Y_time_Base=data_Base.iloc[:,4]
    Y_out_Base=data_Base.iloc[:,0]

    data_Anes=data[(Y_St == 'Anes')]
    X_Anes=data_Anes.iloc[:,5:]
    try:
        X_Anes.columns = names
    except:
        X_Anes.columns = names_combined
    X_Anes=X_Anes.astype(float)
    Y_ID_Anes=data_Anes.iloc[:,2]
    Y_time_Anes=data_Anes.iloc[:,4]
    Y_out_Anes=data_Anes.iloc[:,0]

    data_Reco=data[(Y_St == 'Reco')]
    X_Reco=data_Reco.iloc[:,5:]
    try:
        X_Reco.columns = names
    except:
        X_Reco.columns = names_combined
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

#wPLIdata='data/NEW_dPLI_all_10_1_left_alpha.pickle'
#dPLIdata='data/NEW_wPLI_all_10_1_left_theta.pickle'


def prepare_Combined_Dataset(wPLIdata,dPLIdata):
    data_d = pd.read_pickle(dPLIdata)
    data_w = pd.read_pickle(wPLIdata)

    Part = ['13', '22', '10', '18','19', '20', '02', '09']

    data_combined=pd.DataFrame()

    for p in  Part:
        X_Base = pd.DataFrame()
        X_Anes = pd.DataFrame()
        X_Reco = pd.DataFrame()

        tmp_w = data_w[(data_w['ID'] == p)]
        tmp_d = data_d[(data_d['ID'] == p)]

        tmp_wb = tmp_w[(tmp_w["Phase"] =='Base')]
        tmp_db = tmp_d[(tmp_d["Phase"] =='Base')]

        tmp_b = pd.DataFrame(np.column_stack([tmp_wb, tmp_db.iloc[:,4:]]))
        X_Base = X_Base.append(tmp_b)

        tmp_wa = tmp_w[(tmp_w["Phase"] == 'Anes')]
        tmp_da = tmp_d[(tmp_d["Phase"] == 'Anes')]

        tmp_a = pd.DataFrame(np.column_stack([tmp_wa, tmp_da.iloc[:,4:]]))
        X_Anes = X_Anes.append(tmp_a)

        tmp_wr = tmp_w[(tmp_w["Phase"] == 'Reco')]
        tmp_dr = tmp_d[(tmp_d["Phase"] == 'Reco')]

        tmp_r = pd.DataFrame(np.column_stack([tmp_wr, tmp_dr.iloc[:,4:]]))
        X_Reco = X_Reco.append(tmp_r)

        tmp_data=X_Base.append(X_Anes.append(X_Reco))
        data_combined=data_combined.append(tmp_data)

    names=data_w.columns[:4]
    names = names.append('w_' + data_w.columns[4:])
    names = names.append('d_' + data_w.columns[4:])

    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #scaler.fit(data_combined.iloc[:,4:])
    #data_combined.iloc[:,4:] = pd.DataFrame(scaler.transform(data_combined.iloc[:,4:]))

    data_combined.columns=names

    data_combined.to_pickle('combined_norm_NEW_dPLI_all_10_1_left.pickle')


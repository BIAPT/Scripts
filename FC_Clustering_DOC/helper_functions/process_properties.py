import pandas as pd
import numpy as np

def calculate_occurence(AllPart,k,P_kmc,data):
    occurence = pd.DataFrame(np.zeros((len(AllPart["Part"]), k+1)))

    # name the columns of the dataframe
    names=["group"]
    for i in range(k):
        names.append(str(i))
    occurence.columns = names

    # compute the time spent in one phase
    for s in range(k):
        c = 0
        for t in AllPart["Part_reco"]:
            occurence.loc[c,'group'] = "Reco"
            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

        for t in AllPart["Part_nonr"]:
            occurence.loc[c,'group'] = "Nonr"
            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

        for t in AllPart["Part_ncmd"]:
            occurence.loc[c,'group'] = "NCMD"
            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

        for t in AllPart["Part_heal"]:
            occurence.loc[c,'group'] = "Heal"
            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

    return occurence

def calculate_dynamics(AllPart, P_kmc, data):
    dynamic = pd.DataFrame(np.zeros((len(AllPart["Part"]), 2)))
    names = ["group","p_switch"]
    dynamic.columns=names
    c=0

    for t in AllPart["Part"]:
        if  np.isin(t,AllPart["Part_reco"]):
            dynamic.loc[c, 'group'] = "Reco"

        elif np.isin(t,AllPart["Part_nonr"]):
            dynamic.loc[c, 'group'] = "Nonr"

        elif np.isin(t,AllPart["Part_ncmd"]):
            dynamic.loc[c, 'group'] = "NCMD"

        elif np.isin(t,AllPart["Part_heal"]):
            dynamic.loc[c, 'group'] = "Heal"

        part_cluster = P_kmc[data['ID'] == t]
        switch = len(np.where(np.diff(part_cluster)!=0)[0])/len(part_cluster)
        switch = switch*100

        dynamic.loc[c, "p_switch"] = switch
        c += 1
    return dynamic

def calculate_dwell_time(AllPart, P_kmc, data,k):
    dwelltime = pd.DataFrame(np.zeros((len(AllPart["Part"]), k+1)))

    # name the columns of the dataframe
    names=["group"]
    for i in range(k):
        names.append(str(i))
    dwelltime.columns=names

    c=0
    for t in AllPart["Part"]:
        if  np.isin(t,AllPart["Part_reco"]):
            dwelltime.loc[c, 'group'] = "Reco"

        elif np.isin(t,AllPart["Part_nonr"]):
            dwelltime.loc[c, 'group'] = "Nonr"

        elif np.isin(t,AllPart["Part_ncmd"]):
            dwelltime.loc[c, 'group'] = "NCMD"

        elif np.isin(t,AllPart["Part_heal"]):
            dwelltime.loc[c, 'group'] = "Heal"

        part_cluster = P_kmc[data['ID'] == t]

        # compute the time spent in one phase
        for s in range(k):
            staytime = []
            tmp=0
            for l in range(1, len(part_cluster) - 1):
                if part_cluster[l] == s and part_cluster[l-1] != s:
                    tmp += 1
                if part_cluster[l] == s and part_cluster[l-1] == s:
                    tmp += 1
                elif part_cluster[l] != s and part_cluster[l-1] == s:
                    if tmp > 0:
                        staytime.append(tmp)
                        tmp = 0

            if len(staytime) == 0:
                dwelltime.loc[c,str(s)] = 0
            else:
                dwelltime.loc[c,str(s)] = np.mean(staytime)

        c += 1
    return dwelltime

def get_transition_matrix(states,n_states):
    n = n_states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(states, states[1:]):
        M[i-1][j-1] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

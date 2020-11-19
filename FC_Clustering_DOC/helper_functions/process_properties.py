import pandas as pd
import numpy as np

def calculate_occurence(AllPart,k,P_kmc,data, partnames, groupnames):
    occurence = pd.DataFrame(np.zeros((len(AllPart["Part"]), k+2)))

    # name the columns of the dataframe
    names=["group", "ID"]
    for i in range(k):
        names.append(str(i))
    occurence.columns = names

    # compute the time spent in one phase
    for s in range(k):
        c = 0
        for t in AllPart["Part"]:
            occurence.loc[c, 'ID'] = t

            if np.isin(t, AllPart[partnames[0]]):
                occurence.loc[c, 'group'] = groupnames[0]

            elif np.isin(t, AllPart[partnames[1]]):
                occurence.loc[c, 'group'] = groupnames[1]

            elif np.isin(t, AllPart[partnames[2]]):
                occurence.loc[c, 'group'] = groupnames[2]

            elif np.isin(t, AllPart[partnames[3]]):
                occurence.loc[c, 'group'] = groupnames[3]

            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

    return occurence

def calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames):
    dynamic = pd.DataFrame(np.zeros((len(AllPart["Part"]), 3)))
    names = ["ID", "group","p_switch"]
    dynamic.columns=names
    c=0

    for t in AllPart["Part"]:
        dynamic.loc[c, 'ID'] = t

        if  np.isin(t,AllPart[partnames[0]]):
            dynamic.loc[c, 'group'] = groupnames[0]

        elif np.isin(t,AllPart[partnames[1]]):
            dynamic.loc[c, 'group'] = groupnames[1]

        elif np.isin(t,AllPart[partnames[2]]):
            dynamic.loc[c, 'group'] = groupnames[2]

        elif np.isin(t,AllPart[partnames[3]]):
            dynamic.loc[c, 'group'] = groupnames[3]

        part_cluster = P_kmc[data['ID'] == t]
        switch = len(np.where(np.diff(part_cluster) != 0)[0])/len(part_cluster)
        switch = switch*100

        dynamic.loc[c, "p_switch"] = switch
        c += 1
    return dynamic

def calculate_dwell_time(AllPart, P_kmc, data,k, partnames, groupnames):
    dwelltime = pd.DataFrame(np.zeros((len(AllPart["Part"]), k+2)))

    # name the columns of the dataframe
    names=["group","ID"]
    for i in range(k):
        names.append(str(i))
    dwelltime.columns=names

    c=0
    for t in AllPart["Part"]:
        dwelltime.loc[c, 'ID'] = t
        if  np.isin(t,AllPart[partnames[0]]):
            dwelltime.loc[c, 'group'] = groupnames[0]

        elif np.isin(t,AllPart[partnames[1]]):
            dwelltime.loc[c, 'group'] = groupnames[1]

        elif np.isin(t,AllPart[partnames[2]]):
            dwelltime.loc[c, 'group'] = groupnames[2]

        elif np.isin(t,AllPart[partnames[3]]):
            dwelltime.loc[c, 'group'] = groupnames[3]

        part_cluster = P_kmc[data['ID'] == t]

        # compute the time spent in one phase
        for s in range(k):
            staytime = []
            tmp=0
            for l in range(2, len(part_cluster)-1):
                if l==1 and part_cluster[1] == s:
                    tmp +=1
                if part_cluster[l] == s and part_cluster[l-1] != s:
                    tmp += 1
                if part_cluster[l] == s and part_cluster[l-1] == s:
                    tmp += 1
                if part_cluster[l] != s and part_cluster[l-1] == s:
                    if tmp > 0:
                        staytime.append(tmp)
                        tmp = 0
                if l == len(part_cluster)-2 and part_cluster[l] == s:
                    if tmp > 0:
                        staytime.append(tmp)

            if len(staytime) == 0:
                dwelltime.loc[c,str(s)] = 0
            else:
                # averaged staytime from all visits divided by length of recording
                dwelltime.loc[c,str(s)] = np.mean(staytime)/len(part_cluster)

        c += 1
    return dwelltime

def get_transition_matrix(states,n_states):
    n = n_states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(states, states[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    M_per = np.array(M)/sum(sum(np.array(M)))

    return M_per

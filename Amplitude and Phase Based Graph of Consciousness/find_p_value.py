# Yacine Mahdid 2019-08-13
# The goal of this script is to find at which p values two dataset are significantly different

import pickle
from math import floor


def load_data(path):
    pickle_in = open(path,"rb")
    data = pickle.load(pickle_in)    
    return data



path_aec = "aec_pre_roc"
path_wpli = "wpli_pre_roc"

aec = load_data(path_aec)
wpli = load_data(path_wpli)


num_bootstrap = 5000
stop_p_value = 100
increment = 1
divisor = 1000

for value in range(increment, stop_p_value, increment):
    p = value / divisor 
    percentile = (p / 2)*100
    lb_index = floor((num_bootstrap/100)*(percentile))
    ub_index = floor((num_bootstrap/100)*(100-percentile))
    
    print("Lower bound = " + str(lb_index) + " Upper bound = " + str(ub_index))

    bound_aec_unconscious = (aec[lb_index], aec[ub_index])
    bound_wpli_unconscious = (wpli[lb_index], wpli[ub_index])

    if(wpli[lb_index] > aec[ub_index]):
        print("P  = " + str(p))
        break
    else:
        print("P > " + str(p))

lb_index = floor((num_bootstrap/100)*(2.5))
ub_index = floor((num_bootstrap/100)*(97.5))
conf_interval_aec = (aec[lb_index],aec[ub_index])
print("AEC")
print(conf_interval_aec)
conf_interval_pli = (wpli[lb_index],wpli[ub_index])
print("wPLI")
print(conf_interval_pli)
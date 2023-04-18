"""script with any user-defined functions or classes that are external to 'CoSMoR'.
A common use-case for this script would be to store functions required for creating alloy features.
"""

# import libraries
import numpy as np
import pandas as pd
import re
import os
from core_functions import get_comp_dict

SEP = os.sep

# load element database
data_dir = f"helper_data{SEP}"
db_element = pd.read_csv(data_dir + "db_element.csv",encoding="latin-1")
db_element = db_element.set_index('Symbol') #set 'Symbol' column as index



# -----------------------------------------------
def get_comp_avg(alloy_name_list, feat_key):
    """creates composition-weighted average property for alloys

    Args:
        alloy_name_list (list): list of alloy names as string
        el_prop (str): element property to use

    Returns:
        feat_val_list (list): list of composition-weighted average feature values
    """
    
    feat_val_list = []
    
    for alloy in alloy_name_list:
        dict_alloy_comp = get_comp_dict(alloy)
        el_list = dict_alloy_comp["el_list"]
        el_prop_list = [db_element.loc[el][feat_key] for el in el_list]
        c_el_list = dict_alloy_comp["el_at_frac_list"]
        
        avg_feat = 0
        
        for (c_el, el_prop) in zip(c_el_list, el_prop_list):
            avg_feat += (c_el * el_prop)
            
        feat_val_list.append(avg_feat)
        
    return (feat_val_list)



# -----------------------------------------------
def get_asymmetry(alloy_name_list, feat_key):
    """creates asymmetry properties for alloys

    Args:
        alloy_name_list (list): list of alloy names as string
        el_prop (str): element property to use

    Returns:
        feat_val_list (list): list of asymmetry feature values
    """
    
    feat_val_list = []
    avg_feat_list = get_comp_avg(alloy_name_list, feat_key)
    
    for (alloy, avg_feat) in zip(alloy_name_list, avg_feat_list):
        dict_alloy_comp = get_comp_dict(alloy)
        el_list = dict_alloy_comp["el_list"]
        
        el_prop_list = [db_element.loc[el][feat_key] for el in el_list]
        
        c_el_list = dict_alloy_comp["el_at_frac_list"]

        asymm_feat_sum = 0  # summation term inside the square root

        for (c_el, el_prop) in zip(c_el_list, el_prop_list):
            asymm_feat_sum += (c_el * (1-(el_prop/avg_feat))**2)

        asymm_feat = (asymm_feat_sum)**(0.5)
        
        feat_val_list.append(asymm_feat)
    
    return (feat_val_list)
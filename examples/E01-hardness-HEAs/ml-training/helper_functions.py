# import libraries
import numpy as np
import pandas as pd
import re
import os

SEP = os.sep

# load element database
data_dir = f"database_raw{SEP}"
db_element = pd.read_csv(data_dir + "db_element.csv",encoding="latin-1")
db_element = db_element.set_index('Symbol') #set 'Symbol' column as index


# -----------------------------------------------
def get_comp_dict(alloy_name):
    """generates alloy composition from an 'alloy_name' string
    - This function also present *by default) in the 'core_functions' of cosmor.

    Args:
        alloy_name (str): alloy name string (e.g: 'AlCu2', 'CoCr2FeNi1.5')

    Returns:
        dict: dictionary with 2 keys:
                'el_list' - list of elements present
                'el_at_frac_list' - list of atomic fraction of each element
    """

    exclude = set("() {}[]''/,;:?\|~`@#$%&^*-_")
    alloy_name = ''.join(ch for ch in alloy_name if ch not in exclude)

    # split string wherever capital letter is found
    el_stoich_pairs_list = re.findall("[A-Z][^A-Z]*", alloy_name)
    el_list = []
    el_stoich_list = []

    # from each 'el_stoich_pair' extract element name and stoichiometry
    for el_stoich_pair in el_stoich_pairs_list:
        el = "".join(ch for ch in el_stoich_pair if (not ch.isdigit() and ch != "."))
        stoich = "".join(ch for ch in el_stoich_pair if (ch.isdigit() or ch == "."))
        if stoich == "":
            stoich = float(1)
        el_list.append(el)
        el_stoich_list.append(float(stoich))

    # creating atomic fractions
    stoich_total = np.sum(el_stoich_list)
    el_at_frac_list = list(np.around(np.array(el_stoich_list)/stoich_total, 4))

    # sort with elements arranged in alphabetical order
    tuples = zip(*sorted(zip(el_list, el_at_frac_list)))
    el_list_sort, el_at_frac_list_sort = [list(tuple) for tuple in tuples]

    dict_alloy_comp = {
        "el_list": el_list_sort,
        "el_at_frac_list": el_at_frac_list_sort
    }

    return (dict_alloy_comp)



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
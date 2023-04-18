"""
-This script contains core functions required for running the CoSMoR framework.
-These functions are directly called within the 'run_cosmor' method associated with 'cosmor' class.
-Two of these functions ('create_features' and 'make_predictions') are user-defined i.e., for each 
problem these would have to be created by the user; whereas the rest of the functions will remain as default.

List of functions:
- get_comp_dict : [DEFAULT] generates alloy composition from an 'alloy_name' string
- create_alloys : [DEFAULT] generates compositions for the compositional pathways to be probed
- create_features : [USER-DEFINED] creates alloy features that act as input for ML model
- make_predictions : [USER-DEFINED] creates predictions using pre-trained machine learning models
- calculate_dY_dX : [DEFAULT] calculates gradient of target property wrt each feature at each alloy compostion 
                    along the composition pathway defined
- calculate_delta_X : [DEFAULT] calculates local changes in feature values for each composition step
- calculate_feat_contributions : [DEFAULT] calculates local and cumulative feature contributions along the composition pathway
"""

# import libraries needed for CoSMoR
import numpy as np
import pandas as pd
import re
import os

# import case-specific libraries
import pickle

    
SEP = os.sep


# -----------------------------------------------
def get_comp_dict(alloy_name):
    """generates alloy composition from an 'alloy_name' string

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
def create_alloys(comp_A, comp_B, dc = 0.01, cAmin = 0.0, cAmax = 1.0):
    """generates compositions along a continuous composition pathway for an alloy system defined by two components

    Args:
        comp_A (str): component A whose composition will be controlled (e.g. 'Al' or 'Ti2Cr' or 'CoCr2Fe')
        comp_B (str): component B (e.g. 'Al' or 'Ti2Cr' or 'CoCr2Fe')
        dc (float, optional): composition step size. Defaults to 0.01.
        cAmin (float, optional): starting concentration of component A. Defaults to 0.0.
        cAmax (float, optional): maximum concentration of component A. Defaults to 1.0.

    Returns:
        pandas dataframe: alloy compositions at dicrete points along the composition pathway
    """
    
    A_comp_dict = get_comp_dict(comp_A)
    B_comp_dict = get_comp_dict(comp_B)

    # Creating composition array (col 0 and 1 contain atomic fraction of component A and B)
    xA = cAmin
    xB = np.around((1 - xA), 3)
    comp_array = np.zeros(shape = (1, 2))
    comp_array[0, 0] = np.abs(xA)
    comp_array[0, 1] = np.abs(xB)

    while ((xA + dc) <= (cAmax + dc / 10) and
            (xA + dc) <= (1 + dc / 10)):
        
        xA = np.abs(np.around((xA + dc), 3))    # xA increased by dc in every loop
        xB = np.abs(np.around((1 - xA), 3))
        comp_row = np.array([[xA, xB]])
        comp_array = np.vstack((comp_array, comp_row))
    
    # Creating alloy name for each composition
    alloy_name_list = []
    all_el_list = A_comp_dict["el_list"] + B_comp_dict["el_list"]
    x_el_dict = {el:[] for el in all_el_list}
    
    for (i, comp) in zip(range(len(comp_array)), comp_array):
        alloy_name = ""
        
        for (el, stoich) in zip(A_comp_dict["el_list"], A_comp_dict["el_at_frac_list"]):
            el_at_frac = round(comp[0]*stoich, 4)
            alloy_name += str(el) + str(el_at_frac)
            x_el_dict[el].append(el_at_frac)
        
        for (el, stoich) in zip(B_comp_dict["el_list"], B_comp_dict["el_at_frac_list"]):
            el_at_frac = round(comp[1]*stoich, 4)
            alloy_name += str(el) + str(el_at_frac)
            x_el_dict[el].append(el_at_frac)

        alloy_name_list.append(alloy_name)

    # Creating pandas dataframe with composition details
    df_alloys = pd.DataFrame()
    df_alloys["alloy_name"] = np.array(alloy_name_list)
    df_alloys["x [A=%s]"%(comp_A)] = comp_array[:, 0]
    df_alloys["x [B=%s]"%(comp_B)] = comp_array[:, 1]
    
    for el in all_el_list:
        df_alloys["x [%s]"%(el)] = np.array(x_el_dict[el])


    return (df_alloys)



# -----------------------------------------------
def create_features(df_alloys):
    """creates alloy features that are required as an input for machine learning model

    Args:
        df_alloys (pandas dataframe): alloy compositions

    Returns:
        df_x_feats (pandas dataframe): alloy features for all compositions along composition pathway
    """
    
    # Write your own code here that takes df_alloys (compositions generated by 'create_alloys' function) as input
    # and returns a pandas dataframe containing features used by machine learning model as input 
    
    from helper_functions import get_comp_avg, get_asymmetry
    
    alloy_name_list = df_alloys["alloy_name"]
    df_feats = pd.DataFrame()
    
    # create asymmetry features
    df_feats["r_asymm"] = get_asymmetry(alloy_name_list, feat_key="r")
    df_feats["EN_Pauling_asymm"] = get_asymmetry(alloy_name_list, feat_key="EN_Pauling")
    df_feats["E_GPa_asymm"] = get_asymmetry(alloy_name_list, feat_key="E_GPa")
    df_feats["G_GPa_asymm"] = get_asymmetry(alloy_name_list, feat_key="G_GPa")
    df_feats["K_GPa_asymm"] = get_asymmetry(alloy_name_list, feat_key="K_GPa")
    
    # create composition-weighted average features
    df_feats["VEC_avg"] = get_comp_avg(alloy_name_list, feat_key="VEC")
    df_feats["Tm_avg"] = get_comp_avg(alloy_name_list, feat_key="Tm")
    df_feats["Coh_E_avg"] = get_comp_avg(alloy_name_list, feat_key="Coh_E")
    df_feats["density_avg"] = get_comp_avg(alloy_name_list, feat_key="density")
    
    xmin = pd.read_csv(f"helper_data{SEP}xmin.csv")
    xmax = pd.read_csv(f"helper_data{SEP}xmax.csv")
    xmin = np.array(xmin["0"])
    xmax = np.array(xmax["0"])
    df_x_feats = (df_feats - xmin) / (xmax - xmin)
    
    
    return (df_x_feats)



# -----------------------------------------------
def make_predictions(df_x_feats):
    """create predictions using pre-trained machine learning models

    Args:
        df_x_feats (pandas dataframe): alloy features that act as input for pre-trained machine learning model

    Returns:
        Y_pred_array (numpy array): prediction for each alloy
    """

    # Write your own code here that takes df_x_feats (features generated by 'create_features' function) as input
    # and returns a numpy array containing results predicted by the machine learning model 
    
    model_dir = f"trained_models{SEP}"
    model_list = os.listdir(model_dir)
    ensemble_size = len(model_list)
    
    array_x_feats = np.array(df_x_feats) # model was trained on numpy arrays
    
    Y_total = np.zeros(shape=(len(array_x_feats), ))
    for mod in sorted(model_list):
        
        model = pickle.load(open(f"{model_dir}{mod}", "rb"))
        Y_pred = np.array(model.predict(array_x_feats))
        Y_total += Y_pred
    
    Y_pred_array = Y_total/ensemble_size
    #Y_pred_array = Y_pred_array.reshape((len(Y_pred_array), ))
    
    return (Y_pred_array)



# -----------------------------------------------
def calculate_dY_dX(df_x_feats, dX):
    """calculates gradient of target property wrt each feature at each alloy compostion along the composition pathway defined

    Args:
        df_x_feats (pandas dataframe): alloy features for all compositions along composition pathway
        dX (float): small independent change in feature value to be used for calculating gradient

    Returns:
        dY_dX (pandas dataframe): gradient of target property wrt each feature at every alloy compositions
    """
        
    feat_name_list = df_x_feats.columns.to_list()
    dY_dX = pd.DataFrame(columns=feat_name_list)
    x_feats1 = df_x_feats.copy()
    Y1_pred = make_predictions(x_feats1)
    
    for feat in feat_name_list:
        print("\tRunning for feature - '%s' ..."%(feat), end=" ", flush=True)
        x_feats2 = x_feats1.copy()
        if (np.max(x_feats2[feat]) + dX) <= 1:
            x_feats2[feat] = x_feats1[feat] + dX
        else:
            feat_new_values = []
            for feat_val in x_feats1[feat]:
                if (feat_val + dX) > 1:
                    feat_new_values.append(feat_val - dX)
                else:
                    feat_new_values.append(feat_val + dX)
            x_feats2[feat] = np.array(feat_new_values)
        
        Y2_pred = make_predictions(x_feats2)
        dY = np.subtract(Y2_pred, Y1_pred)
        dY_dX[feat] = dY/dX
        print("DONE.")
    
    
    return (dY_dX)



# -----------------------------------------------
def calculate_delta_X(df_x_feats):
    """calculates local changes in feature values for each composition step

    Args:
        df_x_feats (pandas dataframe): alloy features for all compositions along composition pathway

    Returns:
        delta_X (pandas dataframe): change in feature values for each composition step
    """
    
    delta_X = df_x_feats.diff()
    delta_X = delta_X.dropna(axis=0)
    
    
    return (delta_X)



# -----------------------------------------------
def calculate_feat_contributions(Y_baseline_value, df_x_feats, dY_dX, delta_X):
    """calculates local and cumulative feature contributions along the composition pathway

    Args:
        Y_baseline_value (float): model prediction at baseline composition
        df_x_feats (pandas dataframe): alloy features for all compositions along composition pathway
        dY_dX (pandas dataframe): gradient of target property wrt each feature at every alloy compositions
        delta_X (pandas dataframe): change in feature values for each composition step

    Returns:
        loc_feat_contributions (pandas dataframe): local feature contributions for each composition step
        cum_feat_contributions (pandas dataframe): cumulative feature contributions along the composition pathway
    """
    
    feat_name_list = df_x_feats.columns.to_list()
    
    Y_baseline_row = Y_baseline_value * np.ones(shape=(1, len(feat_name_list)))
    Y_baseline_df = pd.DataFrame(data=Y_baseline_row, columns=feat_name_list)
    
    zero_row = np.zeros(shape=(1, len(feat_name_list)))
    zero_df = pd.DataFrame(data=zero_row, columns=feat_name_list)
    
    dY_wrt_feats1 = dY_dX[1:] * delta_X
    loc_feat_contributions = pd.concat([zero_df, dY_wrt_feats1]).reset_index(drop = True)
    
    dY_wrt_feats2 = pd.concat([Y_baseline_df, dY_wrt_feats1]).reset_index(drop = True)
    cum_feat_contributions = dY_wrt_feats2.cumsum()
    
    
    return (loc_feat_contributions, cum_feat_contributions)
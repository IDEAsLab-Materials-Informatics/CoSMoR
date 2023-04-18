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
    


# -----------------------------------------------
def get_miedema_enthalpy(alloy_name, state="liquid"):

    """Calculates binary enthalpies using Miedema's model and extends it to
    multi-component alloys using Takeuchi and Inoue's extended regular solid solution.

    Parameters
    ----------
    alloy_name : str
        alloy name string e.g. "NiAl2", "BaTiO3", "Al1.5Ti0.25CrFeNi"
    state : str (optional, default="liquid")
        state of alloy; possible values: "liquid" or "solid"; this affects 
        calculation only when the binary pair has a transition and non-transition

    Returns
    ----------
    enthalpy_dict : dict
        enthalpy dict with three keys:
            "H_chem" : chemical enthalpy of mixing
            "H_el" : elastic enthalpy of mixing
            "units" : units of enthalpy (kJ/mol)
    """

    dict_alloy_comp = get_comp_dict(alloy_name)
    el_list = dict_alloy_comp["el_list"]
    c_el_list = dict_alloy_comp["el_at_frac_list"]
    n_element = len(el_list)
    
    # calculating no. of unique binary systems in alloy
    n_binaries = int(
            np.math.factorial(n_element)
            /(np.math.factorial(n_element - 2)*np.math.factorial(2))
            )
    
    H_chem_alloy_array = np.zeros(n_binaries); # array to store H_chem of binaries; shape=(no. of binaries,)
    H_el_alloy_array = np.zeros(n_binaries); # array to store H_elastic of binaries; shape=(no. of binaries,)
    
    count = 0
    
    # loop to iterate over binary systems and calculate one binary enthalpy in each run; runs for 'no. of elements-1' times
    for j in range(0, n_element - 1):              # j represents first element in binary; say A
        for k in range(j + 1, n_element):          # k represents second element in binary; say B: j=0-> k=1,2,3,4 :: j=1->k=2,3,4 ::j =2->
            
            el_A = el_list[j]
            cA = 0.5

            # collecting el_A element data from 'db_element' database
            Vm_A = db_element.loc[el_A]["V_m"]
            w_fn_A = db_element.loc[el_A]["Work_Function"]
            nWS_A = db_element.loc[el_A]["n_WS"]
            K_A = db_element.loc[el_A]["K_GPa"]
            G_A = db_element.loc[el_A]["G_GPa"]
            type_A = db_element.loc[el_A]["Type"]

            el_B = el_list[k]
            cB = 0.5
            
            # collecting el_B element data from 'db_element' database
            Vm_B = db_element.loc[el_B]["V_m"]
            w_fn_B = db_element.loc[el_B]["Work_Function"]
            nWS_B = db_element.loc[el_B]["n_WS"]
            K_B = db_element.loc[el_B]["K_GPa"]
            G_B = db_element.loc[el_B]["G_GPa"]
            type_B = db_element.loc[el_B]["Type"]

            # surface concentrations of el_A and el_B
            cA_s = cA * (Vm_A**2/3) / (cA * (Vm_A**2/3) + 0.5 * (Vm_B**2/3))
            cB_s = cB * (Vm_B**2/3) / (cB * (Vm_A**2/3) + 0.5 * (Vm_B**2/3))

            del_w_fn = w_fn_A - w_fn_B  # diff in work function
            del_nWS1_3 = nWS_A**(1/3) - nWS_B**(1/3)    # delta(nWS^1_3)
            nWS_1_3_avg = (1/2) * (nWS_A**(-1/3) + nWS_B**(-1/3))   # average((nWS^-1_3)

            # volume concentrations of el_A and el_B
            Vm_A_corr = 1.5*cA_s*(Vm_A**(2/3))*(w_fn_A-w_fn_B)*((1/nWS_B)-(1/nWS_A))/(2*nWS_1_3_avg)
            Vm_B_corr = 1.5*cB_s*(Vm_B**(2/3))*(w_fn_B-w_fn_A)*((1/nWS_A)-(1/nWS_B))/(2*nWS_1_3_avg)
            Vm_A_alloy = Vm_A + Vm_A_corr   # corrected volume of el_A
            Vm_B_alloy = Vm_B + Vm_B_corr   # corrected volume of el_B

            # Selecting P,Q,R based on type of elements and alloy_phase
            if (type_A == "transition" and
                type_B == "transition"):

                P = 14.2; Q = 9.4*P; R = 0

            if (type_A == "non_transition" and
                type_B == "non_transition"):

                P = 10.7; Q = 9.4*P; R = 0

            if (type_A != type_B):
                P = 12.35; Q = 9.4*P

                R_P_A = db_element.loc[el_A]["R_P"]
                R_P_B = db_element.loc[el_B]["R_P"]
                R_P = R_P_A * R_P_B

                if state == 'solid': R = 1*R_P*P
                else: R = 0.73*R_P*P
            
            
            tau = (1/nWS_1_3_avg)*(-P*(del_w_fn**2)+Q*(del_nWS1_3**2)-R)    # tau parameter
            H_if_AinB = (Vm_A_alloy**(2/3))*tau
            H_if_BinA = (Vm_B_alloy**(2/3))*tau
            H_chem_AB = (cA*cB)*(cB_s*H_if_AinB + cA_s*H_if_BinA)  # calculating H_chemical
            
            H_el_AinB = 2*K_A*G_B*((Vm_A_alloy-Vm_B_alloy)**2)/(3*K_A*Vm_B_alloy + 4*G_B*Vm_A_alloy); # H_elastic A in B (kJ/mol)
            H_el_BinA = 2*K_B*G_A*((Vm_A_alloy-Vm_B_alloy)**2)/(3*K_B*Vm_A_alloy + 4*G_A*Vm_B_alloy); # H_elastic B in A (kJ/mol)
            H_el_AB = cA*cB*(cB*H_el_AinB + cA*H_el_BinA); # H_elastic of A-B binary (kJ/mol)
            
            cA_alloy, cB_alloy = c_el_list[j], c_el_list[k]     #actual at. conc. of A and B
            H_chem_alloy_array[count] = 4*cA_alloy*cB_alloy*H_chem_AB; # added to 'H_chemical' array
            H_el_alloy_array[count] = 4*cA_alloy*cB_alloy*H_el_AB; # added to 'H_elastic' array
            
            count += 1

    # alloy enthalpies = mean of all binary enthalpies
    H_chem_alloy = np.around(np.sum(H_chem_alloy_array), 4)
    H_el_alloy = np.around(np.sum(H_el_alloy_array), 4)
    
    enthalpy_dict = {
      "H_chem": H_chem_alloy,
      "H_el": H_el_alloy,
      "units": "kJ/mol"
    }
    
    return (enthalpy_dict)
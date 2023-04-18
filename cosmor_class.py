import pandas as pd
import numpy as np
import pickle
import os
from core_functions import (create_alloys, create_features, make_predictions,
                            calculate_dY_dX, calculate_delta_X, calculate_feat_contributions)

SEP = os.sep

class cosmor:
    
    def __init__(self):
        
        print("\n--- Collecting user input to create 'CoSMoR' class instance ---")
        self.A = str(input("\t-Enter component A (e.g.:'Al', 'AlTi', 'Al2Ti'):\t"))
        self.B = str(input("\t-Enter component B (e.g.:'Co', 'CoCr', 'Co2Cr'):\t"))
        self.dc = float(input("\t-Enter composition step size in at. fraction (typically 0.01):\t"))
        self.cAmin = float(input("\t-Enter starting concentration of component A:\t"))
        self.cAmax = float(input("\t-Enter maximum concentration of component A:\t"))
        self.dX = float(input("\t-Enter feature step size (typically 0.02):\t"))
        self.save_bool = input("\t-Save results? [Yes/No (Y/N)]:\t")
        self.plot_bool = input("\t-Plot results? [Yes/No (Y/N)]:\t")
        print("CoSMoR instance created successfully.")
    
            
    def save_cosmor(self):
        
        results_dir = f"cosmor_results{SEP}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        savename = f"{self.A}-{self.B}-dc_{self.dc}-dX_{self.dX}"
        print(f"|Saving cosmor results as excel file '{savename}.xlsx'...", end=" ", flush=True)
        
        writer = pd.ExcelWriter(f"{results_dir}{savename}.xlsx", engine='xlsxwriter')
        
        df_Y_pred = pd.DataFrame(data=self.Y_pred_ML, columns=["Y_predicted"])
        pd.concat([self.df_alloys, df_Y_pred], axis=1).to_excel(writer, sheet_name="Y_Prediction", index = True)
        pd.concat([self.df_alloys, self.df_x_feats], axis=1).to_excel(writer, sheet_name="Feature_values", index = True)
        pd.concat([self.df_alloys, self.delta_X], axis=1).to_excel(writer, sheet_name="delta_X", index = True)
        pd.concat([self.df_alloys, self.dY_dX], axis=1).to_excel(writer, sheet_name="PLD-dY_dX", index = True)
        pd.concat([self.df_alloys, self.loc_feat_contri], axis=1).to_excel(writer, sheet_name="loc_feat_contributions", index = True)
        pd.concat([self.df_alloys, self.cum_feat_contri], axis=1).to_excel(writer, sheet_name="cum_feat_contributions", index = True)
        
        writer.save()
        print("DONE.")
        
        print(f"|Saving cosmor class as pickle file '{savename}.cosmor'...", end=" ", flush=True)
        with open(f"{results_dir}{savename}.cosmor", 'wb') as f:
            pickle.dump(self, f)
        print("DONE.")
    
    
    def plot_cosmor(self):
        
        import matplotlib
        from matplotlib import pyplot as plt
        
        print(f"|Creating plots for CoSMoR results...", end=" ", flush=True)
        x_axis = np.array(self.df_alloys[f"x [A={self.A}]"])
        x_axis_label = f"x [({self.A}) at.fraction]"
        
        font = {'family' : 'Arial',
                'weight' : 'regular',
                'size'   : 20}
        matplotlib.rc('font', **font)

        fig_size = (8, 6)
        title_fs, label_fs, legend_fs = 25, 20, 13
        ss = 60 #symbol size
        lw = 5 #linewidth
        nCols = 2
        nRows = 2

        plt.figure(figsize=(fig_size[0]*nCols, fig_size[1]*nRows))

        # Plot first figure: Cumulative feature contributions
        plt.subplot(nRows, nCols, 1)
        plt.title("Cumulative feature contributions", fontsize=title_fs)
        plt.xlabel(x_axis_label, fontsize=label_fs)
        plt.ylabel("Feature contributions", fontsize=label_fs)

        for feat in self.feat_name_list:
            plt.plot(x_axis, self.cum_feat_contri[feat], linewidth=lw, label=feat, alpha=0.75)

        plt.plot([x_axis[0], x_axis[-1]], [self.Y_baseline_value, self.Y_baseline_value],
                "--", linewidth=2, label="baseline", alpha=1)

        plt.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw-2, label="Y_overall", alpha=0.75)

        plt.legend(fontsize=legend_fs)

        # Plot second figure: Local feature contributions
        plt.subplot(nRows, nCols, 2)
        plt.title("Local feature contributions (For each composition step)", fontsize=title_fs)
        plt.xlabel(x_axis_label, fontsize=label_fs)
        plt.ylabel("Feature contributions", fontsize=label_fs)

        for feat in self.feat_name_list:
            plt.plot(x_axis[1:], np.array(self.loc_feat_contri[feat])[1:],
                     label=feat, alpha=0.75)

        plt.legend(fontsize=legend_fs)

        # Plot third figure: Compare overall predictions from ML model and CoSMoR
        plt.subplot(nRows, nCols, 3)
        plt.title("Comparing Y prediction (ML vs. CoSMoR)", fontsize=title_fs)
        plt.xlabel(x_axis_label, fontsize=label_fs)
        plt.ylabel("Overall Y prediction", fontsize=label_fs)
        plt.plot(x_axis, self.Y_pred_ML, "--", linewidth=lw, label="Y_ML", alpha=0.75)
        plt.plot(x_axis, self.Y_pred_cosmor, linewidth=lw, label="Y_CoSMoR", alpha=0.75)
        plt.legend(fontsize=legend_fs)

        # Plot fourth figure: Feature variations
        plt.subplot(nRows, nCols, 4)
        plt.title("Feature variations", fontsize=title_fs)
        plt.xlabel(x_axis_label, fontsize=label_fs)
        plt.ylabel("Feature values", fontsize=label_fs)

        for feat in self.feat_name_list:
            plt.plot(x_axis, self.df_x_feats[feat], linewidth=lw, label=feat, alpha=0.75)

        plt.legend(fontsize=legend_fs)

        plt.tight_layout()
        print("DONE.")
        
        results_dir = f"cosmor_results{SEP}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        fig_savename = f"{self.A}-{self.B}-dc_{self.dc}-dX_{self.dX}"
        print(f"|Saving plots as pdf file '{fig_savename}-PLOTS.pdf'...", end=" ", flush=True)
        plt.savefig(f"{results_dir}{fig_savename}-PLOTS.pdf", bbox_inches='tight')
        plt.close()
        print("DONE.")
        
    
    def run_cosmor(self):
        
        print("\n--- Running 'CoSMoR' framework ---")
        print("|Creating alloys along composition pathway...", end=" ", flush=True)
        self.df_alloys = create_alloys(self.A, self.B, self.dc, self.cAmin, self.cAmax)
        print("DONE.")
        
        print("|Creating feature values...", end=" ", flush=True)
        self.df_x_feats = create_features(self.df_alloys)
        print("DONE.")
        
        print("|Extracting feature names...", end=" ", flush=True)
        self.feat_name_list = self.df_x_feats.columns.to_list()
        print("DONE.")
        
        print("|Generating overall model predictions along composition pathway...", end=" ", flush=True)
        self.Y_pred_ML = make_predictions(self.df_x_feats)
        print("DONE.")
        
        print("|Identifying baseline value of target property...", end=" ", flush=True)
        self.Y_baseline_value = self.Y_pred_ML[0]
        print("DONE.")
        
        print("|Calculating local partial dependencies [dY/d(Xi)] along the composition pathway...")
        self.dY_dX = calculate_dY_dX(self.df_x_feats, self.dX)

        print("|Calculating feature variations for compositional stimulus along composition pathway...", end=" ", flush=True)
        self.delta_X = calculate_delta_X(self.df_x_feats)
        print("DONE.")
        
        print("|Calculating feature contributions along composition pathway...", end=" ", flush=True)
        
        self.loc_feat_contri, self.cum_feat_contri = calculate_feat_contributions(self.Y_baseline_value,
                                                                                  self.df_x_feats,
                                                                                  self.dY_dX,
                                                                                  self.delta_X)
        
        print("DONE.")
        
        print("|Calculating predictions based on cumulative contributions...", end=" ", flush=True)
        self.Y_pred_cosmor = self.Y_baseline_value + np.sum(self.cum_feat_contri -
                                                            self.Y_baseline_value, axis=1)
        
        print("DONE.")
        
        
        if self.save_bool.lower() in ["yes", "y"]:
            self.save_cosmor()
            
        if self.plot_bool.lower() in ["yes", "y"]:
            self.plot_cosmor()
            
        print("\n--- COMPLETED ---")
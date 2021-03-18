import numpy as np
import pandas as pd

class AlignInputTarget:
    """
    This class aligns the input and target dataframes of the PD datasets based on patient ID.
    The function is written to align variant count data as input, the indices of which are PPMI_SI_<num> (string),
    and PD progression subtype as target, the indices of which are simply <num> (integer).
    Each method returns the appropriately reduced input dataframe and target dataframe.

    Example of how to use this:
    aligner = AlignInputTarget(X, y)
    aligned_input = aligner.align_input()
    aligned_target = aligner.align_target()
    """
    
    def __init__(self, input_df, target_df):
        self.inputindex = input_df.index
        self.targetindex = target_df.index
        self.input_df = input_df
        self.target_df = target_df
        self.ol_patid = []
        # have ol_patid be the patient IDs that i) exists in target_df, and ii) where "label" is not NA in target_df
        for i in range(len(self.inputindex)):
            if self.inputindex[i] in self.targetindex:
                if ~np.isnan(self.target_df.loc[self.inputindex[i]]["label"]):
                    self.ol_patid.append(self.inputindex[i])
                else:
                    continue
    
    def align_input(self): # keeps the input's index order 
        new_input = self.input_df.loc[self.ol_patid]
        return new_input
    
    def align_target(self): # changes the target's index order according to the input's index order
        new_target = self.target_df.loc[self.ol_patid]
        return new_target
    
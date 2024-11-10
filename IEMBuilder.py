import pandas as pd
import numpy as np
# from scipy.spatial import procrustes
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import umap
# from seaborn import color_palette
# from itertools import product

"""
A wrapper class for specific functions of use to build IEM.

"""

class IEM:
    
    def __init__(self, 
                 n_channels = 6, 
                 ch_width = 60*np.pi/180,
                 ch_preferences = None,
                 stimulus_space = np.arange(-np.pi, np.pi, np.pi/180)[:-1],
                 exponent = 7):
        
        self.n_channels = n_channels
        self.channel_width = ch_width
        self.stimulus_space = stimulus_space
        self.cosine_exponent = exponent
        
        if ch_preferences is None: 
            self.channel_preferences = np.linspace(-np.pi, np.pi,n_channels+1)[:-1]
            
        self._test_idx, self._train_idx, self._validation_idx = [], [], []
        self._isValidation = False
    
    @property
    def channel_preferences(self): return self.channels.preferences
    
    @channel_preferences.setter
    def channel_preferences(self, value):
        if len(value) != self.n_channels: 
            raise(ValueError("channel_preferences must contain exact number of elements as n_channels."))
        self.channels.preferences = value
    
    @property
    def D(self): return self.design_matrix
    
    @D.setter
    def D(self, value):
        self.design_matrix = value
        
    @property
    def design_matrix(self): return self._design_matrix
    
    @design_matrix.setter
    def design_matrix(self, value):
        value = np.array(value)
        if value.ndim < 2: value = value[:,np.newaxis]
        self._design_matrix = value
        
    @property
    def training_indices(self): 
        return self._train_idx
    
    @training_indices.setter
    def training_indices(self, value):
        self._train_idx = value
        
    @property
    def testing_indices(self): 
        return self._test_idx
        
    @testing_indices.setter
    def testing_indices(self, value):
        self._test_idx = value 
        
    @property
    def validation_indices(self): return self._validation_idx
    
    @validation_indices.setter
    def validation_indices(self, value):
        self._validation_idx = value 
    
    @property
    def C(self): return self.channels.responses.C
    
    @property
    def C_train(self): 
        return self.C[self.training_indices, :]
    
    @property
    def C_test(self): 
        return self.C[self.testing_indices, :]
    
    @property
    def C_validation(self): 
        return self.C[self.validation_indices, :]
    
    #left out group, either testing or validation subset
    @property
    def _C_lo(self):
        if self._isValidation: return self.C_validation
        else: return self.C_test
    
    @property
    def B(self): return self._B
    
    @B.setter
    def B(self, value):
        self._B = value
    
    @property
    def B_train(self): 
        return self.B[self.training_indices, :]
    
    @property
    def B_test(self): 
        return self.B[self.testing_indices, :]
    
    @property
    def B_validation(self): 
        return self.B[self.validation_indices, :]
    
    #left out group, either testing or validation subset
    @property
    def _B_lo(self):
        if self._isValidation: return self.B_validation
        else: return self.B_test
    
    @property
    def channels(self):
        
        class Channels:
            
            def __init__(self, parent):
                
                self.parent = parent  # Store a reference to the parent object (IEM)
                
            class Bases:
                
                def __init__(self, parent):
                    
                    self.parent = parent # Channels
                    # half-rectified cosine as the basis function
                    def halfrectcos(self, center):        
                        return(np.cos(((self.parent.stimulus_space + center + np.pi) % (2 * np.pi) - np.pi)/self.parent.width)**self.parent.exponent)
                   
                    ch_bases = np.vstack([
                        halfrectcos(self, chN) 
                        for chN in self.parent.preferences
                        ])
                    ch_bases[ch_bases<0] = 0
                    
                    self.set = ch_bases
                
                def plot(self):
                
                    plt.plot(self.parent.stimulus_space, self.set.T);
                    plt.xlabel("stimulus space in radians")
                    plt.ylabel("Channel Response (a.u.)")
                    plt.title("Response tuning functions of the encoding channels");
                    
            
            class Responses:
                
                def __init__(self, parent):
                    
                    self.parent = parent # Channels
                    self.D = self.parent.parent.D # IEM.D
                    
                    def get_indices_in_A_of_each_D_(A, D):
                        return np.argmin(np.abs(D[:, :, np.newaxis] - np.tile(A, (D.shape[0], 1))[:, np.newaxis, :]), axis=2)
                    
                    idx = get_indices_in_A_of_each_D_(self.parent.stimulus_space, self.D)
                    resp = self.parent.bases.set[:, idx].sum(axis=-1)
                    resp /= resp.sum(axis=0)
                    
                    self.C = resp.T
                    
                def plot(self, trial_no = []):
                    
                    if len(trial_no) == 0: trial_no = range(self.C.shape[0])
                    
                    plt.plot(self.parent.stimulus_space, (self.C[trial_no,:] @ self.parent.bases.set).T)
                    plt.xlabel("stimulus space in radians")
                    plt.ylabel("Channel Response (a.u.)")
                    
            @property
            def bases(self): 
                return self.Bases(self)
            
            @property
            def responses(self): 
                
                if not hasattr(self.parent, "design_matrix"): return None
                return self.Responses(self)
            
            @property
            def stimulus_space(self): return self.parent.stimulus_space
            
            @property
            def n(self): return self.parent.n_channels
            
            @property
            def width(self): return self.parent.channel_width
            
            @property
            def exponent(self): return self.parent.cosine_exponent
            
            @property
            def preferences(self): return self._preferences
            
            @preferences.setter
            def preferences(self, value):
                
                self._preferences = value
                
        return Channels(self)
    
    @property
    def W_hat(self):
        
        return (
            self.B_train.T @ 
            self.C_train @ 
            np.linalg.inv(
                self.C_train.T @ 
                self.C_train
                )
            )
        
    @property
    def C_hat(self):
        
        return(
            np.linalg.inv(
                self.W_hat.T @
                self.W_hat
                ) @ 
            self.W_hat.T @ 
            self._B_lo.T
        ).T
        
    @property
    def reconstructions(self): 
        return self.C_hat @ self.channels.bases.set
    
    @property
    def centered_reconstructions(self):
        
        shift_to_cent_by = [
            len(self.stimulus_space)//2 - 
            np.argmin(
                np.abs(
                    self.stimulus_space - centN
                    )
                ) for centN in self.stimulus_centers[self._test_idx]]
        
        return np.vstack([
            np.roll(self.C_hat[n,:], shiftN) 
             for n, shiftN in zip(range(self.C_hat.shape[0]), shift_to_cent_by)
             ])
    
    @property
    def stimulus_centers(self):
        return self._centers
    
    @stimulus_centers.setter
    def stimulus_centers(self, value):        
        self._centers = value
        
    @property
    def _leftout_idx(self):
        if self._isValidation: return self.validation_indices
        else: return self.testing_indices
        
    def initiate_validation(self, 
                            hyperparameter,                           
                            metric):
        
        self._isValidation = True
        self.validation = self.Validation(hyperparameter, metric)
        
        
    class Validation:
        
        def __init__(
            self, 
            hyperparameter,
            obj_metric
            ):
            
            self.hyperparameter = np.array(hyperparameter)
            self.obj_metric = obj_metric
            
        
        @property
        def k_fold(self): return self.hyperparameter.size
        
        
    @property
    def fidelity(self): pass  
    
    @property
    def r2(self): pass
    
    @property
    def loo_svm_accuracy(self): pass
    
    @property
    def nll(self): pass
            
    @staticmethod
    def deg2rad(d): return(d*np.pi/180)
    
    @staticmethod
    def rad2deg(r): return(r*180/np.pi)
    
    
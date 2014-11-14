import pdb
"""
Created on Tue Sep  9 22:21:40 2014

@author: rkp

Code for GLM, ARMA models
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.genmod.families.family as F
import statsmodels.genmod.families.links as L

FILT_COLORS = ['r','g']

class arma():
    """ARMA model"""
    
    def __init__(self):
        """Constructor."""
        
        self.in_columns = None
        self.out_columns = None
        
    def label_columns(self,in_labels,out_label,filter_lengths):
        """Store column labels for easy reference later.
        Args:
            in_labels: List of input column labels.
            out_label: Output column label.
            filter_lengths: dictionary of filter lengths
        """
        self.in_labels = in_labels
        self.out_label = out_label
        # Move filter lengths from dictionary to arrays
        self.in_filter_lengths = [filter_lengths[label] for label in in_labels]
        self.out_filter_length = filter_lengths[out_label]
        # Calculate cumulative sum of filter lengths (+ 1 for constant)
        self.all_filter_lengths = self.in_filter_lengths + \
            [self.out_filter_length,1]
        self.filter_len_cumsum = np.cumsum(self.all_filter_lengths)
        # Store maximum filter length
        self.max_filter_len = np.max(self.all_filter_lengths)
        # Calculate logical filter idxs of flattened input filter array
        self.in_filter_idxs = [None for ii in range(len(in_labels))]
        for ii in range(len(self.in_filter_idxs)):
            filter_idxs = np.zeros((self.filter_len_cumsum[-1],),dtype=bool)
            # Calculate start and stop idx for this filter
            if ii == 0:
                start_idx = 0
            else:
                start_idx = self.filter_len_cumsum[ii-1]
            stop_idx = self.filter_len_cumsum[ii]
            # Set logical idxs for this filter to True and store
            filter_idxs[start_idx:stop_idx] = True
            self.in_filter_idxs[ii] = filter_idxs
        # Calculate logical filter idxs for flattened output filter array
        self.out_filter_idxs = np.zeros((self.filter_len_cumsum[-1],),dtype=bool)
        start_idx = self.filter_len_cumsum[-3]
        stop_idx = self.filter_len_cumsum[-2]
        self.out_filter_idxs[start_idx:stop_idx] = True
        
    def set_filters(self,f,flattened=False):
        """Set the filters for this model.
        Args:
            f: Filter dictionary or 1D array.
            flattened: Set to True if filters provided as 1D array."""
        if flattened:
            # Set input filters
            self.in_filters = [f[f_idx] for f_idx in self.in_filter_idxs]
            # Set output filter
            self.out_filter = f[self.out_filter_idxs]
            # Set constant
            self.constant = f[-1]
        else:
            # Set input filters
            self.in_filters = [f[feat] for feat in self.in_labels]
            # Set output filter
            self.out_filter = f[self.out_label]
            # Set constant
            self.constant = f['constant']
        
    def get_filters(self,flattened=False):
        """Return all filters as a dictionary.
        
        Args:
            flatten: Set to true to concatenate all filters in 1D array"""
        filters = self.in_filters + [self.out_filter]
        # Return either dictionary or flattened array
        if flattened:
            return np.concatenate(filters + [np.array([self.constant])])
        else:
            # Make dictionary from labels and filters
            filter_dict = dict(zip(self.in_labels + [self.out_label],filters))
            filter_dict['constant'] = self.constant
            return filter_dict

    def show_filters(self):
        """Show all filters graphically."""
        # Get all filters
        filters = self.get_filters()
        # Figure out which filters to show/print & decide how many axes to open
        filters_to_show = []
        filters_to_print = []
        for (k,v) in filters.items():
            if not isinstance(v,(float,int)):
                if len(v) > 1:
                    filters_to_show += [k]
                elif len(v) == 1:
                    filters_to_print += [k]
            else: filters_to_print += [k]
        num_axs = len(filters_to_show) + 1
        # Open figure & axes
        fig,axs = plt.subplots(1,num_axs,facecolor='white')
        for ax_idx,ax in enumerate(axs):
            if ax_idx < len(axs) - 1:
                # Plot filter
                ax.plot(filters[filters_to_show[ax_idx]],c=FILT_COLORS[ax_idx],lw=3)
                ax.set_xlim(0,self.max_filter_len)
                ax.set_xlabel('t')
                ax.set_ylabel('strength')
                ax.set_title(filters_to_show[ax_idx])
            elif ax_idx == len(axs) - 1:
                # Display all '1D' filters as constants
                text_y = .9
                text_delta_y = .12
                for f_idx,filter_to_print in enumerate(filters_to_print):
                    if filter_to_print != 'constant':
                        val = filters[filter_to_print][0]
                        ax.text(.1,text_y,'%s: %.4f'%(filter_to_print,val))
                        # Move to next location
                        text_y -= text_delta_y
                # Display constant
                ax.text(.1,text_y,'constant: %.4f'%filters['constant'])
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(16)
        
        return fig,axs
    
    def sample_output(self,in_data):
        """Sample output time-series using input data.
        
        Args:
            in_data: Array/list of input arrays.
        Returns:
            Array of output data arrays."""
        # Loop through all trials
        mean_data = [None for tr in in_data]
        out_data = [None for tr in in_data]
        for tr_idx, tr in enumerate(in_data):
            tr_len = tr.shape[0]
            # Set output data to zeros
            tr_out = np.zeros((tr_len,),dtype=float)
            # Skip if shorter than longest filter
            if tr_len < self.max_filter_len:
                out_data[tr_idx] = np.zeros((tr_len,),dtype=float)
                out_data[tr_idx][:] = np.nan
                mean_data[tr_idx] = out_data[tr_idx].copy()
                continue
            # Calculate filtered input all at once
            filtered_in = self.constant*np.ones((tr.shape[0],),dtype=float)
            # Set everything too early for filter to work on to nan
            filtered_in[:self.max_filter_len] = np.nan
            # Loop through all inputs and filter them
            for idx,in_filter in enumerate(self.in_filters):
                # Add to filter input to total
                filtered_in[len(in_filter):] += \
                    fftconvolve(tr[:,idx],in_filter,mode='valid')[:-1]
            # Generate output with history filter
            for t in range(self.max_filter_len,tr_len):
                # Filter output
                filtered_out = np.dot(self.out_filter[::-1],
                                      tr_out[t-self.out_filter_length:t])
                # Calculate mean
                tr_out[t] = filtered_in[t] + filtered_out
            out_data[tr_idx] = tr_out
                                 
        return out_data    
        
    def predict(self,in_data,out_data):
        """Calculate mean/intensity using input & output data.
        Points in the beginning of the array that cannot be predicted because
        of nonzero filter length are left as nan. Therefore each array in the 
        output data will be the same length as its corresponding array in the 
        input data.
        
        Returns:
            Array of 1D output arrays."""
        # Loop through all trials
        tr_means = [None for tr in in_data]
        for tr_idx, tr in enumerate(in_data):
            # Get corresponding output data
            tr_out = out_data[tr_idx]
            tr_len = tr.shape[0]
            # Skip if shorter than longest filter
            if tr_len < self.max_filter_len:
                tr_means[tr_idx] = np.zeros((tr_len,),dtype=float)
                tr_means[tr_idx][:] = np.nan
                continue
            # Calculate linear part
            tr_linear = self.constant*np.ones((tr.shape[0],),dtype=float)
            # Set everything too early for filter to work on to nan
            tr_linear[:self.max_filter_len] = np.nan
            # Loop through all inputs and filter them
            for idx,in_filter in enumerate(self.in_filters):
                filtered_input = fftconvolve(tr[:,idx][:-1],in_filter,mode='valid')
                # Add to linear filter output
                tr_linear[len(in_filter):] += filtered_input
            # Add filtered output
            if self.filter_output_estimate:
                # Set up space for mean
                tr_mean = np.zeros((tr_len,),dtype=float)
                tr_mean[:] = np.nan
                tr_mean[:self.max_filter_len] = tr_out[:self.max_filter_len]
                # Calculate rest of filtered output sequentially
                for t in range(self.max_filter_len,tr_len):
                    # Calculate filtered output contribution to linear part
                    tr_linear[t] += np.dot(self.out_filter[::-1],
                                           tr_mean[t-self.out_filter_length:t])
                    # Pass complete linear part through nonlinearity
                    tr_mean[t] = self.nonlin(tr_linear[t])
                tr_means[tr_idx] = tr_mean
            else:
                filtered_output = fftconvolve(tr_out[:-1],self.out_filter,mode='valid')
                tr_linear[len(self.out_filter):] += filtered_output
                # Pass through nonlinearity and store
                tr_means[tr_idx] = tr_linear
        return tr_means
        
    def fit(self,in_data,out_data,f0='rand',flattened=False,**kwargs):
        """Fit ARMA to data.
        Args:
            in_data: Array of 2D input data arrays, where rows are time points.
            out_data: Array of 1D output arrays.
            f0: initial guess for filter coefficients
            
        """
        
        # Generate data matrix
        self.build_data_matrix(self, in_data, out_data)
                
        
        # Store filters
        self.set_filters(best_filters,flattened=True)
        
    def rand_filters(self,flattened=True):
        """Generate & return random filters.
        
        Args:
            flatten: Set to true to concatenate all filters in 1D array"""
        # Draw filters from normal distribution
        filters = [np.random.normal(0,.01,(l,)) for l in self.all_filter_lengths]
        # Draw constant from normal distribution
        constant = np.random.normal(0,.1)
        # Return either dictionary or flattened array
        if flattened:
            return np.concatenate(filters + [np.array([constant])])
        else:
            # Make dictionary from labels and filters
            filter_dict = dict(zip(self.in_labels + [self.out_label],filters))
            filter_dict['constant'] = constant
            return filter_dict
    
    def build_data_matrix(self, in_data, out_data):
        """Construct & store data matrix from the time-series data"""
        print 'Generating data matrix...'
        # Calculate usable time-points based on max filter lengths
        usable_outs = [d[self.max_filter_len:] for d in out_data]
        usable_lengths = np.array([len(d) for d in usable_outs])
        in_matrix = np.zeros((usable_lengths.sum(),self.filter_len_cumsum[-1]),dtype=float)
        # Fill in inputs
        in_matrix[:,-1] = 1.
        # Loop through trials
        row_ctr = 0
        for tr_idx,tr_in in enumerate(in_data):
            start_row = row_ctr
            stop_row = row_ctr + usable_lengths[tr_idx]
            # Loop through input filters
            for col,f_idxs in enumerate(self.in_filter_idxs):
                # Make toeplitz matrix for this input
                l = self.in_filter_lengths[col]
                in_toeplitz = toeplitz(tr_in[self.max_filter_len-1:-1,col],tr_in[self.max_filter_len-l:self.max_filter_len,col][::-1])
                in_matrix[start_row:stop_row,f_idxs] = in_toeplitz
            # Put outputs into input matrix
            tr_out = out_data[tr_idx]
            l = self.out_filter_length
            out_toeplitz = toeplitz(tr_out[self.max_filter_len-1:-1],tr_out[self.max_filter_len-l:self.max_filter_len][::-1])
            in_matrix[start_row:stop_row,self.out_filter_idxs] = out_toeplitz
            row_ctr += usable_lengths[tr_idx]
        # Fill in outputs
        out_vec = np.concatenate(usable_outs)
        self.in_matrix = in_matrix
        self.out_vec = out_vec
        print 'Complete.'
            
            
class glm_time(arma):
    """GLM specifically designed to fit time-series data.
    
    Args:
        family_name: which distribution to use (normal, binomial, poisson)
        link_name: which link to use (logit, log, identity, sqrt, probit)
        fam_params: parameter(s) of family
        """
    
    def __init__(self,family_name='normal',link_name='identity',fam_params=None):
        """Constructor."""
        
        # Store link
        self.link_name = link_name
        if self.link_name.lower() == 'logit':
            self.link = L.logit
        elif self.link_name.lower() == 'log':
            self.link = L.log
        elif self.link_name.lower() == 'identity':
            self.link = L.identity
        elif self.link_name.lower() == 'sqrt':
            self.link = L.sqrt
        elif self.link_name.lower() == 'probit':
            self.link = L.probit
        family_kwargs = {}
        if self.link_name:
            family_kwargs['link'] = self.link
        # Store family
        self.family_name = family_name
        if self.family_name.lower() == 'normal':
            self.family = F.Gaussian(**family_kwargs)
            def rand(x): return np.random.normal(x,fam_params)
        elif self.family_name.lower() == 'binomial':
            self.family = F.Binomial(**family_kwargs)
            def rand(x): return np.random.binomial(1,x)
        elif self.family_name.lower() == 'poisson':
            self.family = F.Poisson(**family_kwargs)
            def rand(x): return np.random.poisson(x)
                
        self.rand = rand
        self.in_columns = None
        self.out_columns = None
    
    def sample_output(self,in_data,out_init_data=None,mean_only=False):
        """Sample output time-series using input data.
        
        Args:
            in_data: Array/list of input arrays.
            out_init_data: List of initial outputs.
        Returns:
            Array of output data arrays."""
        # Loop through all trials
        mean_data = [None for tr in in_data]
        out_data = [None for tr in in_data]
        for tr_idx, tr in enumerate(in_data):
            tr_len = tr.shape[0]
            # Set output data to zeros
            tr_out = np.zeros((tr_len,),dtype=float)
            # Set initial output data if given
            if out_init_data is not None:
                tr_out[:self.max_filter_len] = out_init_data[tr_idx][:,0]
            tr_mean = np.zeros((tr_len,),dtype=float)
            tr_mean[:] = np.nan
            # Skip if shorter than longest filter
            if tr_len < self.max_filter_len:
                out_data[tr_idx] = np.zeros((tr_len,),dtype=float)
                out_data[tr_idx][:] = np.nan
                mean_data[tr_idx] = out_data[tr_idx].copy()
                continue
            # Calculate filtered input all at once
            filtered_in = self.constant*np.ones((tr.shape[0],),dtype=float)
            # Set everything too early for filter to work on to nan
            filtered_in[:self.max_filter_len] = np.nan
            # Loop through all inputs and filter them
            for idx,in_filter in enumerate(self.in_filters):
                if len(in_filter):
                    # Add to filter input to total
                    filtered_in[len(in_filter):] += \
                        fftconvolve(tr[:,idx],in_filter,mode='valid')[:-1]
            # Generate output with history filter
            for t in range(self.max_filter_len,tr_len):
                # Filter output
                filtered_out = np.dot(self.out_filter[::-1],
                                      tr_out[t-self.out_filter_length:t])
                # Calculate mean
                tr_mean[t] = self.link().inverse(filtered_in[t] + filtered_out)
                if mean_only:
                    tr_out[t] = tr_mean[t]
                else:
                    # Sample output from mean
                    tr_out[t] = self.rand(tr_mean[t])
            out_data[tr_idx] = tr_out
                                 
        return out_data    
        
    def predict_mean(self,in_data,out_data):
        """Calculate mean/intensity using input & output data.
        Points in the beginning of the array that cannot be predicted because
        of nonzero filter length are left as nan. Therefore each array in the 
        output data will be the same length as its corresponding array in the 
        input data.
        
        Returns:
            Array of 1D output arrays."""
        # Loop through all trials
        tr_means = [None for tr in in_data]
        for tr_idx, tr in enumerate(in_data):
            # Get corresponding output data
            tr_out = out_data[tr_idx]
            tr_len = tr.shape[0]
            # Skip if shorter than longest filter
            if tr_len < self.max_filter_len:
                tr_means[tr_idx] = np.zeros((tr_len,),dtype=float)
                tr_means[tr_idx][:] = np.nan
                continue
            # Calculate linear part
            tr_linear = self.constant*np.ones((tr.shape[0],),dtype=float)
            # Set everything too early for filter to work on to nan
            tr_linear[:self.max_filter_len] = np.nan
            # Loop through all inputs and filter them
            for idx,in_filter in enumerate(self.in_filters):
                filtered_input = fftconvolve(tr[:,idx][:-1],in_filter,mode='valid')
                # Add to linear filter output
                tr_linear[len(in_filter):] += filtered_input
            # Add filtered output
            filtered_output = fftconvolve(tr_out[:-1],self.out_filter,mode='valid')
            tr_linear[len(self.out_filter):] += filtered_output
            # Pass through nonlinearity and store
            tr_means[tr_idx] = self.link().inverse(tr_linear)
        return tr_means
        
    def fit(self, in_data, out_data,**kwargs):
        """Fit GLM to data.
        Args:
            in_data: Array of 2D input data arrays, where rows are time points.
            out_data: Array of 1D output arrays.
            filter_lengths: Dict of filter lengths.
            method: What method to use to fit ('lsq','max_like')
            filter_output_estimate: Whether to filter the output estimate in
                calculating the next output (as opposed to filtering the true
                output"""
        
        self.build_data_matrix(in_data,out_data)
        # Fit glm
        print 'Fitting GLM...'
        self.glm = sm.GLM(self.out_vec, self.in_matrix, self.family)
        self.fit_results = self.glm.fit(**kwargs)
        best_filters = self.fit_results.params
        # Store filters
        self.set_filters(best_filters,flattened=True)
        print 'Complete.'
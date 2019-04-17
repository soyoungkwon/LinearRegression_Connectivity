# statistics
# import libraries, files

import numpy as np
from scipy import stats
import scipy.stats  as stats
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# define ROI name
hemis = ['L', 'R']
DXN_Name = ['FEF','IPS','LP','MFC']
dxn_name = []
for d_name in DXN_Name:
    for hemi in ['L', 'R']:
        dxn_name += [d_name + hemi]

# load linear coef (gain, base)
linear_param = np.load("linear_3d.npy")
linear_param.shape

# get each parameters
gain_rest  = linear_param[:,:,0]
base_rest  = linear_param[:,:,1]
gain_atten = linear_param[:,:,2]
base_atten = linear_param[:,:,3]


# anova of rest vs atten
for dxn in range(8):
    gain_atten_dxn = gain_atten[:,dxn]
    gain_rest_dxn =  gain_rest[:,dxn]
    base_atten_dxn = base_atten[:,dxn]
    base_rest_dxn =  base_rest[:,dxn]
    [t_base, p_base ] = stats.f_oneway(base_atten_dxn, base_rest_dxn)
    [t_gain, p_gain ] = stats.f_oneway(gain_atten_dxn, gain_rest_dxn)
    print('base diff ' + dxn_name[dxn] + ' p:' +  str(p_base))
    print('gain diff ' + dxn_name[dxn] + ' p:' + str(p_gain))


# change numpy --> pandas dataframe
# gain_rest_pd = pd.DataFrame(gain_rest.T, index = dxn_name).T

sub_id_all = np.matlib.repmat(np.arange(20), 1, 8).T
sub_id_1d = sub_id_all.tolist()
sub_id_1d
# a = np.matlib.repmat('FEF', 2, 1), np.matlib.repmat('IPS', 2, 1)
area = []
for rep1 in range(20):
    for d_name in DXN_Name:
        for rep2 in range(2):
            area  += [d_name]
hemi = []
for rep3  in range(80):
    for n in ['L', 'R']:
        hemi += n
len(hemi)
# create coef list
gain_rest_1d = gain_rest.T.ravel().tolist()

# creat sub_id list
sub_id_once = (np.arange(20).reshape(1,20))
sub_id_all_np = np.repeat(sub_id_once, 8,0).ravel()
sub_id_all_list = sub_id_all_np.tolist()

# create pandas
help(AnovaRM)
gain_rest_pd = pd.DataFrame({'sub_id': sub_id_all_list, 'gain_rest': gain_rest_1d, 'area': area, 'hemi': hemi})
aovrm2way = AnovaRM(gain_rest_pd, 'gain_rest', 'sub_id', within = ['area', 'hemi'])#, aggregate_func = 'mean')
print(gain_rest_pd)

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
gain_rest_pd = pd.DataFrame(gain_rest.T, index = dxn_name).T
gain_rest.shape
len(dxn_name)
sub_id = np.matlib.repmat(np.arange(20), 1, 8)
sub_id.shape
a = np.matlib.repmat('FEF', 2, 1), np.matlib.repmat('IPS', 2, 1)
area = []
for rep1 in range(20):
    for d_name in DXN_Name:
        for rep2 in range(2):
            area  += [d_name]
area.shape

gain_rest_1d = gain_rest.T.ravel()
gain_rest_1d = gain_rest_1d[..., np.newaxis]

a=np.concatenate((sub_id.T, gain_rest_1d), axis=1)
b=a.tolist()
len(b)
c=area + b
len(c)
pd.DataFrame(a)
df = pd.DataFrame(a.T, index = ['sub_id', 'coef']).T# 'coef': gain_rest_1d})

# numpy to list
sub_id_list = sub_id.tolist()
gain_rest_list = gain_rest_1d.tolist()#tuple(map(tuple, gain_rest_1d.T))

A1 = ('sub_id', sub_id_list)
A2 = ('gain_rest', gain_rest_list)
A = [A1, A2]
sub_id_tuple[0]
A = [sub_id_list, gain_rest_list]
t = pd.DataFrame.from_items(A1)
type(A1)
A1[1]

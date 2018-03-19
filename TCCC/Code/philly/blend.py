import numpy as np
import pandas as pd
import time
from time import gmtime, strftime

data_dir = "../"
sub1 = pd.read_csv(data_dir + 'blend_it_all.csv')
sub2 = pd.read_csv(data_dir + 'sub_2018_03_19_09_49_31.csv')
coly = [c for c in sub1.columns if c not in ['id','comment_text']]
#blend 1
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = blend[c] * 0.8 + blend[c+'_'] * 0.2
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]

blend2 = blend
#blend 2
blend2.columns = [x+'_' if x not in ['id'] else x for x in blend2.columns]
blend2 = pd.merge(sub1, blend2, how='left', on='id')
for c in coly:
    blend2[c] = np.sqrt(blend2[c] * blend2[c+'_'])
    blend2[c] = blend2[c].clip(0+1e12, 1-1e12)

time_label = strftime('_%Y_%m_%d_%H_%M_%S', gmtime())
sub_name = "sub" + time_label + ".csv"
blend2[coly + ['id']].to_csv(sub_name, index=False)
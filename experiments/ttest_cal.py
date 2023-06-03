import numpy as np

from scipy.stats import ttest_rel 

with open('../results/metrics/DecisionTreeClassifier()_results.npy', 'rb') as f:
    a = np.load(f)
    c = np.load(f)
    e = np.load(f)

with open('../results/metrics/MLPClassifier()_results.npy', 'rb') as f:
    b = np.load(f)
    d = np.load(f)
    f = np.load(f)

#ttest dla acc score
ttest_val_acc = ttest_rel(a,b)
#ttest dla f1 score
ttest_val_f1 = ttest_rel(c,d)
#ttest dla prec score
ttest_val_prec = ttest_rel(e,f)


with open(f'../results/ttest/results1.npy', 'wb') as f:
    np.save(f, ttest_val_acc)
    np.save(f, ttest_val_f1)
    np.save(f, ttest_val_prec)
import numpy as np
from copy import deepcopy

def merge(*f_names):
    e_base = 0

    dat_train, dat_test = [], []
    for f_name in f_names:
        d = np.loadtxt(f_name + '_train.txt')

        # Add up epochs
        d[-1, :] += e_base
        e_base += d[-1, -1] + 1

        dat_train.append(d.astype(float))
        dat_test.append(np.loadtxt(f_name + '_test.txt'))

    np.savetxt(f_names[0] + '_merge_train.txt', np.concatenate(dat_train, axis=1))
    np.savetxt(f_names[0] + '_merge_test.txt', np.concatenate(dat_test, axis=1))

if __name__ == '__main__':
    merge('2022-04-12-14-42-24', '2022-04-13-00-50-56')
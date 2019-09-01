import os
import pickle
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

def res_stats(results):
    avg = sum(results)/len(results), 
    return avg, avg - min(results), max(results) - avg
    
def main():
    resf = 'results/'
    respkl = [f for f in os.listdir(resf) if '.pkl' in f]
    envnames = [f.split('.')[0] for f in os.listdir('experts/') if '.pkl' in f]
    nenv = len(envnames)
    for i, env in enumerate(envnames):
        plt.subplot(1,nenv,i+1)
        pnum=0
        for res in respkl:
            if env not in res:
                continue
            filename = resf + res
            with open(filename, 'rb') as f:
                results = pickle.loads(f.read())
            if 'batchnorm' in res:
                expname = 'batchnorm'
            elif 'reg_0.001' in res:
                expname = 'l2 regularization'
            else:
                expname = 'default'
            bc_res = [r[-1] for r in results['Behavior Cloning'][20]['returns']]
            da_res = [r[-1] for r in results['Dagger'][1]['returns']]
            bc_avg, bc_down, bc_up = res_stats(bc_res)
            da_avg, da_down, da_up = res_stats(da_res)
            plt.errorbar(pnum, bc_avg, yerr=[[bc_down],[bc_up]], fmt='o', capsize=10, 
                         label='Behavior Cloning '+ expname)
            pnum += 1
            plt.errorbar(pnum, da_avg, yerr=[[da_down],[da_up]], fmt='o', capsize=10, 
                         label='DAgger '+expname)
            pnum += 1

        baseline = results['Behavior Cloning'][5]['baseline_reward']
        b_avg, b_down, b_up = res_stats(baseline)
        plt.errorbar(pnum, b_avg, yerr=[[b_down],[b_up]], fmt='o', capsize=10, 
                     label='expert')
        plt.title(env)
    plt.legend(loc='upper right', bbox_to_anchor=(2.4, 1))
    plt.savefig('results/all_results.png')

if __name__ == '__main__':
    main()
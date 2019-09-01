import os
import pickle
import argparse
import tensorflow as tf
import gym
import seaborn as sns
import matplotlib.pyplot as plt

from policy_model import policy_model
from load_policy import load_policy
from imitation_algorithms import behavior_cloning, dagger

"""
Code for HW1 of Fall 2018 CS229-112 Deep Reinforcement Learning from UC Berkeley
Usage example python run_imitation.py HalfCheetah-v2
"""

def main():
    """
    Runs Behavior Cloning and Dagger on the specified environment
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #tf.keras.backend.set_session(tf.Session(config=config))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('envname', type=str)
    parser.add_argument("--cloning_steps", type=float, default=100e3)
    parser.add_argument("--dagger_steps", type=float, default=100e3)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--batchnorm", action='store_true')
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--outputfolder", default='./results')
    args = parser.parse_args()
    
    envname = args.envname
    f = args.outputfolder
    outfolder =  f + '/' if not f.endswith('/') else f
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    resname = envname + "_lr_%s_reg_%s"%(args.lr, args.l2_reg)
    resname = resname + "_batchnorm" if args.batchnorm else resname
    plotname = outfolder + resname + ".png"
    outfile = outfolder + resname + ".pkl"

    env = gym.make(envname)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    model = policy_model(env.observation_space, env.action_space, learning_rate=args.lr, 
                         l2_reg=args.l2_reg, batchnorm=args.batchnorm)
    expfile = 'experts/' + envname + '.pkl'
    expert_policy = load_policy(expfile)

    verbose_train = True
    bclone_num_steps = int(args.cloning_steps)
    dagger_num_steps = int(args.dagger_steps)
    cloning_rollouts = [2, 5, 20]
    dagger_rollouts = [1, 2, 5]

    with tf.Session(config=config) as sess:
        init_weights = model.get_weights()
        bc_results = {}
        for tr in cloning_rollouts:
            print("Training with %i rollouts of expert data"%tr)
            bc_results[tr] = behavior_cloning(env, model, expert_policy, 
                                              num_steps=bclone_num_steps, num_expert_rollouts=tr,
                                              verbose=verbose_train)
            model.set_weights(init_weights)

        init_weights = model.get_weights()
        da_results = {}
        for tr in dagger_rollouts:
            print("Training with %i rollouts of expert data"%tr)
            da_results[tr] = dagger(env, model, expert_policy, 
                                    num_steps=dagger_num_steps, num_expert_rollouts=tr,
                                    verbose=verbose_train)
            model.set_weights(init_weights)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) # for suppressing seaborn tsplot deprecation warning
    plt.subplot(1, 2, 1)
    plot_imitation_learning(bc_results, "Behavior cloning", "Gradient Steps", resname)
    plt.subplot(1, 2, 2)
    plot_imitation_learning(da_results, "DAgger", "Environment Steps", resname)
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    #plt.show()
    plt.savefig(plotname)
    with open(outfile, 'wb') as f:
        results = {"Behavior Cloning": bc_results,
                   "Dagger": da_results,
                   "Parameters": args}
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    
def plot_imitation_learning(results, algoname, xtext, expname):
    """ Plot on results from imitation learning training """
    colors = "bgrcmyk"
    for i,res in enumerate(results):
        returns = results[res]['returns']
        steps = results[res]['steps']
        sns.tsplot(time=steps, data=returns, color=colors[i%len(colors)], condition=str(res)+" expert rollouts")
    base_rews = results[res]['baseline_reward']
    rews_ts = []
    for i in range(len(base_rews)): 
        rews_ts.append([base_rews[i] for _ in range(len(steps))])
    sns.tsplot(time=steps, data=rews_ts, color=colors[len(results)%len(colors)], condition="expert")
    plt.xlabel(xtext, fontsize=15)
    plt.ylabel("Rewards", fontsize=15)
    plt.title("%s %s"%(expname, algoname))
    plt.legend(loc='best')
    
if __name__ == '__main__':
    main()
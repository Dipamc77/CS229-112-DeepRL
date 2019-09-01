import tensorflow as tf
import numpy as np

""" Imitation algorithms and utility functions for HW1 of CS229-112 """

def behavior_cloning(env, model, expert_policy, num_steps, num_expert_rollouts=20, verbose=True): 
    """
    Execute behavior cloning
    Collect rollouts with expert policy
    Train a policy with all the expert data
    inputs: env - gym environment
            model - keras model that will be trained
            num_steps - number of gradient steps to take on train data
            num_expert_rollouts - number of expert rollouts
            verbose - prints while training
    results: A dictionary with the keys 'train_loss', 'val_loss', 'returns', 
                                        'steps', 'baseline_reward'
             Each value is a list, evaluated after the 'steps' gradient steps
             returns is evaluated on 4 rollouts
    Note: Batch size is fixed as 100, steps per per epoch is 10
    """
    eval_rollouts = 4
    eval_interval = 5000
    results = {'train_loss': [], 'val_loss': [], 
               'returns': [[] for _ in range(eval_rollouts)], # For seaborn 'long-form' list
              'steps': []}
    model_policy = model.predict
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    #Collect training rollouts
    train_obs, train_ac, train_rew = None, None, []
    for r in range(num_expert_rollouts):
        steps, obs, ac, reward = get_rollout(env, expert_policy)
        if train_obs is not None:
            train_obs = np.concatenate((train_obs, obs))
            train_ac = np.concatenate((train_ac, ac))
        else:
            train_obs, train_ac = obs, ac
        train_rew.append(reward)
    average_expert_rew = sum(train_rew)/len(train_rew)
    results['baseline_reward'] = train_rew
    if verbose:
        print("%s - Baseline Reward %f"%(env.spec.id, average_expert_rew))
    _, val_obs, val_ac, _ = get_rollout(env, expert_policy)
    for steps in range(1000,num_steps,1000):
        hist = model.fit(x=train_obs, y=train_ac,
                         batch_size=100, epochs=1,
                         steps_per_epoch = 10, verbose=0, 
                         validation_data=(val_obs, val_ac))
        eval_rews = []
        if steps%eval_interval == 0:
            for e in range(eval_rollouts):
                _, _, _, reward = get_rollout(env, model_policy)
                eval_rews.append(reward)
                results['returns'][e].append(reward)
            avg_rew = sum(eval_rews)/len(eval_rews)
            results['train_loss'].append(hist.history['loss'][0])
            results['val_loss'].append(hist.history['val_loss'][0])
            results['steps'].append(steps)
            if verbose:
                print("Behavior cloning steps", steps, "reward", avg_rew,
                     ",Train loss", results['train_loss'][-1],
                     ",Val loss", results['val_loss'][-1])
    return results


def dagger(env, model, expert_policy, num_steps, num_expert_rollouts=1, verbose=True): 
    """
    Execute DAgger
    Collect rollouts from expert, then collect rollouts from learned policy
    Label the observations of the rollouts with actions from the expert
    inputs: env - gym environment
            model - keras model that will be trained
            expert_policy - expert policy used to generate labels
            num_steps - number of environment interactions with the learned policy
            num_expert_rollouts - number of initial expert rollouts
            verbose - prints while training
    results: A dictionary with the keys 'train_loss', 'val_loss', 'returns', 
                                        'steps', 'baseline_reward'
             Each value is a list, evaluated after every rollout, after 'steps'
             returns is evaluated on 4 rollouts
    Note: Batch size is fixed as 100, steps per epoch is fixed as 10
    """
    eval_rollouts = 4
    eval_interval = 5000
    results = {'train_loss': [], 'val_loss': [], 
               'returns': [[] for _ in range(eval_rollouts)],
               'steps': []}
    model_policy = model.predict
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #Collect training rollouts
    train_obs, train_ac, train_rew = None, None, []
    for r in range(num_expert_rollouts):
        steps, obs, ac, reward = get_rollout(env, expert_policy)
        if train_obs is not None:
            train_obs = np.concatenate((train_obs, obs))
            train_ac = np.concatenate((train_ac, ac))
        else:
            train_obs, train_ac = obs, ac
        train_rew.append(reward)
    average_expert_rew = sum(train_rew)/len(train_rew)
    results['baseline_reward'] = train_rew
    if verbose:
        print("%s - Baseline Reward %f"%(env.spec.id, average_expert_rew))
    _, val_obs, val_ac, _ = get_rollout(env, expert_policy)
    steps = 0
    prev_eval = -1
    while steps < num_steps:
        hist = model.fit(x=train_obs, y=train_ac,
                         batch_size=100, epochs=1, steps_per_epoch=10,
                         verbose=0, validation_data=(val_obs, val_ac))
        if steps//eval_interval == prev_eval+1:
            prev_eval += 1
            rews = []
            for e in range(eval_rollouts):
                _, _, _, reward = get_rollout(env, model_policy)
                rews.append(reward)
                results['returns'][e].append(reward)
            avg_rew = sum(rews)/len(rews)
            results['train_loss'].append(hist.history['loss'][0])
            results['val_loss'].append(hist.history['val_loss'][0])
            results['steps'].append(steps)
            if verbose:
                print("DAgger steps", steps, "reward", avg_rew,
                     ",Train loss", results['train_loss'][-1],
                     ",Val loss", results['val_loss'][-1])
        rollout_steps, model_obs, _, _ = get_rollout(env, model_policy)
        steps += rollout_steps # misses the last rollout for training
        exp_ac = expert_policy(model_obs)
        train_obs = np.concatenate((train_obs, model_obs))
        train_ac = np.concatenate((train_ac, exp_ac))
    return results


def get_rollout(env, policy):
    """ 
    Perform one rollout on env with policy 
    inputs: env - environment
            policy - policy functiion
    returns: steps, observations, actions, rewards
    """
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    observations = []
    actions = []
    while not done:
        action = policy(obs[None,:])
        observations.append(obs)
        actions.append(np.squeeze(action))
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
    return steps, np.asarray(observations), np.asarray(actions), totalr

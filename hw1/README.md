# CS294-112 HW 1: Imitation Learning - General Instructions

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

# Running Imitation Learning Experiments

The results can be reproduced by running ``` run_imitation.py ```

```
python run_imitation.py <ENVIRONMENT-NAME> <ADDITIONAL-ARGUMENTS>
```
For example
```
# Run with default settings - Learning rate 1, 100e3 DAgger environment interactions, 100e3 Behavior Cloning Steps
python run_imitation.py Ant-v2

# Run with 0.5 learning rate, batchnorm and l2 weight regularization for 200e3 steps
python run_imitation.py HalfCheetah-v2 --lr 0.5 --batchnorm --l2_reg 1e-3 --cloning_steps 200e3 --dagger_steps 200e3
```

# Results Analysis

The result plots are stored in folder ```results/``` The ```.pkl``` files contain additional information such as training losses.

Experiments were done with 2, 5 and 20 rollouts taken from expert as the training set for behavior cloning. For DAgger, the initial training sets were taken from 1, 2 and 5 expert rollouts.

![Imitation Learning Results](https://github.com/Dipamc77/CS229-112-DeepRL/blob/master/hw1/results/all_results.png)

In some of the environments, Behavior Cloning is able to achieve nearly equal rewards to the expert policy. Whereas in some environments only DAgger is able to get optimal rewards. Behavior cloning especially fails with environments like Humanoid-v2 which terminate when the human starts falling, which is out of the training distribution as expert policy doesn't fall.

Using L2 Regularization and Batch Normalization seem to be a mixed bag. For some environments batchnorm helps DAgger speed up training, whereas in other case it slows down training. L2 Regularization seems to hinder behavior cloning in most cases, but can help training for the case of higher number of exoert rollouts are part of the training set. More thorough tuning of the parameters may be required.

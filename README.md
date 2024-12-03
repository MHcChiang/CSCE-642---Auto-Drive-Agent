This project focuses on developing an autonomous driving system using Reinforcement Learning (RL). The training environment is based on the CarRacing environment from Gymnasium Box2D. 
The objective is to train a Reinforcement Learning agent capable of navigating along a road and avoiding obstacles. 

This code is based on [stable-baseline3]([https://www.google.com](https://stable-baselines3.readthedocs.io/en/master/index.html))


To run training code, simply run following command after install requirment:
```
python train_SAC_sb3.py  --exp_name test_run 
```
To monitor training, you can use tensorboard by following commnad:
```
tensorboard --logdir path/to/file
```

The usage of other .py files are as follows:

1. eval_sb3_mdl.py: evaluate the model and display the process.
2. eval_mdl_arg.py: same as the former, but with arg parse.
3. plot_log: plot the training logs with matplotlib.
4.  

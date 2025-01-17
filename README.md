# Project Overview

This project aims to develop an autonomous driving system using Reinforcement Learning (RL). The training environment is based on the [CarRacing](https://stable-baselines3.readthedocs.io/en/master/index.html) environment from Gymnasium Box2D, enhanced with the addition of randomly placed cubic obstacles on the road.The primary goal is to train an RL agent capable of efficiently navigating the road while avoiding obstacles, demonstrating robust decision-making in dynamic scenarios.
![Display](https://github.com/user-attachments/assets/c4bbda7c-318e-4cad-af67-3c8420d0c77f)

## observation space
In addition to the original observation, which consists of a top-down 96x96 RGB image, we include additional car measurements in the observation space. These measurements comprise the car’s true speed, four ABS sensor readings, steering wheel position, and gyroscope data. All of these measurements are accessible as they are displayed at the bottom of the image.

# Run 
The training code is implemented using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).
To run training code, simply run following command after install requirment:
```
python train_SAC_sb3.py  --exp_name test_run 
```
To monitor training, you can use tensorboard by following command:
```
tensorboard --logdir path/to/file
```
The resulting agent will be saved at the directory "logs". To evaulate it, run 
```
python eval_mdl_arg.py --mdl "directory of model"
```

![trained_display](https://github.com/user-attachments/assets/4b9c6091-4105-471f-8a88-2ee5b10ce1f1)


# Project Overview

This project aims to develop an autonomous driving system using Reinforcement Learning (RL). The training environment is based on the [CarRacing](https://stable-baselines3.readthedocs.io/en/master/index.html) environment from Gymnasium Box2D, enhanced with the addition of randomly placed cubic obstacles on the road.The primary goal is to train an RL agent capable of efficiently navigating the road while avoiding obstacles, demonstrating robust decision-making in dynamic scenarios.
![Display](https://github.com/user-attachments/assets/c4bbda7c-318e-4cad-af67-3c8420d0c77f)

## observation space
In addition to the original observation, which consists of a top-down 96x96 RGB image, we include additional car measurements in the observation space. These measurements comprise the car’s true speed, four ABS sensor readings, steering wheel position, and gyroscope data. All of these measurements are accessible as they are displayed at the bottom of the image.

# Run 
The training code is implemented using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).

## Steps to Run Training

To run training code, 
1.	Ensure the required dependencies are installed.
2.	Run the training script with the following command:
```
python train_SAC_sb3.py  --exp_name test_run 
```

## Monitoring Training
To monitor the training process, use TensorBoard by running:
```
tensorboard --logdir path/to/file
```

## Evaluating the Trained Agent
The resulting trained agent will be saved in the logs directory. To evaluate the agent, execute the following command with the model directory. For example, to evaluate the model in logs/CarRacing_Example, run
```
python eval_mdl_arg.py --mdl logs/CarRacing_Example/best_model_175000.zip
```

![trained_display](https://github.com/user-attachments/assets/4b9c6091-4105-471f-8a88-2ee5b10ce1f1)


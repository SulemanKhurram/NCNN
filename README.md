############ Prerequisites ######################
Ran and tested on follwing setup
- Python 3.6
- pytorch 0.4.1
- matplotlib 3.0.2
- pillow 5.3.0
- opencv 3.4.2
- numpy 1.14.2
- cudnn 7.1.2

############ Instructions #######################

Following are some of the files that can be run for comparison of three approaches.

- main_Bayes.py
- main_hybrid.py
- main_nonBayes.py

In each of the files, do check for the arguments to select the intended data-set.
According to each data-set selected. Following parameters should be set in the config.py file.
- exp_folder
- trainFile
- testFile 

There are also many other training related hyperparameters in config.py file. That can be set to tweak experiments.

For hybrid approach, you will need a pretrained deterministic feature extractor that can be set using following parameter in the config file.

- hybrid_feature_extractor

give path of the checkpoint of determinstic feature extractor.

Following parameters in config file are for visualization generation purposes.

- viz_base_folder 
- viz_bayes_base_folder 
- viz_test_file 
- viz_checkpoint
- viz_filter 
- viz_layers 

#################### Training, Testing and Evaluation ###############

After activating the desired environment with required packages. Train each approach using above mentioned files. For example

export CUDA_VISIBLE_DEVICES=1
python main_nonBayes.py


#################### Visualization (in testing) ###############

You will need a trained checkpoint to visualize CAM and features of layers and analysis and then after setting up required visualization parameters in config file. 
you can run following file.

- python bayes_viz.py  # for bayesian visualizations
- python nonBayes_viz.py #for deterministic

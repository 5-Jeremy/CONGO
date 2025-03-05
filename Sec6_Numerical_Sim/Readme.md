# Environment and required packages
We have tested this code on Python 3.9.20. The only packages that need to be installed to run these experiments are Numpy,  matplotlib, and pyproximal (along with thier dependencies). We recommend Numpy 1.23.5,  matplotlib 3.6.2, and pyproximal 0.9.0 but any similar versions should work.

# Basic usage
To reproduce one of the plots from Figure 2 in our paper, first determine the experiment number based on the plot to reproduce (the leftmost plot is experiment 1, the center plot is experiment 2, and the rightmost plot is experiment 3). Then, run the script Run_Experiment.py with the following command:

python Run_Experiment.py \<#\>

where <#> is replaced with the experiment number. You should see the text "Begin seed 0" pop up shortly afterwards. As the script runs through the seeds, it will deposit the results of each run for each algorithm in a directory named "trajectories" (which will be created if it does not already exist). These files will be read by the script after the trials are complete in order to generate a figure which will be output to the current working directory.

# Files used from public Github repos
In order to run CONGO-Z and CONGO-E, three files were taken from the Github repository for the paper "Zeroth-order regularized optimization (ZORO): Approximately sparse gradients and adaptive sampling" by Cai et. al. which can be found at https://github.com/caesarcai/ZORO. The files are found in the CONGO_Z_utils folder. The files base.py and Cosamp.py were taken without modification, but the file optimizers.py was modified by us.
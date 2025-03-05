# Environment and required packages
We have tested this code on Python 3.10.12. It depends on several packages, all of which are included in the requirements.txt file. If you have virtualenv installed, you can setup a virtual environment with all the dependencies using the following commands:

virtualenv \<name\> && \<name\>/bin/pip install -r requirements.txt

source \<name\>/bin/activate

# Files used from public Github repos
In order to run CONGO-Z and CONGO-E, three files were taken from the Github repository for the paper "Zeroth-order regularized optimization (ZORO): Approximately sparse gradients and adaptive sampling" by Cai et. al. which can be found at https://github.com/caesarcai/ZORO. The files are found in the CONGO_Z_utils folder. The files base.py and Cosamp.py were taken without modification, but the file optimizers.py was modified by us.

# Running the algorithms
Each algorithm has its own script (named using the corresponding acronym) which runs that algorithm on the specified configuration. There are six configuration files in the conf directory corresponding to the six plots in Figure 4 of the paper. To set which configuration is used, open the desired algorithm's script and replace the config_name argument of the hydra.main decorator. When the script is run, it will create an entry in the outputs directory (which will be created if it does not exist) which contains the results of the run, including a plot of cost vs rounds and a CSV file which can be used for plotting multiple seeds and algorithms. By default the seed is randomly chosen, so to use a specific seed you need to add a command line argument as follows:

python \<alg name\>.py env_config.seed=\<seed\>

If you want to iterate over five seeds like we did to produce our plots, you can use the following command:

array=(0 1 2 3 4); for g in ${array[@]}; do python3 \<alg name\>.py env_config.seed=$g; done

# Plotting
The script plotting_rounds.py was used to generate the plots in Figure 4. Before running it, you must place the CSV files you wish to include in a subdirectory of plotting_scripts called "to_plot". By default the generated plot will not have the title or any of the configuration-specific annotations seen on Figure 4, but you can include those by uncommenting the appropriate lines in plotting_rounds.py
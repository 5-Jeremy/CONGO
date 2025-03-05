<h1 align="center"> CONGO: Compressive Online Gradient Optimization </h1>

This repository contains the code used to run the experiments in the ICLR 2025 paper [CONGO: Compressive Online Gradient Optimization](https://arxiv.org/abs/2407.06325). Each top-level folder in the repository corresponds to a different group of experiments, and each has different software requirements. For instructions on how to set up and run the experiments in each group, see the ReadMe file in the corresponding folder.

# Acknowledgements
In order to implement CONGO-Z, which is built upon the [ZORO](https://arxiv.org/abs/2003.13001) algorithm, we have borrowed some code from the repo for that paper (found [here](https://github.com/caesarcai/ZORO)). Furthermore, the Jackson network simulations make use of the Python package [queueing-tool](https://github.com/djordon/queueing-tool). Finally, we use the social network application from the [DeathStarBench suite](https://dl.acm.org/doi/10.1145/3297858.3304013) to test the performance of CONGO algorithms applied to autoscaling.

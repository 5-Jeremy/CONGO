
# Instructions for running the program

### Pre-requirements

* Docker
* Docker-compose
* Python 3.5+ (with asyncio and aiohttp)
* libssl-dev (apt-get install libssl-dev)
* libz-dev (apt-get install libz-dev)
* luarocks (apt-get install luarocks)
* luasocket (luarocks install luasocket)
* Make sure the following ports are available: port `8080` for Nginx frontend, `8081` for media frontend and `16686` for Jaeger.


### Setting up the social network application
Unzip the DeathStarBench.zip file


#### Before you start

* go into socialNetwork folder

    ```bash
    cd DeathStarbench
    cd socialNetwork
    ```

### Start docker containers
     `docker compose up -d`  
 
**Tip:** use `sudo` or `dzdo` prefix to this command to avoid any permission based issues

### Register users and construct social graphs

Register users and construct social graph by running

`python3 scripts/init_social_graph.py --graph=socfb-Reed98`

**Tip:** python3 issues can resolved by using `python3.x` as instead

It will initialize a social graph from a small social network [Reed98 Facebook Networks](http://networkrepository.com/socfb-Reed98.php)

### Running HTTP workload generator

#### Make

```bash
cd ../wrk2
make
```
back to socialNetwork
```bash
cd ../socialNetwork
```
### Installing jaeger for metric collection
### Follow the below steps if you face an issue with reagards to jaeger
- The jaeger endpoint must be started in the corresponding docker/kubernetes yaml - port number 16685 (by default only port 16686 is exposed).  
- Install the Jaeger stubs:  
    - Clone the Jaeger IDL repo:  
	  `git clone --recurse-submodules https://github.com/jaegertracing/jaeger-idl.git`.    
	- Enter the repo and make the proto files: `cd jaeger-idl; make proto`.  
	- Copy the `proto_gen_python/` directory to the `DeathStarBench/socialNetwork` directory. 

### Now we have successfully installed the socialNetwork microservice application


 
## Running algorithms and generating plots
#### Below are instructions for running 4 algorithms (as discussed in the paper). These codes will output 2 things - a plot of costs vs rounds and csv file storing costs, rounds, timestamps and other relevant information for detailed assessment of performance of algorithm
 
### Implementing Fixed workload with Fixed jobtype scenario
open 2 terminals

#### Terminal one (generating workload)
```bash
cd DeathStarBench
cd socialNetwork
../wrk2/wrk -D exp -t 4 -c 8 -d 120m -L -s ./wrk2/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R 2000
```
#### Terminal two (Running algorithms)

```bash
cd DeathStarBench
cd socialNetwork
cd algos
python3 algo_file.py
```
algo_file should be replaced by any 1 of the 6 algorithms {CONGO-B.py, CONGO-Z.py, CONGO-E.py, NSGD.py, SGDSP.py, PPO.py}
 

### Implementing Fixed workload with Variable jobtype scenario
open 2 terminals

#### Terminal one (generating workload)
```bash
cd DeathStarBench
cd socialNetwork
../wrk2/wrk -D exp -t 4 -c 8 -d 120m -L -s ./wrk2/scripts/social-network/mixed-workload.lua http://localhost:8080/wrk2-api/post/compose -R 2000
```
#### Terminal two (Running algorithms)

```bash
cd DeathStarBench
cd socialNetwork
cd algos
python3 algo_file.py
```
algo_file should be replaced by any 1 of the 6 algorithms {CONGO-B.py, CONGO-Z.py, CONGO-E.py, NSGD.py, SGDSP.py, PPO.py}

### Implementing Variable arrival rate with Fixed jobtype scenario
open 2 terminals

#### Terminal one (generating workload)
```bash
cd DeathStarBench
cd socialNetwork
python3 variable_workload.py
```
#### Terminal two (Running algorithms)

```bash
cd DeathStarBench
cd socialNetwork
cd algos
python3 algo_file.py
```
algo_file should be replaced by any 1 of the 6 algorithms {CONGO-B.py, CONGO-Z.py, CONGO-E.py, NSGD.py, SGDSP.py, PPO.py}

The output of these code is 1 plot (costs vs iteration) and a csv with other relevant information for detailed analysis
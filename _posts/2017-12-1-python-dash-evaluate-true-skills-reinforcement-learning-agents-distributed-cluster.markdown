---
layout: post
title:  "Python Dask: evaluate true skill of reinforcement learning agents with a distributed cluster of instances"
date:   2017-12-01 00:00:51
categories: reinforcement learning
---

Reinforcement learning requires a high number of matches for an agent to learn from a game. Once multiple agents have been trained, evaluating the quality of the agents requires to let them play multiple times against each others.

Microsoft has released and patented a [Bayesian based library](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) named [TrueSkill](http://trueskill.org/) for  Xbox Live that can also be used to compute the skill of agents given the scores they achieved in multi-player matches. The Bayesian theory gives a framework to update the skill value distribution of an agent each time the agent is implied in a game for which a result of the game has been known. Under gaussian assumptions, each update will modify the mean skill value and sharpen the distribution, which means reducing the uncertainty of the new (or a posteriori) skill value. The library is available as a [Python package](http://trueskill.org/).

When the number of agents is important, the number of plays becomes also important. Moreover, agents and games might require lot's of computations, in particular when they are based on deep learning neural networks. Let's see in practice an implementation to distribute the plays of the agents to evaluate their true skills.

To follow this article, the full code can be cloned from [here](https://github.com/christopher5106/distributed-trueskill-eval-of-agents).

### Python package manager

It is a good practice to create a `requirements.txt` file to list the required Python modules to run the code. In this project, we'll use TrueSkill library, as well as Dask for code distribution :

    trueskill
    dask
    distributed
    paramiko

To install the modules on your local computer, run:

```bash
pip3 install -r requirements.txt
```

### Emulate a cluster on the local computer


Let's create a virtual cluster on our local computer using Docker, which enables to launch multiple virtual machines. A local cluster will be tremendously useful to develop fast. For that purpose, we also suppose that the instances have installed the required Python modules.

First, I write a `Dockerfile` to create a Docker image based on Ubuntu 17.04 with an SSH server and the Python modules that are required by my code:

```bash
FROM ubuntu:17.04
RUN apt-get update
RUN apt-get install -y openssh-server vim man
RUN apt-get install -y python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENV HOME=/root
WORKDIR /root/
COPY docker.py docker.py
RUN python3 -c "from docker import setup_ssh; setup_ssh()"
RUN echo "export LC_ALL=C.UTF-8 && export LANG=C.UTF-8" >> ~/.bashrc
COPY . /root/
```

Then, I'm writing a small Python script `docker.py` to run Docker commands
- `docker build . -t distributed` to build the Docker image under the name "distributed"
- `docker run -d -rm distributed /bin/bash -c /etc/init.d/ssh start && while [ ! -f /root/ips.txt ]; do sleep 1s; done && ls -l && cat ips.txt && sleep 60m` to run multiple instances with this image and let them wait for the script to get their IPs
- `docker inspect --format {{ .NetworkSettings.IPAddress }} DOCKER_ID` on each container ID to get their IPs  

The list of IPs is written to `ips.txt` file. From now on, we can use this file, containing the list of hostnames or IPs, to build a generic code that will work wether the cluster is virtual (as here with Docker) or a real cluster of different physical instances.


### Dask, a Python library for distribution

In the past, I wrote many articles about PySpark which is a great library to distribute computations on multiple instances.

Since Spark, there has been some new libraries, and to reduce the overhead of running Python inside Java, directly developed in and for Python.

There is Dask, for example, that extends the Python multiprocessing library, aimed at paralellizing code execution on the multipe cores of a computer. Dask keeps its API very close to the Python multiprocessing library and the PySpark collection methods, which greatly reduces the time to learn it and become familiar with it.

So, to launch a Dask cluster on the provided instances, real or virtual, two options:

The first option is to connect manually to each instance, launch one Dask worker on each of them with `dask-worker` command, and run a Dask scheduler on the first instance with `dask-scheduler` command. Since the image contains the Python modules `dask` and `distributed`, the commands will be available.

The second option is to use `dask-ssh` that will do it for you, given the `ips.txt` file listing the IPs or hostnames of different instances:

```bash
dask-ssh --hostfile ips.txt
```

Now, our Dask cluster is set.


### Build a test game


Let's build a test game, that will enable to verify that the implementation works without bugs, ie :

1- be reliable to failures

2- predict the correct skills



In order to see if the implementation predicts the skills correctly, let's us define a game level to each created agent in a `game.py` file:

```python
class Agent(object):

    def __init__(self):
        self.r = random.randint(0, RATING_RANGE -1)
```

and implement a simple game, in which one could think logical that :

1- a better agent should win more often when playing with a lower agent

2- a better agent should have harder times winning when it was close in level to the lower agent

3- in the case the best agent does not win against the lower one, there might be some tie / drawn matches.

Moreover, there might some unresponsive tasks in a real game evaluation, and we'll need to emulate such failures. Let's use the `sleep` method, random errors (exceptions or invalid results) on top of a stochastic strategy that will not always give the success to the best agent:

```python
def play(self, agent0, agent1):
    sleep(1) # emulate an unresponsive task

    if random.random() < 0.1: # emulate errors with low probability
        raise Exception

    if random.random() < 0.01: # return some non-valid values
        return random.choice(["", -10, 0.5, {}])

    res = int(agent0.r - agent1.r < 0) # result of the match if it was deterministic

    if random.random() < MAX_RAND_PROB - float(abs(agent0.r - agent1.r)) / RATING_RANGE: # add some randomness
        if random.random() < MAX_TIE_PROB - float(abs(agent0.r - agent1.r)) / RATING_RANGE : # when players have close level, return tie game with a certain probability
            return None
        return 1 - res
    else:
        return res
```

The implementation of our distributed framework should deliver a prediction of skill values of the agents, that should be more or less in the same order as the ground truth levels we choose behind the scene.

Note that, once the test game has proved the implementation is correct, any game with the same interface can be used in place of this test game.


### Running the game matches


To submit the game matches to play to the cluster, let's create a `sketch.py` in a Python 3 environment, using Dask API to connect to the cluster:

```python
from dask.distributed import Client
client = Client(scheduler_IP + ':8786')
client.upload_file('game.py')
for _ in range(num_plays):
    jobs.append(client.submit(play, game, agents))
```

Let's run the matches:

```bash
>> python3 sketch.py

Connecting to cluster scheduler 172.17.0.2 with workers:
             tcp://172.17.0.3:46703 8 cores
             tcp://172.17.0.2:34123 8 cores
1000/1000 - Pending: 0, Error: 109, Completed: 891, Elapsed time: 2.96
Game run in 3.97
Skills computed in 4.81
Accuracy of the ratings: 1.0
```

More precisely, the final script `sketch.py` does the following actions:

- it reads the hostname file `ips.txt` that acts as our configuration file. The scheduler is considered to be set on the first instance in the list.

- it prints the cluster configuration, ie available workers, number of threads per workers. In the default settings, one thread is used per core, and a thread pool on each instance.

- it runs the matches in a distributed way connecting to Dask. The elapsed time does not reflect a real game setting, and I had difficulty to emulate a heavy computation game with `sleep` method or operations based on elapsed time since the proc scheduler as well as Dask considers them as non responsive tasks and rotates them with other tasks (that's my current interpretation of the very sublinear growth of computation times). The message can be seen in the Dask logs : `distributed.core - WARNING - Event loop was unresponsive for 2.86s.`

- it computes the skills with the TrueSkill library. We can notice the times to compute these skills confirms us it is not necessary to distribute this computation which will stay very small compared to the game times in a real world setting.

- it estimates the accuracy of the predicted skills. For this estimation, I'm using a very simple algorithm in which I pick all pairs of agents and I check if their skills are aligned with the level they were assigned. I could not simply compare orders since a small error in ordering a pair of players could have an impact on the complete ordering.

To get more information about the parameters to run the code:

```bash
>>> python3 sketch.py --help

usage: sketch.py [-h] [--num-agents NUM_AGENTS] [--num-matches NUM_MATCHES]
                 [--ip-file IP_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --num-agents NUM_AGENTS
                        number of players
  --num-matches NUM_MATCHES
                        number of matches to play
  --ip-file IP_FILE     location of the nodes
```


Let's check that if I use less plays, the accuracy will drop:

```bash
>>> python3 sketch.py --num-matches=10

Connecting to cluster scheduler 172.17.0.2 with workers:
             tcp://172.17.0.2:34123 8 cores
             tcp://172.17.0.3:46703 8 cores
10/10 - Pending: 0, Error: 1, Completed: 9, Elapsed time: 1.03
Game run in 2.03
Skills computed in 0.05
Accuracy of the ratings: 0.8
```

The accuracy is still high due to the way the algorithm computes it: there is lot's of redundant evaluations since all pairs are checked against all in `O(n**2)` while some ordering algorithm only require a complexity of `O(n)` or `O(log(n))`. Anyway, an accuracy of 1 indicates that the order is fully correct.

```bash
>>> python3 sketch.py --num-matches=100

Connecting to cluster scheduler 172.17.0.2 with workers:
             tcp://172.17.0.3:46703 8 cores
             tcp://172.17.0.2:34123 8 cores
100/100 - Pending: 0, Error: 13, Completed: 87, Elapsed time: 0.27
Game run in 1.27
Ratings computed in 0.47
Accuracy of the ratings: 0.91
```

### About my implementation

Please note the following points:

- there might be some other cases to test the distribution reliability, and this comes with experience. For example, a play that never finishes. It is possible in the `check_status()` method to exit after a timeout or when a certain percentage of the matches has finished.

- the reliability of the cluster is the job of `Dask` team

- here, I do not re-submit plays for which there has been a failure. In this case, such failures stay irrelevant, and do not disturb the final skill accuracy computation. But, in a real world setting, it is always important to understand all sources of failures because they might hide bigger problems with a huge impact on the final result.

- if Trueskill were computation costly, or agents heavy to move from one machine to another one, we could have partitionned the game, as in a real world competition, with semi-finals, finals, ... and shuffling the partitions after a few matches inside each partition.

- it is not so easy to test the correctness of the implementation in another way than by implementing a test game, in particular for the tie matches. For example, after a tie game between an agent and himself, at the beginning, when its skill is 25.0, the Trueskill skill becomes 25.000000000000004 or 24.999999999999993.

For me, there are still some open questions, I'm not sure :

- how TrueSkill updates the skills in a tie game / drawn game between an agent and himself, so so I removed this case in my implementation, and

- about the behavior of Dask default distributed scheduler and the Python interpretation in the case of unresponsive tasks (`sleep` method or time-based algorithm). It looks like the Python interpreter has optimized the sequence of instructions of the game, and the pool of threads takes into account such unresponsive tasks. I also tested the resource limitation feature implemented in Dask, by simulating limited resource workers with `dask-worker 172.17.0.2:8786 --resources "GPU=8"` command line, and adding `resources={'GPU': 1}` to the job submission, but this did not help block and wait the scheduler. So, the only way would be to use real costly operations to emulate a heavy game.

Last but not least, it would be interesting to use the quality estimation of a game given by TrueSkill to reduce the number of matches required to estimate the skills. The more likely a match to be a drawn one, the better the match will be to update the skills.

**Well done!**

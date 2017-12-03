---
layout: post
title:  "Python Dash: evaluate true skill of reinforcement learning agents with distributed cluster"
date:   2017-12-01 00:00:51
categories: reinforcement learning
---

Reinforcement learning requires a high number of plays for an agent to learn from a game. Once multiple agents have been trained, evaluating the quality of the agents requires to let them play multiple times as well.

Microsoft has released a Bayesian based library to help us compute the true skill of agents given the scores they achieved against each other. The Bayesian theory gives a framework to update the skill value distribution of an agent each time the agent is implied in a game for which a result of the game has been known. Under gaussian assumptions, each update will modify the mean skill value and sharpen the distribution, which means reducing the uncertainty of the new (or a posteriori) skill value.

When the number of agents is important, the number of plays becomes also important. Moreover, agents and games might require lot's of computations, in particular when they are based on deep learning neural networks. Let's see in practice an implementation to distribute the plays of the agents to evaluate their true skills.


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

Now, your Dask cluster is set.


### Build a test game



Last, the game was not implemented at all, not even a random simulation. My objective was to be able to test that the implementation was working correctly, ie :

1- be reliable to failures

2- predict correct skills

For the first point, I affected a level to each created agent in the `game.py` file:

```python
class Agent(object):

    def __init__(self):
        self.r = random.randint(0, RATING_RANGE -1)
```

and I implemented a simple game, where one could think logical that :

1- a better agent should win more often when playing with a lower agent

2- a better agent should have harder times winning when it was close in level to the lower agent

3- in the case the best agent does not win against the lower one, there might be some tie game / drawn games.

So, here is my implementation, emulating also an unresponsive task with `sleep` method, random errors (exceptions or invalid results) on top of the stochastic strategy:

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

Please note that the interface has not been modified.


### Running the games


From now on, everything happens in my implementation file `sketch.py`.

To submit a task to the cluster, simply run `sketch.py` in a Python 3 environment:

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

The script does the following actions:

- read the hostname file `ips.txt` and consider the scheduler to be on the first instance in the list

- print the cluster configuration, ie available workers, and the number of threads per workers. In the default settings, it is one thread per core, and a thread pool on each instance.

- run the game matches in a distributed way. Here is the elapsed time does not reflect a real game setting, and I had difficulty to emulate a heavy computation game with `sleep` method or other operations based on elapsed time since the proc scheduler as well as Dask will consider non responsive tasks and rotate them with other tasks (that's my current interpretation of the very sublinear growth of computation times). The message can be seen in the Dask logs : `distributed.core - WARNING - Event loop was unresponsive for 2.86s.`

- compute the skills with the proposed TrueSkill library. I might be good to notice that the times to compute these skills is ok and will be very small compared to the times required for the game matches in a real setting, so it is not a priority to distribute theses computations. Also, distributing these computations would require to understand the TrueSkill implementation and check for data transfer it will imply in a distributed version.

- estimage the accuracy of the predicted skills. I used a very simple algorithm in which I pick all pairs of agents and I check if their skills are aligned with the level they were assigned. I do not know much about evaluation of *ratings* but I'm sure there is some scientific literature on this subject. I could not simply compare orders since a small error in ordering a pair of players could have an impact on the complete ordering.

To get more information about the parameters to run the code:

```bash
>>> python3 sketch.py --help

usage: sketch.py [-h] [--num-agents NUM_AGENTS] [--num-matches NUM_GAMES]
                 [--ip-file IP_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --num-agents NUM_AGENTS
                        number of players
  --num-matches NUM_MATCHES
                        number of games to play
  --ip-file IP_FILE     location of the nodes
```


Let's check that if I use less matches, the accuracy will drop:

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

The purpose is not necessary a specification at this time but more an exchange to check I understood correctly the problem.

Please note the following points:

- there might be some other cases to test for reliability, and this comes with experience. For example, a match that never finishes. It is possible in the `check_status()` method to exit after a timeout or when a certain percentage of the matches has finished.

- reliability of the cluster is the job of `Dask` team

- I do not re-submit matches for which there has been a failure. In this case, it does not disturb the final skill accuracy. But, in a real world setting, it is still important to check why they fail because there might hide some bigger problems having a huge impact on the final result.

- if Trueskill were computation costly, or agents heavy to move from one machine to another one, we could have partition the game, as in a real world competition, with semi finals, finals, ... and shuffling the partitions after a few matches inside each partition.

- I'm not sure what it means to have an agent compete against himself, so I removed this case.

- it was not so easy to test the correctness of the implementation in another way than by implementing a test game, in particular for the tie games. For example, after a tie game between an agent and himself, at the beginning, when its skill is 25.0, the Trueskill skill becomes 25.000000000000004 or 24.999999999999993.

### Questions

- I'm new with TrueSkill, and also not the expert about Bayes inferences. I do not understand what quality of a game means in their framework.

- I'm not sure how TrueSkill updates the skills in a tie game / drawn game between an agent and himself...

- I'm not sure about the behavior of Dask default distributed scheduler and the Python interpretation in the case of unresponsive tasks (`sleep` method or time-based algorithm). It looks like the Python interpreter has optimized the sequence of instructions of the game, and the pool of threads takes into account such unresponsive tasks. I also tested the resource limitation, adding `resources={'GPU': 1}` to the job submition and simulating limited resource workers with `dask-worker 172.17.0.2:8786 --resources "GPU=8"` but still without success. So, the only way would be to use real costly operations to emulate a heavy games.

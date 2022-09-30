# Introduction to Reinforcement Learning and Control

## Markov Decision Process (MDP)

A Markov decision process is a tuple $\left( \mathcal{S} ,\mathcal{A} ,\left\{ P_{sa} \right\} ,\gamma ,\mathcal{R} \right)$ :

>$$
\left( \mathcal{S} ,\mathcal{A} ,\left\{ P_{sa} \right\} ,\gamma ,\mathcal{R} \right)
$$
>
>- $\mathcal{S}$ : set of **states**
>- $\mathcal{A}$ : set of **actions**
>- $P_{sa}$ : **state transition probabilities**
>- $\gamma \in \left[ 0,1 \right)$ : **discount factor**
>- $\mathcal{R} :\mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$ : **reward function** (sometimes: $\mathcal{R} :\mathcal{S} \mapsto \mathbb{R}$)

Note: $\sum_{s\prime}{P_{sa}\left( s\prime \right)}=1$

The dynamics of an MDP can be formulated as:
$$
s_0\xrightarrow{a_0}s_1\left( \sim P_{s_0a_0} \right) \xrightarrow{a_1}s_2\left( \sim P_{s_1a_1} \right) \xrightarrow{a_2}s_3\left( \sim P_{s_2a_2} \right)
$$

The **Total Rewards** (return) can be defined as:
$$
R\left( s_0,a_0 \right) +\gamma R\left( s_1,a_1 \right) +\gamma ^2R\left( s_2,a_2 \right) +\cdots
$$
or simply:
$$
R\left( s_0 \right) +\gamma R\left( s_1 \right) +\gamma ^2R\left( s_2 \right) +\cdots
$$

The goal in reinforcement learning is to choose actions over time so as to maximize the *expected value* of the total payoff:
$$
\mathbb{E} \left[ R\left( s_0 \right) +\gamma R\left( s_1 \right) +\gamma ^2R\left( s_2 \right) +\cdots \right]
$$

The reward at time step $t$ is discounted by a factor of $\gamma ^t$.

A **policy** is any function $\pi :\mathcal{S} \mapsto \mathcal{A} $ mapping from the states to the actions.
We say that we are **executing** some policy $\pi$ if, whenever we are in state $s$ , we take action $a=\pi \left( s \right)$ .

## Value Function

For a policy $\pi$, we define the **Value Function** $V^{\pi}:\mathcal{S} \mapsto \mathbb{R}$ as the expected sum of discounted rewards upon starting in state $s$, and taking actions according to $\pi$ :
$$
V^{\pi}\left( s \right) =\mathbb{E} \left[ R\left( s_0 \right) +\gamma R\left( s_1 \right) +\gamma ^2R\left( s_2 \right) +\cdots |s_0=s,\pi \right]
$$

$V^{\pi}$ satisfies the ***Bellman equations*** :
$$
V^{\pi}\left( s \right) =\underset{\begin{array}{c}
	\mathrm{Immediate}\\
	\mathrm{Reward}\\
\end{array}}{\underbrace{R\left( s \right) }}+\gamma \cdot \underset{\mathrm{Expected}\, \mathrm{Future}\, \mathrm{Rewards}}{\underbrace{\sum_{s\prime\in \mathcal{S}}{P_{s\pi \left( s \right)}\left( s\prime \right) V^{\pi}\left( s\prime \right)}}}
$$

The second term above gives the expected sum of discounted rewards obtained after the first step in the MDP.

Note that $s\prime\sim P_{s\pi \left( s \right)}$. In state $s$, we take action $a=\pi(s)$, then:
$$
V^{\pi}\left( s \right) =\mathbb{E} \left[ R\left( s \right) +\gamma V^{\pi}\left( s\prime \right) \right] =R\left( s \right) +\gamma \cdot \mathbb{E}_{s\prime\sim P_{s\pi \left( s \right)}}\left[ V^{\pi}\left( s\prime \right) \right]
$$

We can solve the linear equations for the value function.

We also define the **optimal value function** :
$$
V^*\left( s \right) =\max_{\pi} V^{\pi}\left( s \right)
$$

There is also a version of Bellman’s equations for the optimal value function:
$$
V^*\left( s \right) =R\left( s \right) +\max_{a\in \mathcal{A}} \gamma \cdot \sum_{s\prime\in \mathcal{S}}{P_{sa}\left( s\prime \right) V^*\left( s\prime \right)}
\\
=R\left( s \right) +\gamma \cdot \max_{a\in \mathcal{A}} \mathbb{E} _{s\prime\sim P_{sa}}\left[ V^*\left( s\prime \right) \right]
$$

Therefore, the **optimal policy** can be defined as:
$$
\pi ^*\left( s \right) =\mathrm{arg}\max_{a\in \mathcal{A}} \sum_{s\prime\in \mathcal{S}}{P_{sa}\left( s\prime \right) V^*\left( s\prime \right)}
$$

For every state $s$ and every policy $\pi$, we have:
>$$
V^*\left( s \right) =V^{\pi ^*}\left( s \right) \geqslant V^{\pi}\left( s \right)
$$

## Value Iteration and Policy Iteration

Assume that we know the state transition probabilities $P_{sa}$ and the reward function $R$.

### Value Iteration

Algorithm:
>For each state $s$, initialize $V\left( s \right) \coloneqq 0$
**for** until convergence **do**:
>>For every state, update
$$
V\left( s \right) \coloneqq R\left( s \right) +\max_{a\in \mathcal{A}} \gamma \cdot \sum_{s\prime\in \mathcal{S}}{P_{sa}\left( s\prime \right) V\left( s\prime \right)}
$$

Finally, having found $V^*$, we can then use Equation $\pi ^*\left( s \right) =\mathrm{arg}\max_{a\in \mathcal{A}} \sum_{s\prime\in \mathcal{S}}{P_{sa}\left( s\prime \right) V^*\left( s\prime \right)}$ to find the optimal policy.

- ***Synchronous Updates***: In the first, we can first compute the new values for $V(s)$ for every state $s$, and then overwrite all the old values with the new values. In this case, the algorithm can be viewed as implementing a “**Bellman backup operator**” that takes a current estimate of the value function, and maps it to a new estimate.
- ***Asynchronous Updates***: We would loop over the states in some order, updating the values one at a time.

### Policy Iteration

Algorithm:
>Initialize $\pi$ randomly
**for** until convergence **do**:
>>Let $V\coloneqq V^{\pi}$ (using linear equation solver)
For each state $s$, let
$$
\pi \left( s \right) \coloneqq \mathrm{arg}\max_{a\in \mathcal{A}} \sum_{s\prime}{P_{sa}\left( s\prime \right) V\left( s\prime \right)}
$$

## Learning a Model for an MDP

What if we don't know the state transition probabilities $P_{sa}$ and the reward function $R$? We can learn from the data:
$$
P_{sa}\left( s\prime \right) =\frac{\#\mathrm{times}\, \mathrm{took}\, \mathrm{we}\, \mathrm{action}\, a\,\,\mathrm{in}\, \mathrm{state}\, s\,\,\mathrm{and}\, \mathrm{got}\, \mathrm{to}\, s\prime}{\#\mathrm{times}\, \mathrm{we}\, \mathrm{took}\, \mathrm{action}\, a\,\,\mathrm{in}\, \mathrm{state}\, s}
$$

Here is one possible algorithm for learning in an MDP with unknown state transition probabilities:

>1. Initialize $\pi$ randomly
>2. Repeat:
>
>>a. Execute $\pi$ in the MDP for some number of trials (or *epsilon greedy*)
b. Using the accumulated experience in the MDP, update our estimates for $P_{sa}$ (and $R$, if applicable)
c. Apply value iteration with the estimated state transition probabilities and rewards to get a new estimated value function $V$
d. Update $\pi$ to be the greedy policy with respect to $V$

Exploration V.S. Exploitation

## Continuous state MDPs

### Discretization

Not recommended

### Value Function Approximation

#### Using a Model or Simulator

- Using physics simulation
- Learn one from data collected in the MDP

#### Fitted Value Iteration

Assume that the problem has a continuous state space $\mathcal{S} =\mathbb{R} ^d$, but the action space $\mathcal{A}$ is small and discrete.

Recall in value iteration:
$$
V\left( s \right) \coloneqq R\left( s \right) +\gamma \cdot \max_{a\in \mathcal{A}} \int_{s\prime}{P_{sa}\left( s\prime \right) V\left( s\prime \right) \mathrm{d}s\prime}
\\
=R\left( s \right) +\gamma \cdot \max_{a\in \mathcal{A}} \mathbb{E} _{s\prime\sim P_{sa}}\left[ V\left( s\prime \right) \right]
$$

We will use a supervised learning algorithm (e.g. linear regression) to approximate the value function as a linear or non-linear function of the states:
$$
V\left( s \right) =\theta ^T\phi \left( s \right)
$$

>1. Randomly sample $n$ states: $s^{\left( 1 \right)},s^{\left( 2 \right)},\cdots ,s^{\left( n \right)}\in \mathcal{S}$
>2. Initialize $\theta \coloneqq 0$
>3. Repeat
{
&emsp;&emsp;	For $i=1,2,\cdots ,n$
&emsp;&emsp;	{
&emsp;&emsp;&emsp;&emsp;	For each action $a\in \mathcal{A}$
&emsp;&emsp;&emsp;&emsp;		{
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;			Sample $s_1\prime,\cdots ,s_k\prime\sim P_{s^{\left( i \right)}a}\,\,\left( \mathrm{using}\, \mathrm{a}\, \mathrm{model}\, \mathrm{of}\, \mathrm{MDP} \right)$
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;			Set $q\left( a \right) =\frac{1}{k}\sum_{j=1}^k{R\left( s^{\left( i \right)} \right) +\gamma V\left( s_j\prime \right)}$
&emsp;&emsp;&emsp;&emsp;		}
&emsp;&emsp;&emsp;&emsp;		Set $y^{\left( i \right)}=\max_a q\left( a \right)$
&emsp;&emsp;	}
&emsp;&emsp;	Set $\theta \coloneqq arg\min_{\theta} \frac{1}{2}\sum_{i=1}^n{\left( \theta ^T\phi \left( s^{\left( i \right)} \right) -y^{\left( i \right)} \right) ^2}$
}



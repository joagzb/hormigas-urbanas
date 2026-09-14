# Urban Ants - A map routing algorithm

Created: June 24, 2024 8:42 PM
technologies: Python
Website: https://github.com/joagzb/hormigas-urbanas

**Table of Contents**

🔧 **Technologies Used**

| Technology | Purpose                  |
| ---------- | ------------------------ |
| Python     | scripting, data analisys |

[GitHub - joagzb/hormigas-urbanas: Encontrar el camino mas rápido y menos costoso entre dos puntos de una ciudad, utilizando colectivos públicos.](https://github.com/joagzb/hormigas-urbanas)

# ⚡ Summary

---

## ❓ Case of study

The movement of people relies on a variety of transportation options, such as public transport, cars, motorcycles, and bicycles. Navigating large and medium cities, sometimes, can be difficult, for those unfamiliar with the city streets and traffic patterns.

Effective transportation planning becomes indispensable in such scenarios. Tools like Google Maps are very helpful to users for planning their journeys and reducing time and cost. Behind these user-friendly interfaces lie sophisticated algorithms designed to calculate the optimal routes between destinations. Efforts to optimize urban travel continue to evolve, driven by the need for efficient and sustainable mobility solutions.

## 🎯 Goal

The goal of this project is to compare routes between two points using a static generalized-cost score and considering urban buses and walking. The current weights are not calibrated minutes or converted fares, and the model does not include schedules or headways.

# 📸 Media Showcase

---

[Untitled](Untitled%20aba4a2d58f7d458aa4a534ac222b2db3.csv)

# 📁 Source Code and Documentation

---

| Resource              | Link                                       |
| --------------------- | ------------------------------------------ |
| **GitHub Repository** | https://github.com/joagzb/hormigas-urbanas |

# 🔎 Detailed Algorithmic & Mathematical Specifications

---

The project uses Collective Intelligence methods inspired by how ants find food efficiently. This approach, known as Ant Colony Optimization (ACO), is adapted to help find the best routes in cities. 

Ant Colony Optimization (ACO) draws its inspiration from the collective behavior of ants. When ants forage for food, they communicate through pheromone trails, collectively finding the shortest path to food sources. This natural behavior serves as the foundation for ACO algorithms in solving complex optimization problems.

In the context of our project, we applied ACO to urban navigation. Just as ants optimize their foraging routes, ACO algorithms compare paths through the city's walking and bus edges using the configured generalized-cost score.

This prototype compares walking and bus routes with several ACO variants. Bus travel is represented by separate outbound and inbound directed services. Onboard nodes have opaque IDs containing line, direction, and stop sequence, while aligned edge metadata distinguishes `walk`, `board`, `ride`, and `alight` transitions.

The urban navigation problem is modeled on a weighted directed graph $G = (V, E)$, where $V$ represents intersections, bus stops, or transfer hubs, and $E$ represents directed street or transit connections.

Path construction by artificial ants relies on two complementary guidance mechanisms:

Heuristic Information ($\eta_{ij}$): A static problem-specific metric calculated a priori. In this multimodal experiment, explicit edge types adjust bus boarding and alighting weights only during selection so ants do not leave the bus immediately. Route evaluation and pheromone deposits still use the original edge generalized cost $d_{ij}$.

$$\eta_{ij} = \frac{1}{\tilde d_{ij}}$$

where $\tilde d_{ij}$ is the selection-only surrogate and the reported route cost remains the sum of original $d_{ij}$ values.

Pheromone Information ($\tau_{ij}$): A dynamic memory matrix stored across graph edges, updated iteratively based on the collective search experience of the foraging artificial ant colony.

## Ant Colony Optimization Baseline 

All the tested Ant Colony algorithms share the same basic idea:

At the beginning, the behavior of the ants is **random**. Each ant decides whether to follow a path based on the **amount of pheromone** detected on each alternative. When an ant finds a **food source** (in our problem, this means it **reaches the destination**), it returns and **deposits pheromone** on the path it used.

The amount of pheromone deposited is **proportional to the quality of the solution**. In this implementation, the deposit is **inversely proportional to the route's static generalized-cost score**:

- Lower score → more pheromone
- Higher score → less pheromone

After this, the other ants are influenced by the reinforced pheromone trail. This increases the **probability that they will follow the same path**. As more ants pass through the same route, the pheromone trail becomes stronger. At the same time, the algorithm simulates **pheromone evaporation**, just like in nature, to avoid getting stuck in poor solutions.

There are two main pheromone update strategies:

#### **Global Update (Offline Update)**

After **all ants have completed their solutions**, pheromone levels are updated on each path based on the **quality of each solution**. This is called: **Offline pheromone update** Or **global pheromone update**

#### **Local Update (Online Update)**

A small amount of pheromone is deposited **while the ants are building their solution**, step by step. This encourages **exploration during the construction process**.

The decision of doing the next step and where, depends on both:

- **Pheromone trails**
- **Heuristic information (time, distance, cost, etc.)**

Two extreme cases help to understand this balance:

- If **pheromone influence is set to zero**, the algorithm becomes similar to a **probabilistic greedy algorithm**, guided only by heuristic information.
- If **heuristic influence is set to zero**, the algorithm uses only pheromones. This can cause **fast stagnation**, where all ants follow the same path too early.

### **Initial setup**
The standard Ant System serves as the foundational variant. Initial pheromone concentrations $\tau_0$ are set uniformly across all edges $E$ based on a deterministic depth-first/backtracking baseline route cost $L_{gb}$:

$$\tau_0 = \frac{m}{L_{gb}}$$

where $m = \vert{}V\vert{}$ represents the total number of vertices or artificial ants.

### **Iterative solution Overview**
so, at each iteration:

1. Ants construct a solution step by step.
2. Each move is based on a **probabilistic decision rule**.
3. Once all ants finish, pheromone levels are updated based on solution quality.
4. Pheromone evaporates over time.
5. The search gradually focuses on the **most promising regions**.

### **Termination**

Simple ACO stops after 10 consecutive completed epochs without a strict finite global-best cost improvement; the first finite best and every later strict improvement reset its configurable patience counter. ACS and BWAS stop only after a completed epoch when both conditions hold: at least $\lceil t n_f \rceil$ of the $n_f$ finite ants completed the same exact route, and the finite iteration-best cost is exactly unchanged from the preceding epoch. Their configured threshold is $t = 0.85$. Lost or infinite ants are excluded from $n_f$, while equal-cost alternative routes remain distinct. No algorithm can stop before epoch 2, `max_epochs` remains the hard cap, and Dijkstra is used only as an external reference.

### **Global Pheromone Deposit Formula**
Once all $m$ ants complete their complete paths, graph-wide evaporation occurs across all edges $(i, j) \in E$:

$$
τ_{ij}=(1-p)τ_{ij}+ \sum Δτ_{ij}^k
$$

Where:

- $τ_{ij}$ = pheromone on edge between nodes $i$ and $j$
- ρ∈[0,1] → global evaporation rate
- $Δτ_{ij}^k$  → pheromone deposited by ant k at node $i$.

Where the Pheromone Deposit by One Ant is


$$
\Delta \tau_{ij}^k = \begin{cases} \frac{Q}{L_k} & \text{if edge } (i,j) \text{ belongs to path } T^k \\ 0 & \text{otherwise} \end{cases}
$$

### **Transition Rule**

An ant $k$ moves from node $i$ to node $j$ with probability:

$$
p_{ij}^k=\frac {(τ_{ij}^α)(η_{ij}^β)}{\sum_k(τ_{ik}^α)(η_{ik}^β)}
$$

Where:

- $τ_{ij}$ = pheromone level between nodes $i$ and $j$
- $\eta_{ij}$ = heuristic value derived from the edge's selection-only generalized cost
- α = pheromone importance. Controls the influence of accumulated historical pheromone trails.
- β = heuristic importance. controls the influence of local heuristic desirability.

### **Heuristic Information**

$$
η_{ij}=\frac {1}{\tilde d_{ij}}
$$

- $\tilde d_{ij}$ = selection-only edge weight; route cost still uses the original $d_{ij}$
- $\eta_{ij}$ = heuristic desirability


## Ant Colony System (ACS)

ACS combines an exploitation-biased transition rule with local trail updates during solution construction. These mechanics are implemented, but this project has not established a convergence-rate guarantee.

### Initialization

All edges are initialized to a baseline trail intensity $\tau_0 = \frac{1}{\vert{}V\vert{} \cdot L_{gb}}$.

Pseudo-Random Proportional Rule

An ant at node $i$ generates a uniform random variable $q \sim U(0, 1)$ and compares it against a threshold $q_0 \in [0, 1]$. The current project configuration uses $q_0 = 0.9$, giving a 90% exploitation probability:

$$j = \begin{cases} \arg\max_{l \in N_i^k} \left\{ \tau_{il} \cdot [\eta_{il}]^\beta \right\} & \text{if } q \le q_0 \quad \text{(Exploitation)} \\ J & \text{if } q > q_0 \quad \text{(Exploration)} \end{cases}$$

where $J$ is a random variable sampled according to the standard AS probability distribution $p_{iJ}^k$ with $\alpha = 1$.

### Local Pheromone Update (Online Decay)

When ant $k$ traverses edge $(i, j)$ during path construction, it immediately attenuates the edge's pheromone trail:

$$\tau_{ij} \leftarrow (1 - \xi) \tau_{ij} + \xi \tau_0$$

where $\xi \in (0, 1]$ is the local pheromone decay parameter. This intra-iteration self-repulsion mechanism lowers the edge's attraction for subsequent ants in the same iteration, forcing the colony to explore alternative paths.

### Global Pheromone Update

Global update is applied only to the edges belonging to the global-best path $T^{gb}$:

$$\tau_{ij} \leftarrow (1 - \rho) \tau_{ij} + \rho \Delta \tau_{ij}^{gb} \quad \forall (i,j) \in T^{gb}$$

$$\Delta \tau_{ij}^{gb} = \frac{1}{L_{gb}}$$

## Best-Worst Ant System (BWAS)

BWAS incorporates principles from Evolutionary Computation—specifically Population-Based Incremental Learning (PBIL)—introducing explicit negative reinforcement, dynamic pheromone mutation, and search restarts.

### Global Best-Worst Update Rule

After applying standard evaporation $\tau_{ij} \leftarrow (1 - \rho) \tau_{ij}$ across all edges:

**Positive Reinforcement**: Pheromone is added to the global-best tour $T^{gb}$:

$$\tau_{ij} \leftarrow \tau_{ij} + \Delta \tau_{ij}^{gb} \quad \forall (i,j) \in T^{gb}$$

**Negative Reinforcement (Worst-Path Penalization)**: For the current iteration's worst tour $T^{worst}$, any edge $(i, j) \in T^{worst}$ that is not part of $T^{gb}$ receives an extra evaporation penalty:

$$\tau_{ij} \leftarrow (1 - \rho_w) \tau_{ij} \quad \forall (i,j) \in T^{worst} \setminus T^{gb}$$

### Dynamic Pheromone Mutation

To perturb stagnant search trails, each matrix row $i$ undergoes stochastic mutation with probability $P_{mut}$:

$$\tau_{ij}' = \max(\tau_{min}, \tau_{ij} \pm s \cdot \sigma \cdot \bar{\tau}_{gb})$$

where:

$z \in \{0, 1\}$ is a uniform binary random variable.

$\bar{\tau}_{gb}$ is the average pheromone intensity across edges of $T^{gb}$.

$s$ is the configured mutation scale, and $\sigma$ grows with search progress since the last restart:

$$\sigma = \frac{i_{current} - i_{restart}}{i_{max}}$$

## Hard Restart Mechanism

If $T^{gb}$ fails to improve for 8 iterations, all matrix entries are reset to $\tau_0$, clearing accumulated bias while preserving the best score found. This is a pheromone-diversity restart, not a terminal condition; BWAS retains the same consensus and stable-cost termination rule as ACS.

# 🚦 Project Status and Updates

---

| Status                | Description                    |
| --------------------- | ------------------------------ |
| **Prototype**         | ACO, ACS, and BWAS implementations with automated tests |
| **Cost model**        | Static generalized-cost weights; no schedules or calibrated travel-time claims |

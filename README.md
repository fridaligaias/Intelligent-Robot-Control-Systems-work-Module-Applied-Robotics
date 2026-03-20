# An Exploration of Autonomous Robot Navigation using Reinforcement Learning

**Fridalí Gaias · w1905050 · January 2026**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Reinforcement Learning Controller (Tasks 1 & 3)](#2-reinforcement-learning-controller-tasks-1--3)
   - [Algorithm Implementation](#21-algorithm-implementation)
   - [Results (Task 1)](#22-results-task-1)
   - [Task 3: File-Based Maze Solving](#23-task-3-file-based-maze-solving)
3. [Neural Network Controller (Task 2)](#3-neural-network-controller-task-2)
   - [Initial Development and Challenges](#31-initial-development-and-challenges)
   - [Optimisation and Parameter Tuning](#32-optimisation-and-parameter-tuning)
   - [Final Evaluation](#33-final-evaluation)
4. [Conclusion](#4-conclusion)
5. [References](#5-references)
6. [Appendix](#6-appendix)

---

## 1. Introduction

The objective of this coursework is to design and implement intelligent control algorithms for a mobile robot operating in an unknown grid-based environment. The robot navigates from an arbitrary starting position $I$ to a target goal $G$ while avoiding obstacles.

Two distinct approaches were implemented and evaluated:

1. **Reinforcement Learning (RL):** A Q-Learning algorithm implemented from scratch to learn optimal policies through interaction with the environment.
2. **Neural Network (NN):** A Multi-Layer Perceptron (MLP) trained using Supervised Learning to predict optimal actions based on the robot's coordinates.

---

## 2. Reinforcement Learning Controller (Tasks 1 & 3)

### 2.1 Algorithm Implementation

The Q-Learning algorithm was implemented without external libraries, extending the basic concepts from the module materials. The module code provided a basic Q-learning structure for a fixed graph; this was extended into a dynamic `MazeEnvironment` class using NumPy arrays to represent a 6×6 grid.

- **State space:** Adapted from symbolic dictionary keys to numerical coordinate tuples `(row, col)`.
- **Environment coordinates:** The initial state I at visual row 4, col 3 was converted to 0-based indexing as `(3, 2)`; goal G at row 2, col 6 became `(1, 5)`.
- **Collision detection:** If a move leads to an obstacle (`grid[r,c] == 1`), the robot remains in the current state.

The core Q-table update rule applied is:

$$
Q(s, a) = r + γ · max Q(s', a')
$$

**Reward structure:**

| Event | Reward |
|-------|--------|
| Goal state reached | +100 |
| Step / obstacle collision | 0 |

A sparse reward structure combined with a discount factor of $γ = 0.9$ encourages shortest-path solutions — longer paths accumulate more discounting, yielding lower Q-values at the start state.

**Sample output:**

```
Maze initialised. Number of free cells: 28
Task 1 config: start at I: (3, 2), goal at G: (1, 5)
Generating CLEAN dataset for 28 possible goal positions...
Dataset generation is done. Total samples: 756

--- Solving I ---> G ---
Tracing path from (3, 2) to (1, 5)...
Path taken: [(3,2),(4,2),(4,3),(5,3),(5,4),(5,5),(4,5),(3,5),(2,5),(1,5)]
Result: Success! Goal has been reached!!
X Shape: (756, 4), y Shape: (756,)
```

### 2.2 Results (Task 1)

The Q-Learning controller successfully converged on the optimal path.

| Field | Value |
|-------|-------|
| Start | `(3, 2)` |
| Goal | `(1, 5)` |
| Path length | 9 steps |
| Outcome | ✅ Success |

**Path taken:** `(3,2) → (4,2) → (4,3) → (5,3) → (5,4) → (5,5) → (4,5) → (3,5) → (2,5) → (1,5)`

The agent successfully navigated around local obstacles, demonstrating that the Q-table correctly propagated the goal reward back to the start state.

### 2.3 Task 3: File-Based Maze Solving

The system was extended to parse maze configurations from `.txt` files, correctly interpreting dimensions, start/goal coordinates, and the obstacle grid.

**Example output (`maze_config.txt`):**

```
Maze loaded successfully!
Maze dimensions: 6x5
Start position: (0, 1)
Goal position: (2, 4)
Running Q-Learning on File Maze...
Tracing path from (0, 1) to (2, 4)...
Path: [(0,1),(0,2),(0,3),(0,4),(1,4),(2,4)]
Success! The maze provided by the txt file has been solved!
```

The algorithm was successfully tested on further maze configurations to demonstrate robustness (see [Appendix](#6-appendix)).

---

## 3. Neural Network Controller (Task 2)

### 3.1 Initial Development and Challenges

The Neural Network was implemented using `MLPClassifier` from Scikit-learn to predict the optimal action (N, E, S, W) given the robot's current position and goal coordinates.

`MLPClassifier` was chosen because the task involves selecting one of four **discrete** actions — a regressor would incorrectly treat directions as ordered numerical values.

**Initial architecture:**
```python
MLPClassifier(
    hidden_layer_sizes=(20, 20),
    max_iter=1000
)
```

**Initial performance:** ~73.68% accuracy. When testing the path from I`(3,2)` to G`(1,5)`, the controller failed completely — the robot looped indefinitely at the start position. The network correctly identified the general goal direction (North-East) but failed to account for immediate obstacles.

### 3.2 Optimisation and Parameter Tuning

Two targeted optimisations were applied:

1. **Dataset filtering** — added a condition to exclude states where the Q-table had no learned values:
   ```python
   if np.max(Q_table[sx, sy, :]) != 0:
   ```

2. **Increased network capacity:**
   ```python
   mlp = MLPClassifier(
       hidden_layer_sizes=(100, 100),  # Increased from (20, 20)
       max_iter=5000                   # Increased from 1000
   )
   ```

### 3.3 Final Evaluation

| Metric | Before | After |
|--------|--------|-------|
| Test Accuracy | 73.68% | 78.29% |
| Training Accuracy | — | 82.95% |
| Train/Test Gap | — | <5% |

The small gap between training and test accuracy confirms the model generalised without overfitting.

**Persistent issues identified from the confusion matrix:**
- The model confused **North** and **East** — when the optimal action was East, it predicted North 7 times.
- **East** had the lowest precision (0.61); **South** had the highest (0.97).

Despite the accuracy improvement, the robot still failed to complete the I→G path:

```
--- Testing Neural Network Robot on I ---> G Path ---
Neural Network Path from (3, 2) to (1, 5)...
Path taken by NN: [(3,2),(3,2),(3,2),(3,2),(3,2) ... (3,2)]
Failure... Unfortunately, the Neural Network has failed to reach the goal...
```

The model's North bias meant the robot repeatedly collided with the obstacle above `(3,2)` and remained stuck. This confirms that while NNs generalise well, they **lack the specific state awareness** required to navigate obstacle traps in this problem.

---

## 4. Conclusion

| Controller | Outcome | Notes |
|------------|---------|-------|
| Q-Learning (RL) | ✅ 100% success | Optimal path found; generalises to file-based mazes |
| Neural Network | ❌ Failed navigation | 78.29% accuracy but unable to pass obstacle at `(3,2)` |

The Reinforcement Learning approach proved superior for this static, discrete environment. Its ability to fully map the state space allowed it to find the most efficient path in both the default maze and the file-based Task 3 configurations.

The Neural Network's failure can largely be attributed to:
- The **discrete, grid-based** nature of the environment.
- Using raw **Cartesian coordinates (x, y)** as inputs, which provide limited spatial context.
- A **North prediction bias** that causes the robot to collide repeatedly with the same obstacle.

That said, the fundamental strength of NNs lies in scalability: unlike Q-Learning (which requires visiting and storing every state, suffering from the curse of dimensionality in larger environments), a NN learns an underlying input-output function and can generalise to unseen states. In **continuous state-space problems** such as autonomous driving, a NN-based approach is far more suitable than tabular Q-Learning.

---

## 5. References

- Dracopoulos, D. (2025) *6ELEN018W Applied Robotics Coursework Specification 2025/2026*. University of Westminster.
- Dracopoulos, D. (2025) *Lecture 8: Robot Control — Intelligent Control Algorithms*, 6ELEN018W Applied Robotics. University of Westminster, unpublished.
- Dracopoulos, D. (2025) *Tutorial 9: Reinforcement Learning Exercises*, 6ELEN018W Applied Robotics. University of Westminster, unpublished.
- Scikit-learn Developers (2024) *Neural Network Models (Supervised)*. Available at: [scikit-learn.org](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) (Accessed: December 2025).

---

## 6. Appendix

### maze_2.txt

```
Reading maze from maze_2.txt... please stand by...
Maze loaded successfully!
Maze dimensions: 6x6
Start position: (0, 0)
Goal position: (4, 4)

Running Q-Learning on File Maze...
Tracing path from (0, 0) to (4, 4)...
Path: [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1),(4,2),(4,3),(4,4)]
Success! The maze provided by the txt file has been solved!
```

### maze_3.txt

```
Reading maze from maze_3.txt... please stand by...
Maze loaded successfully!
Maze dimensions: 10x10
Start position: (0, 0)
Goal position: (3, 8)

Running Q-Learning on File Maze...
Tracing path from (0, 0) to (3, 8)...
Path: [(0,0),(0,1),(0,2),(0,3),(0,4),(1,4),(2,4),(2,5),(2,6),
       (1,6),(0,6),(0,7),(0,8),(0,9),(1,9),(2,9),(3,9),(3,8)]
Success! The maze provided by the txt file has been solved!
```

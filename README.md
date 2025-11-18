# Q-Learning

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
→ Initialize Q-table and hyperparameters.
→ Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.
→ After training, derive the optimal policy from the Q-table.
→ Implement the Monte Carlo method to estimate state values.
→ Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION

```

Name: Chandrapriyadharshini C
Register Number: 212223240019
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
          action = select_action(state, Q, epsilon[e])
          next_state, reward, done, _ = env.step(action)
          td_target = reward + gamma * Q[next_state].max() * (not done)
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track

```
    
## OUTPUT:

#### Optimal State Value Functions:

<img width="571" height="304" alt="image" src="https://github.com/user-attachments/assets/8272fe99-ea48-4aa1-9331-0989a9451403" />

#### Optimal Action Value Functions:

<img width="970" height="624" alt="image" src="https://github.com/user-attachments/assets/e836af1c-8410-44b8-87e2-0eadcbf4383d" />

#### State value functions of Monte Carlo method:

<img width="1461" height="692" alt="image" src="https://github.com/user-attachments/assets/76abaf0d-a59d-4a35-a9a5-1e86963928fc" />


#### State value functions of Qlearning method:

<img width="1461" height="698" alt="image" src="https://github.com/user-attachments/assets/87d80e3b-ba1b-4f39-b57d-d94b88375552" />


## RESULT:

Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.

import numpy as np

maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 2]
])
start = (0, 0)
goal  = (4, 4)
max_steps = 100
episodes = 200

q_table = np.zeros((maze.shape[0]*maze.shape[1], 4))
learning_rate = 0.1
gamma = 0.9
epsilon = 0.1

def state_to_index(state):
    return state[0]*5 + state[1]

def step(state, action):
    x, y = state
    if action == 0:
        next_x, next_y = x-1, y
    elif action == 1:
        next_x, next_y = x+1, y
    elif action == 2:
        next_x, next_y = x, y-1
    else:
        next_x, next_y = x, y+1

    if (next_x < 0 or next_x >= 5 or next_y < 0 or next_y >= 5 or maze[next_x, next_y] == 1):
        return state, -5
    
    next_state = (next_x, next_y)
    if next_state == goal:
        reward = 10
    else:
        reward = -1
    
    return next_state, reward

def get_action(state):
    if np.random.rand() < epsilon:
        action = np.random.randint(4)
    else:
        action = np.argmax(q_table[state_to_index(state)])

    return action

def learn(state, action, next_state, reward):
    q_value = q_table[state_to_index(state), action]
    next_max = np.max(q_table[state_to_index(next_state)])
    q_table[state_to_index(state), action] = q_value + learning_rate*(reward + gamma*next_max - q_value)


for _ in range(episodes):
    state = start
    for _ in range(max_steps):
        action = get_action(state)
        next_state, reward = step(state, action)

        learn(state, action, next_state, reward)

        state = next_state
        if state == goal:
            break

state = start
step_counter = 0

maze[start] = 7
while state != goal and step_counter < max_steps:
    action = np.argmax(q_table[state_to_index(state)])
    state, _ = step(state, action)
    step_counter += 1
    maze[state] = 7

print(maze)
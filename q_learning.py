import gym
import numpy as np
import random

def learn_q_table(
    environment,
    learning_rate,
    discount_factor,
    random_factor,
    num_episodes=100000,
    max_time_steps=1000,
    logging_interval=500,
    save_file="saved_q_table"
):
    env = gym.make(environment)

    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))

    # Build Q-table, with 1 row for every state and 1 column for every action
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # keep track of average score and average number of timesteps our agent takes to complete the game
    total_score = 0
    total_timesteps = 0

    for episode in range(num_episodes):
        curr_state = env.reset()
        next_state = None
        score_for_episode = 0

        for t in range(max_time_steps):
            env.render(mode='ansi')

            if (random.uniform(0, 1) < random_factor):
                action = env.action_space.sample() # random action
            else:
                # returns the max value action from the q_table
                action = np.argmax(q_table[curr_state])

            next_state, reward, done, info = env.step(action)

            # update q_table entry with q learning update rule
            q_table[curr_state, action] = (1 - learning_rate) * q_table[curr_state, action]
            update = learning_rate * (reward + (discount_factor * np.max(q_table[next_state])))
            q_table[curr_state, action] += update

            curr_state = next_state
            score_for_episode += reward

            if done:
                break
        total_score += score_for_episode
        total_timesteps += t

        # log metrics as we learn the q table
        if (episode % logging_interval == 0):
            log_metrics(logging_interval, total_score, total_timesteps, episode)
            total_score = 0
            total_timesteps = 0

	np.save(save_file, q_table)    

    print (q_table)
    env.close()
    return q_table

def log_metrics(logging_interval, total_score, total_timesteps, episode=None):
    print("========")

    if (episode):
        print("Finished episode #{}".format(episode))
    print("Stats for last {} episodes:".format(logging_interval))
    print("Average score: {}".format(total_score / logging_interval))
    print("Average timesteps: {}".format(total_timesteps / logging_interval))

    print("========")

def play_game(
    environment,
    q_table=[],
    num_episodes=1000,
    max_time_steps=1000,
    mode='ansi',
    ai='random'
):
    print ("Playing {} with {} AI".format(environment, str(ai)))
    env = gym.make(environment)
    # keep track of average score and average number of timesteps our agent takes to complete the game
    total_score = 0
    total_timesteps = 0

    for episode in range(num_episodes):
        curr_state = env.reset()
        score_for_episode = 0

        for t in range(max_time_steps):
            env.render(mode=mode)

            # decide action to take:
            # if we have a q_table, take the action the q_table says is best
            # otherwise follow a different policy
            if (ai == 'q_learning'):
                action = np.argmax(q_table[curr_state])
            elif (ai == 'random'):
                action = env.action_space.sample() # fallback ai, take random action
            else: 
                action = ai(curr_state) # custom ai for game
                

            curr_state, reward, done, info = env.step(action)

            if (mode == 'human'):
                print("move taken: {}".format(action))
            score_for_episode += reward

            if done:
                break
        
        if (mode == 'human'):
            print("score for episode: {}".format(score_for_episode))

        total_score += score_for_episode
        total_timesteps += t

    log_metrics(num_episodes, total_score, total_timesteps)

    # return average score
    return (total_score / num_episodes)

# greedy bot that simulates human behavior
# makes moves that go closer to the goal without heading towards a hole or out of bounds
def greedy_4x4_bot(curr_state):
    holes = [5, 7, 11, 12]
    # get the row and column position of our agent in the grid
    row = curr_state // 4
    col = curr_state % 4

    # check if we can move down without falling in a hole or going out of bounds
    if ((curr_state + 4) not in holes and (row + 1 < 4)):
        action = 1 # move down
    elif ((curr_state + 1) not in holes and (col + 1 < 4)):
        action = 2 # move right
    elif ((curr_state - 1) not in holes and (col - 1 >= 0)):
        action = 0 # move left
    else:
        action = 3 # move up 

    return action

# learn the q_tables
# uncomment to learn new q tables. make sure to comment out hard_codes q_tables below
# q_table = learn_q_table('FrozenLake-v0', learning_rate=0.1, discount_factor=1, random_factor=0.1)
# q_table_discounted = learn_q_table('FrozenLake-v0', learning_rate=0.1, discount_factor=0.95, random_factor=0.1)

# learned q_table with no discount after 100k episodes
# uncomment if you don't want to train
q_table = np.array([[0.38689906, 0.37441442, 0.37631358, 0.36515488],
 [0.24115092, 0.19194264, 0.26814398, 0.31177541],
 [0.24655721, 0.24832002, 0.24346735, 0.25516114],
 [0.18785431, 0.14927093, 0.1673444 , 0.20294013],
 [0.41103298, 0.27772682, 0.28815905, 0.29614146],
 [0.        , 0.        , 0.        , 0.        ],
 [0.24431509, 0.12812513, 0.12433312, 0.05390116],
 [0.        , 0.        , 0.        , 0.        ],
 [0.29548997, 0.38527344, 0.29276212, 0.47066015],
 [0.4807615 , 0.5785883 , 0.31331405, 0.35295741],
 [0.44399654, 0.3600765 , 0.42077233, 0.26702024],
 [0.        , 0.        , 0.        , 0.        ],
 [0.        , 0.        , 0.        , 0.        ],
 [0.39401069, 0.33802772, 0.7033619 , 0.46248769],
 [0.65959693, 0.85742038, 0.76518557, 0.73468326],
 [0.        , 0.        , 0.        , 0.        ]])

# learned q_table with discount=0.95 after 100k episodes
# uncomment if you don't want to train
q_table_discounted = np.array([[0.11086028, 0.10873646, 0.11937905, 0.10764452],
 [0.05457366, 0.0521126 , 0.06920769, 0.10343085],
 [0.12007901, 0.07751225, 0.08306081, 0.08510609],
 [0.06058289, 0.05500062, 0.05100563, 0.08659944],
 [0.16391285, 0.09694021, 0.11711073, 0.10711821],
 [0.        , 0.        , 0.        , 0.        ],
 [0.10753944, 0.16301227, 0.05027957, 0.04267554],
 [0.        , 0.        , 0.        , 0.        ],
 [0.10461931, 0.16113925, 0.120788  , 0.19534531],
 [0.20950422, 0.34757397, 0.21332139, 0.15570236],
 [0.40224346, 0.26028362, 0.31256761, 0.15306386],
 [0.        , 0.        , 0.        , 0.        ],
 [0.        , 0.        , 0.        , 0.        ],
 [0.21537173, 0.31379267, 0.49150142, 0.38470216],
 [0.46608636, 0.78706054, 0.6564961 , 0.47320564],
 [0.        , 0.        , 0.        , 0.        ]])

# for reading in a previously saved q_table:
## q_table =  np.load("saved_q_table.npy")

# pass in mode='human' to display the game
play_game('FrozenLake-v0', q_table, ai='q_learning')
play_game('FrozenLake-v0', q_table_discounted, ai='q_learning')
play_game('FrozenLake-v0', ai=greedy_4x4_bot)
play_game('FrozenLake-v0', ai='random')

import gym
import numpy as np
import random


def learn_q_table(
        environment,
        learning_rate,
        discount_factor,
        random_factor,
        num_episodes=1000000,
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
                action = env.action_space.sample()  # random action
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

    print(q_table)
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
        mode='human',
        ai='random'
):
    print("Playing {} with {} AI".format(environment, str(ai)))
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
                action = env.action_space.sample()  # fallback ai, take random action
            else:
                action = ai(curr_state)  # custom ai for game

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
def greedy_8x8_bot(curr_state):
    holes = [5, 7, 11, 12]
    # get the row and column position of our agent in the grid
    row = curr_state // 8
    col = curr_state % 8

    # check if we can move down without falling in a hole or going out of bounds
    if ((curr_state + 8) not in holes and (row + 1 < 8)):
        action = 1  # move down
    elif ((curr_state + 1) not in holes and (col + 1 < 8)):
        action = 2  # move right
    elif ((curr_state - 1) not in holes and (col - 1 >= 0)):
        action = 0  # move left
    else:
        action = 3  # move up

    return action


# learn the q_tables
# uncomment to learn new q tables. make sure to comment out hard_codes q_tables below
# q_table = learn_q_table('FrozenLake8x8-v0', learning_rate=0.5, discount_factor=1, random_factor=0.5)
# q_table_discounted = learn_q_table('Blackjack-v0', learning_rate=0.1, discount_factor=0.95, random_factor=0.1)

# learned q_table with no discount after 100k episodes
# uncomment if you don't want to train
q_table = np.array([[9.09238558e-01, 9.09137783e-01, 9.09339350e-01, 9.09378478e-01],
                    [9.09370958e-01, 9.09381339e-01, 9.09385149e-01, 9.09386126e-01],
                    [9.09386967e-01, 9.09388951e-01, 9.09390647e-01, 9.09387725e-01],
                    [9.09389938e-01, 9.09392063e-01, 9.09392689e-01, 9.09395607e-01],
                    [9.09395266e-01, 9.09396856e-01, 9.09396310e-01, 9.09396972e-01],
                    [9.09395183e-01, 9.09395710e-01, 9.09368511e-01, 9.09396875e-01],
                    [9.09340806e-01, 9.09388736e-01, 9.09276696e-01, 9.09386667e-01],
                    [9.09288887e-01, 9.09354567e-01, 9.09360628e-01, 9.09371461e-01],
                    [9.08899671e-01, 9.08248166e-01, 9.08911673e-01, 9.09338956e-01],
                    [9.03242476e-01, 9.00619838e-01, 9.07573386e-01, 9.09383483e-01],
                    [9.04719094e-01, 9.07945395e-01, 9.04837264e-01, 9.09391197e-01],
                    [2.25123773e-01, 8.54296391e-01, 6.29638987e-01, 9.09393512e-01],
                    [8.80955733e-01, 8.26742498e-01, 8.83309273e-01, 9.09366833e-01],
                    [9.02912089e-01, 9.03422652e-01, 9.08476966e-01, 9.09272437e-01],
                    [9.05688767e-01, 9.05239836e-01, 9.08319095e-01, 9.09164652e-01],
                    [9.09089602e-01, 9.08643349e-01, 9.08480750e-01, 9.09264050e-01],
                    [9.02365837e-01, 8.99303215e-01, 8.99007351e-01, 9.06040533e-01],
                    [9.01163454e-01, 8.90997521e-01, 8.87283126e-01, 8.92512249e-01],
                    [9.04948948e-01, 8.34824837e-01, 6.28770204e-01, 8.17826562e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [5.72532080e-01, 1.86012726e-01, 6.56828015e-01, 3.35244162e-01],
                    [8.80181694e-01, 5.51269162e-01, 8.98337511e-01, 7.77137447e-01],
                    [8.97420329e-01, 8.76710304e-01, 8.60212339e-01, 9.02992550e-01],
                    [9.02869177e-01, 9.03624576e-01, 9.00059022e-01, 9.05635636e-01],
                    [8.99210023e-01, 8.87464531e-01, 8.87068909e-01, 8.93639372e-01],
                    [8.44011767e-01, 8.80975805e-01, 8.76324458e-01, 8.81892833e-01],
                    [6.33399251e-01, 7.05190077e-01, 5.40941401e-01, 8.16865260e-01],
                    [5.12949167e-01, 7.52371579e-01, 1.03774444e-01, 4.35046356e-01],
                    [4.64001252e-01, 7.89691348e-02, 3.98783637e-01, 3.93624118e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [6.62772085e-01, 5.77802214e-01, 8.78540669e-01, 4.22259872e-01],
                    [8.88841130e-01, 8.69363899e-01, 8.89019188e-01, 8.90761718e-01],
                    [8.32963344e-01, 8.33650992e-01, 8.62802174e-01, 7.83972931e-01],
                    [6.03936222e-01, 5.80998667e-01, 4.61652512e-01, 7.97424872e-01],
                    [8.02196672e-01, 1.08940239e-01, 6.67878131e-03, 4.13119821e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [2.56076062e-01, 2.46752759e-01, 3.64228674e-01, 2.42858927e-01],
                    [1.75052455e-01, 6.00837444e-01, 9.33808077e-02, 2.28892748e-02],
                    [3.12485611e-01, 5.60470819e-01, 1.10495942e-01, 8.51103506e-01],
                    [8.35403847e-01, 8.28320128e-01, 8.29934342e-01, 8.48285502e-01],
                    [8.02275832e-01, 1.56778350e-01, 4.65969406e-01, 7.12749252e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [2.11619297e-02, 3.50979291e-02, 3.88004809e-02, 3.78478812e-02],
                    [3.93683993e-02, 9.25140014e-02, 1.54443018e-01, 1.77169028e-01],
                    [2.57165941e-01, 1.60347486e-01, 2.47120173e-01, 1.72660994e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [5.86416821e-01, 9.48112666e-02, 7.53740716e-01, 7.87755981e-01],
                    [4.94469720e-01, 2.23424243e-01, 3.99119868e-01, 2.39982684e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [2.91531721e-02, 5.71228248e-02, 1.60485377e-01, 2.15291197e-02],
                    [6.06693539e-03, 7.28833764e-04, 8.00039804e-03, 2.73166043e-02],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [7.87676070e-02, 3.85568717e-02, 1.24890280e-01, 1.37516368e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [3.42442666e-01, 4.94199195e-01, 8.25062015e-01, 1.89478804e-01],
                    [5.20156093e-01, 4.68372307e-01, 4.60355623e-01, 4.52176523e-01],
                    [1.07724126e-01, 1.70281736e-01, 1.08481403e-01, 2.17428483e-01],
                    [7.31348390e-02, 5.47144009e-02, 6.44234734e-02, 8.51624953e-02],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [1.31963905e-01, 2.99019995e-01, 3.84622967e-01, 4.80915687e-02],
                    [4.84492831e-01, 4.50469722e-01, 6.07597901e-01, 4.89748079e-01],
                    [4.60506072e-01, 8.35883302e-01, 3.30710468e-01, 3.16498619e-01],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

# learned q_table with discount=0.95 after 100k episodes
# uncomment if you don't want to train
# q_table_discounted = np.array([[0.11086028, 0.10873646, 0.11937905, 0.10764452],
# [0.05457366, 0.0521126 , 0.06920769, 0.10343085],
# [0.12007901, 0.07751225, 0.08306081, 0.08510609],
# [0.06058289, 0.05500062, 0.05100563, 0.08659944],
# [0.16391285, 0.09694021, 0.11711073, 0.10711821],
# [0.        , 0.        , 0.        , 0.        ],
# [0.10753944, 0.16301227, 0.05027957, 0.04267554],
# [0.        , 0.        , 0.        , 0.        ],
# [0.10461931, 0.16113925, 0.120788  , 0.19534531],
# [0.20950422, 0.34757397, 0.21332139, 0.15570236],
# [0.40224346, 0.26028362, 0.31256761, 0.15306386],
# [0.        , 0.        , 0.        , 0.        ],
# [0.        , 0.        , 0.        , 0.        ],
# [0.21537173, 0.31379267, 0.49150142, 0.38470216],
# [0.46608636, 0.78706054, 0.6564961 , 0.47320564],
# [0.        , 0.        , 0.        , 0.        ]])

# for reading in a previously saved q_table:
## q_table =  np.load("saved_q_table.npy")

# pass in mode='human' to display the game
play_game('FrozenLake8x8-v0', q_table, ai='q_learning')
# play_game('FrozenLake-v0', q_table_discounted, ai='q_learning')
# play_game('FrozenLake8x8-v0', ai=greedy_8x8_bot)
# play_game('FrozenLake8x8-v0', ai='random')

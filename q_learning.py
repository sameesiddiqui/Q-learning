import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep

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

    frames = [] # list of frames for playing video later

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
            elif (ai == 'human'):	# allows us, the scientist to play the game.
                action = int(input("select an action (int): "))
            else:
                action = ai(curr_state) # custom ai for game: TODO not defined here, must be defiend...

            curr_state, reward, done, info = env.step(action)

            if (mode == 'human'):
                print("move taken: {}".format(action))
            score_for_episode += reward

			## save frame to frames...
            frames.append({
                'frame': env.render(mode="ansi"),
                'state': curr_state,
                'action': action,
                'reward': reward
            })

            if done:
                frames.append(score_for_episode)
                break
		#endfor

        if (mode == 'human'):
            print("score for episode: {}".format(score_for_episode))

        total_score += score_for_episode
        total_timesteps += t

    log_metrics(num_episodes, total_score, total_timesteps)

	#print_frames(frames)

    # return average score
    # return (total_score / num_episodes)
    #
    # return video frames:
    return frames
#endplay_game

def print_frames(frames):
    for i, frame in enumerate(frames):
        if (i == len(frames) - 1):
            print("===========")
            print("Final score: {}".format(frame))
            break

        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")

        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        #sleep(.1)
        sleep(.5)
	#endfor
#endprint_frames


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
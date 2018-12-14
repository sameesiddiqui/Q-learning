import time
import gym

env = gym.make("Blackjack-v0")

env.reset()
print("action space:")
print(env.action_space)

print("observation space:")
print(env.observation_space)

# each game of blackjack is an episode:
# actions:
#	true: hit
#	false: stick
#
# observations: touple of three values:
#	1) value of player's hand
#	2) value of dealer's revieled card
#	3) 0 or 1 does the player have a usable ace (does not matter really)
#
# the reward is: (a > b) - (a < b) where a is player's hand and b is dealers hand.
#	+1	: player wins
#	 0	: draw
#	-1	: dealer wins

# seed the rng:
env.seed(int(time.time()))

# for 5 episodes:
for i in range(20):
	observation = env.reset()
	print("")
	print("initial observation:")
	print(observation)
	done = False;
	#for j in range() # bots behavior:
	# possible actions 0, 1: stick, hit
	while(not done):
		if (observation[0] < 18):
			observation, reward, done, info = env.step(1)
			print("hit: {}".format(observation[0]), end=" " )
		else:
			observation, reward, done, info = env.step(0)
			print ("stick: {}".format(observation), end=" " )
	#end of game:
	if(reward == -1):
		print("\nLOST!")
	elif (reward == 0):
		print("\nDRAW!")
	elif (reward == 1):
		print("\nWON!")

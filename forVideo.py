import sys
import time
import numpy as np
import q_learning as ql

# runs the environment using previously learned behaviour from learningDriver
def main():
	if(len(sys.argv) < 2):
		print("usage: python3 testingDriver.py <environment>")
	#endif

	# get given environment from command line argument
	runenv = sys.argv[1]

	# load previously learned tables for given ennvironment
	q_table = np.load(runenv+"_saved_q_table.npy")
	q_table_discounted = np.load(runenv+"_saved_q_table_discounted.npy")

	# run and print average results
	starttime = time.time()
	
	framesA = ql.play_game(runenv, q_table, ai="q_learning", num_episodes=10, mode="ansi")
	endtime = time.time()

	elapsedtime = starttime - endtime
	print("elapsed time: "+str(elapsedtime)+" seconds")

	ql.print_frames(framesA)

	starttime = time.time()
	
	framesB =ql.play_game(runenv, q_table_discounted, ai="q_learning", num_episodes=10, mode="ansi")
	endtime = time.time()

	elapsedtime = starttime - endtime
	print("elapsed time: "+str(elapsedtime)+" seconds")

	ql.print_frames(framesB)

#endmain

main()

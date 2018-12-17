import sys
import time
import numpy as np
import q_learning as ql

# runs the environment using previously learned behaviour from learningDriver
def main():
    if(len(sys.argv) < 2):
        print("usage: python3 forVideo.py <environment> discounted=<True/False>")
        exit()
    #endif

    # get given environment from command line argument
    runenv = sys.argv[1]
    rundiscounted = sys.argv[2][11:].lower() == 'true'

    # load previously learned tables for given ennvironment
    q_table = np.load(runenv+"_saved_q_table.npy")
    q_table_discounted = np.load(runenv+"_saved_q_table_discounted.npy")

    # run and print average results
    starttime = time.time()

    if (rundiscounted):
        frames = ql.play_game(runenv, q_table, ai="q_learning", num_episodes=1, mode="ansi")
    else:
        frames = ql.play_game(runenv, q_table_discounted, ai="q_learning", num_episodes=1, mode="ansi")
    endtime = time.time()

    elapsedtime = starttime - endtime
    print("elapsed time: "+str(elapsedtime)+" seconds")

    ql.print_frames(frames)

#endmain

main()

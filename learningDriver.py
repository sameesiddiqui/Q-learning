import sys
import q_learning as ql
import time

## creates a q_table trainned to play the given environment:
def main():
	if (len(sys.argv) < 2):
		print("usage: python3 driver.py <environment>")
		exit()
	#endif

	runenv = sys.argv[1]

#	q_table = learn_q_table("Taxi-v2", learning_rate=0.1, discount_factor=1, random_factor=0.1, save_file="saved_q_table")
#	q_table_discounted = learn_q_table("Taxi-v2", learning_rate=0.1, discount_factor=0.95, random_factor=0.1, save_file="saved_q_table_discounted")

	starttime = time.time()
	q_table = ql.learn_q_table(runenv, learning_rate=0.1, discount_factor=1, random_factor=0.1, save_file=(runenv+"_saved_q_table"))
	endtime = time.time()

	elapsedtime = starttime - endtime
	print("elapsed time: "+str(elapsedtime)+" seconds")

	starttime = time.time()
	q_table_discounted = ql.learn_q_table(runenv, learning_rate=0.1, discount_factor=0.95, random_factor=0.1, save_file=(runenv+"_saved_q_table_discounted"))
	endtime = time.time()

	elapsedtime = starttime - endtime
	print("elapsedtime: "+str(elapsedtime)+" seconds")

main()

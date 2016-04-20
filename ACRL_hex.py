import theano
from lasagne.updates import rmsprop
from theano import tensor as T
import numpy as np
import numpy.random as rand
from inputFormat import *
from network import network
import cPickle
import argparse
import time
import os
from Actor_Network import *
npa = np.array

class Actor():
    def __init__(self):
        self.alpha = 0.0001
        self.weights = np.zeros((num_channels*input_size*input_size,1))
        self.n = num_channels*input_size*input_size
        
    def weights_update(self,delta,cf):
        self.weights += self.alpha * delta * cf
    

def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

def save():
	print "saving network..."
	if args.save:
		save_name = args.save
	else:
		save_name = "Q_network.save"
	if args.data:
		f = file(args.data+"/"+save_name, 'wb')
	else:
		f = file(save_name, 'wb')
	cPickle.dump(network, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	np.savetxt('Actor_weights',actor_net.weights)
	if args.data:
		f = file(args.data+"/replay_mem.save", 'wb')
		cPickle.dump(mem, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = file(args.data+"/costs.save","wb")
		cPickle.dump(costs, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()
		f = file(args.data+"/values.save","wb")
		cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


def epsilon_greedy_policy(state, evaluator):
	rand = np.random.random()
	played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
	
	counter = 0
	for n in played:
	    if n:
		counter += 1
	num = boardsize*boardsize - counter
	
	if(rand>epsilon_q):
		scores = evaluator(state)
		 
		#set value of played cells impossibly low so they are never picked
		scores[played] = -2
		#np.set_printoptions(precision=3, linewidth=100)
		#print scores.argmax(),scores.max(), scores.shape
		mem.add_action_epsilon(1 - epsilon_q + epsilon_q/num)
		return scores.argmax(), scores.max()
	#choose random open cell
	mem.add_action_epsilon(epsilon_q/num)
	return np.random.choice(np.arange(boardsize*boardsize)[np.logical_not(played)]), 0
'''
def softmax(x, t):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x))/t)
    return e_x / e_x.sum()
'''
def softmax_policy(state, evaluator, temperature=1):
	rand = np.random.random()
	not_played = np.logical_not(np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding])).flatten()
	scores = evaluator(state)
	prob = softmax(scores[not_played], temperature)
	tot = 0
	choice = None
	for i in range(prob.size):
		tot += prob[i]
		if(tot>rand):
			choice = i
			break
	return not_played.nonzero()[0][choice]


def compatible_features(states1,targets,actions,states2,epsils):
	rho_actor = []
	features = []
	flag = False
    
	i = 0
	comp_features = []
	for eps in epsils:
	    state = states1[i]	 
	    prob_a_s = softmax(actor_net.weights)
	    #print "Weights shape = ", weights.shape, "Prob shape = ", prob_a_s.shape, "Feat shape = ", feat.shape
	    played = np.logical_or(state[white,padding:boardsize+padding,padding:boardsize+padding],\
		      state[black,padding:boardsize+padding,padding:boardsize+padding]).flatten()
	    not_played = np.logical_not(played)
	    feat_sum = 0
	    rho_actor.append(prob_a_s[actions[i]]/eps)
	    j = count = 0
	    for a in not_played:
		if a:
		    simulate_move = action_to_cell(j)		
		    if j != actions[i]:
			state[white,simulate_move[0],simulate_move[1]] = 1
			f = np.reshape(state,(actor_net.n,1))
			f.astype(float)
			feat_sum += prob_a_s[a] * f
			count += 1
		    else:
			flag = True
		    state = states1[i]
		j += 1
	    phi_s_a = np.reshape(states2[i],(actor_net.n,1))
	    phi_s_a.astype(float)
	    comp_feat = phi_s_a - feat_sum
	    comp_features.append((comp_feat))		    
	    i += 1
	    
	#print len(comp_features),comp_features[1].shape
	#print count, flag
	
	for feat,delta,r in zip(comp_features,targets,rho_actor):
	    #print "Shapes = ",r,feat.shape,delta
	    actor_net.weights_update(delta,r*feat)
	pass

def Q_update():
	states1, actions, rewards, states2, epsils = mem.sample_batch(batch_size)
	scores = evaluate_model_batch(states2)
	scores1 = evaluate_model_batch(states1)
	played = np.logical_or(states2[:,white,padding:boardsize+padding,padding:boardsize+padding],\
		     states2[:,black,padding:boardsize+padding,padding:boardsize+padding]).reshape(scores.shape)
	played1 = np.logical_or(states1[:,white,padding:boardsize+padding,padding:boardsize+padding],\
		     states1[:,black,padding:boardsize+padding,padding:boardsize+padding]).reshape(scores.shape)
	#set value of played cells impossibly low so they are never picked
	scores[played] = -2
	scores1[played1] = -2
	targets = np.zeros(rewards.size).astype(theano.config.floatX)
	targets[rewards==1] = 1
	targets[rewards==0] = -np.amax(scores, axis=1)[rewards==0]
	cost = train_model(states1,targets,actions)
	targets1 = -np.amax(scores1,axis=1)
	#print states1.shape, actions.shape, rewards.shape, scores.shape, targets.shape, played.shape 
	#print "Action = :",actions[1], "Reward = ", rewards[1]
	delta = targets - targets1
	compatible_features(states1,delta,actions,states2,epsils)
    
	return cost

def action_to_cell(action):
	cell = np.unravel_index(action, (boardsize,boardsize))
	return(cell[0]+padding, cell[1]+padding)

def flip_action(action):
	return boardsize*boardsize-1-action

class replay_memory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.size = 0
		self.index = 0
		self.full = False
		self.state1_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)
		self.action_memory = np.zeros(capacity, dtype=np.uint8)
		self.epsilon_memory = np.zeros(capacity, dtype=np.float64)
		self.reward_memory = np.zeros(capacity, dtype=bool)
		self.state2_memory = np.zeros(np.concatenate(([capacity], input_shape)), dtype=bool)

	def add_entry(self, state1, action, reward, state2):
		self.state1_memory[self.index, :, :] = state1
		self.state2_memory[self.index, :, :] = state2
		self.action_memory[self.index] = action
		self.reward_memory[self.index] = reward
		self.index += 1
		if(self.index>=self.capacity):
			self.full = True
			self.index = 0
		if not self.full:
			self.size += 1

	def sample_batch(self, size):
		batch = np.random.choice(np.arange(0,self.size), size=size)
		states1 = self.state1_memory[batch]
		states2 = self.state2_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		epsils = self.epsilon_memory[batch]
		return (states1, actions, rewards, states2, epsils)
		
	def add_action_epsilon(self,eps):
		self.epsilon_memory[self.index] = eps
		pass


parser = argparse.ArgumentParser()
parser.add_argument("--load", "-l", type=str, help="Specify a file with a prebuilt network to load.")
parser.add_argument("--save", "-s", type=str, help="Specify a file to save trained network to.")
parser.add_argument("--data", "-d", type =str, help="Specify a directory to save/load data for this run.")
args = parser.parse_args()

#save network every x minutes during training
save_time = 30

print "loading starting positions... "
datafile = open("data/scoredPositionsFull.npz", 'r')
data = np.load(datafile)
positions = data['positions']
datafile.close()
numPositions = len(positions)

input_state = T.tensor3('input_state')

state_batch = T.tensor4('state_batch')
target_batch = T.dvector('target_batch')
action_batch = T.ivector('action_batch')

replay_capacity = 100000

if args.data:
	if not os.path.exists(args.data):
		os.makedirs(args.data)
		mem = replay_memory(replay_capacity)
		costs = []
		values = []
	else:
		if os.path.exists(args.data+"/replay_mem.save"):
			print "loading replay memory..."
			f = file(args.data+"/replay_mem.save")
			mem = cPickle.load(f)
			f.close
		else:
			#replay memory from which updates are drawn
			mem = replay_memory(replay_capacity)
		if os.path.exists(args.data+"/costs.save"):
			f = file(args.data+"/costs.save")
			costs = cPickle.load(f)
			f.close
		else:
			costs = []
		if os.path.exists(args.data+"/values.save"):
			f = file(args.data+"/values.save")
			values = cPickle.load(f)
			f.close
		else:
			values = []
else:
	#replay memory from which updates are drawn
	mem = replay_memory(replay_capacity)
	costs = []
	values = []

numEpisodes = 100000
batch_size = 64

#if load parameter is passed load a network from a file
if args.load:
	print "loading model..."
	f = file(args.load, 'rb')
	network = cPickle.load(f)
	batch_size = network.batch_size
	f.close()
else:
	print "building model..."
	network = network(batch_size=batch_size)
	print "network size: "+str(network.mem_size.eval())

#zeros used for running network on a single state without modifying batch size
input_padding = theano.shared(np.zeros(np.concatenate(([network.batch_size],input_shape))).astype(theano.config.floatX))
evaluate_model_single = theano.function(
	[input_state],
	network.output[0],
	givens={
        network.input: T.set_subtensor(input_padding[0,:,:,:], input_state),
	}
)

evaluate_model_batch = theano.function(
	[state_batch],
	network.output,
	givens={
        network.input: state_batch,
	}
)

cost = T.mean(T.sqr(network.output[T.arange(target_batch.shape[0]),action_batch] - target_batch))

alpha = 0.001
rho = 0.9
epsilon = 1e-6
updates = rmsprop(cost, network.params, alpha, rho, epsilon)

train_model = theano.function(
	[state_batch,target_batch,action_batch],
	cost,
	updates = updates,
	givens={
		network.input: state_batch,
	}
)

actor_net = Actor()

print "Running episodes..."
epsilon_q = 0.1
last_save = time.clock()
try:
	for i in range(numEpisodes):
		cost = 0
		num_step = 0
		value_sum = 0
		#randomly choose who is to move from each position to increase variability in dataset
		move_parity = np.random.choice([True,False])
		#randomly choose starting position from database
		index = np.random.randint(numPositions)
		#randomly flip states to capture symmetry
		if(np.random.choice([True,False])):
			gameW = np.copy(positions[index])
		else:
			gameW = flip_game(positions[index])
		gameB = mirror_game(gameW)
		while(winner(gameW)==None):
			t = time.clock()
			action, value = epsilon_greedy_policy(gameW if move_parity else gameB, evaluate_model_single)
			value_sum+=abs(value)
			state1 = np.copy(gameW if move_parity else gameB)
			move_cell = action_to_cell(action)
			play_cell(gameW, move_cell if move_parity else cell_m(move_cell), white if move_parity else black)
			play_cell(gameB, cell_m(move_cell) if move_parity else move_cell, black if move_parity else white)
			#print gameW.shape, move_cell, network.output
			if(not winner(gameW)==None):
				#only the player who just moved can win, so if anyone wins the reward is 1
				#for the current player
				reward = 1
			else:
				reward = 0
			#randomly flip states to capture symmetry
			if(np.random.choice([True,False])):
				state2 = np.copy(gameB if move_parity else gameW)
			else:
				state2 = flip_game(gameB if move_parity else gameW)
			move_parity = not move_parity
			mem.add_entry(state1, action, reward, state2)
			if(mem.size > batch_size):
				cost += Q_update()
				#print state_string(gameW)
			num_step += 1
			if(time.clock()-last_save > 60*save_time):
				save()
				last_save = time.clock()
		run_time = time.clock() - t
		print "Episode", i, "complete, cost: ", 0 if num_step == 0 else cost/num_step, " Time per move: ", 0 if num_step == 0 else run_time/num_step, "Average value magnitude: ", 0 if num_step == 0 else value_sum/num_step
		costs.append(0 if num_step == 0 else cost/num_step)
		values.append(0 if num_step == 0 else value_sum/num_step)
		np.savetxt('Actor_weights',actor_net.weights)
except KeyboardInterrupt:
	#save snapshot of network if we interrupt so we can pickup again later
	save()
	exit(1)

save()

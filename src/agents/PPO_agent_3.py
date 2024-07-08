# Agent Script for Autonomous Vehicle Control (AV) (Single-Agent)
# Version: 2
# Algorithm: PPO (Continuous)
# Remark: No critic, Use sum reward instead of advantages function (Same configuration as the original paper (Alex Bayen))

# ************************************************************** SUMO related headers start ******************************************************************************
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary
# *************************************************************** SUMO related headers end *******************************************************************************

# ****************************************************************** Other headers start *********************************************************************************
import tensorflow as tf  # Tensorflow
import keras
import numpy as np   # Numpy
from utility import my_utils
# ******************************************************************* Other headers end **********************************************************************************

# Hyperparameters for PPO. (Globally declared)
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE parameter
EPSILON = 0.2  # Prob ratio clipping value

# Parameters for SUMO's vehicle. (Globally declared)
STATE_DIM = 3  # 1-AV's speed, 2-Lead veh's speed, 3-Lead veh's distance
ACTION_DIM = 1  # 1-Acceleration
MAX_SPEED = 10 
MIN_SPEED = 0  
MAX_ACCEL = 0.5  # Max accel of AV  (Action's Upper Bound)
MAX_DECEL = -0.5  # max decel of AV. (Action's Lower Bound)
LEAD_DIST = 100  # Leading vehicle detecting distance

# ******************************************************************* PPO Agent Class Start ******************************************************************************
class Autonomous_Vehicle_Agent():
    # Initialise Variables
    def __init__(self, load, 
                 time_step,
                 observation_budget, 
                 horizon=10000, 
                 collector=False,
                 action_config='continuous',
                 reward_objective='greedy',
                 reward_config='original',
                 policy_objective='sum_reward', 
                 id='AV'):
        self.alg = 'PPO'  # Algorithm
        self.revision = 3  # Revision
        self.vehicle_id = id  # Get the vehicle'id
        self.vehicle_type = 'rl_agent'  # vehicle_type in the sumo config.
        self.sumo_time_step = time_step  # sumo time step
        self.collector_config = collector  # collector
        self.action_config = action_config  # action configuration (discrete or continuous)
        self.reward_objective = reward_objective  # reward objective (greedy or global)
        self.reward_config = reward_config  # reward configuration (original or my_reward)
        self.policy_objective = policy_objective  # policy objective (advantages or sum_reward)
        # State 
        self.state = np.zeros([STATE_DIM], dtype=float)  # State (S_t)
        self.next_state = self.state  # Next state (S_{t+1})
        # Action Space Parameters
        self.action = 0  # Action (a_t)
        self.policy = 0  # Policy of Action (Probability) (\pi_t)
        # Others
        self.termination = False  # Termination parameters (T_t)
        self.prev_reward = 0  # Previous reward (R_t)
        self.reward = 0  # Reward (R_{t+1})
        self.accum_reward = 0  # Accumulated_Reward
        self.reward_append = []  # Append Reward (For calculating the mean)
        self.accum_reward_append = []  # Append Accumulated Reward (For computing the std)
        self.episode_reward = 0  # Episode reward
        self.avg_reward = 0  # Average episode reward
        self.step = 0  # Agent step
        self.index_counter = 0  # Agent index counter (used for storing experiences.) (Not used in this alg.)
        self.av_speed_append = []  # Storage for av's speed (For visualisation)
        self.avg_speed_append = []  # Storage for avg's speed of all vehs (For visualisation)
        self.trajectory = 0  # Trajectory counter.
        self.evaluation_score = 0  # Evaluation score used for checking whether to save the updated models or not.
        self.prev_evaluation_score = 0  # Store the previous eva score.

        # Initialise observation buffers
        self.observation_budget = observation_budget  # Num of batched
        self.observation_size = int(horizon)  # Horizon
        self.state_obs = np.zeros([self.observation_size, STATE_DIM], dtype=float)
        self.action_obs = np.zeros([self.observation_size, ACTION_DIM], dtype=float)  
        self.policy_obs = np.zeros([self.observation_size, ACTION_DIM], dtype=float) 
        self.reward_obs = np.zeros([self.observation_size, 1], dtype=float) 
        self.next_state_obs = np.zeros([self.observation_size, STATE_DIM], dtype=float)
        self.minibatch_size = 1000 
        self.total_buffers_dict = dict()  # Storage for all batched 

        # Declare file paths
        self.TRAINING_PATH = 'src/training_record/{:s}{:d}_{:s}_single_ring_training_record.npz'.format(self.alg, self.revision, self.reward_objective)
        self.ACTOR_PATH = 'src/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective)
        self.CRITIC_PATH = 'src/trained_model/{:s}{:d}_{:s}_single_ring_critic.keras'.format(self.alg, self.revision, self.reward_objective)
        self.TEST_PATH = 'src/testing_record/{:s}{:d}_{:s}_single_ring_testing_record'.format(self.alg, self.revision, self.reward_objective)

        # Check if this is the first time training or not ?
        # If load == False --> this is the first time training --> create new array for these information
        if not load:
            self.sum_reward_append = []
            self.training_record = 0  # For on-policy, the training record will be started from zero(0).

            # Initialise networks
            # It is the actor-critic framework in PPO.
            # Creating the function approximator (neural networks)
            self.actor = self.networkCreation(net_type='policy')
            self.critic = self.networkCreation(net_type='q_network')

        # If load == True --> Continue training --> Load the existing training_record.
        else:
            saved_data = np.load(self.TRAINING_PATH)
            self.sum_reward_append = np.squeeze(saved_data['arr_0'])
            self.training_record = np.squeeze(saved_data['arr_1'])
            # Load the existing models.
            self.actor = keras.models.load_model(self.ACTOR_PATH)
            self.critic = keras.models.load_model(self.CRITIC_PATH)
        
        # Intialise network parameters
        self.actor_glob_gradient = 0  # Global gradient of the actor network (For visualisation only).
        self.critic_glob_gradient = 0  # Global gradient of the critic network (For visualisation only).

    # Network creation method
    # Same network's architecture as the original paper (Alex Bayen).
    def networkCreation(self, net_type):
        # Define local hyperparameters
        _NUM_LAYER = 64
        _LR = 3e-4

        # Input layer
        input = keras.layers.Input(shape=[STATE_DIM,])

        # Fully Connected layers
        dense_1 = keras.layers.Dense(_NUM_LAYER, activation='relu')(input)
        dense_2 = keras.layers.Dense(_NUM_LAYER, activation='relu')(dense_1)
        dense_3 = keras.layers.Dense(_NUM_LAYER, activation='relu')(dense_2)
        # dense_3 = keras.layers.BatchNormalization()(dense_3)

        # Output layer
        actor_output_1 = keras.layers.Dense(ACTION_DIM, activation='linear')(dense_3)  # Mean of actor's output
        actor_output_2 = keras.layers.Dense(ACTION_DIM, activation='softplus')(dense_3)  # Variance of actor's output
        critic_output = keras.layers.Dense(1, activation='linear')(dense_3)

        if net_type == 'policy':
            output_config = [actor_output_1, actor_output_2]
        else:
            output_config = critic_output

        # Determine the model
        # Actor -- Policy Network
        model = keras.models.Model(inputs=input, outputs=output_config, name='policy_network')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=_LR))
        
        # Return the model
        return model

    # State Retrieving Method
    def getState(self, noise=False):
        # ----- State condition 1 (Follow Alex Bayen's paper):
        # state[0] = # Speed of the AV (m/s)
        # state[1] = # Speed of vehicle in front (m/s)
        # state[2] = # Distance between AV and the vehicle in the front. (m)
        state = np.zeros([STATE_DIM], dtype=float)
        state[0] = traci.vehicle.getSpeed(vehID=self.vehicle_id)  # Get It's speed (m/s)
        # Get Leader
        leader_info = traci.vehicle.getLeader(vehID=self.vehicle_id, dist=LEAD_DIST)
        # Check if there is a vehicle in front ?
        if leader_info != None:
            # Yes -- There is
            state[1] = traci.vehicle.getSpeed(vehID=leader_info[0])
            state[2] = leader_info[1]
        else:
            # No -- There is not
            state[1] = MAX_SPEED  # Assign the maximum speed (Assume that the leader is faster than AV.)
            state[2] = LEAD_DIST  # Assign distance = maximum observeable distance.
        
        # Inject noise to the observation (if noise=True)
        if noise == True:
            noise_factor = 5  # If == 0, no noise in the observation
            for n in range(len(state)):
                if n == 0:
                    # No noise in it's own speed
                    ran_noise = 0
                else:
                    ran_noise = np.random.random() * noise_factor
                state[n] += ran_noise

        # Return state
        return state

    # Policy Retrieving Method (Numpy Format)
    def getActionPolicy(self, input, num:int) -> float:
        input_tensor = my_utils.tensorConversion(input)
        # Get mean, variance and standard deviation
        mean_tensor, var_tensor = self.actor(input_tensor)
        mean = mean_tensor.numpy()
        var = var_tensor.numpy()
        std = np.sqrt(var)
        # Get sample
        sample = np.random.normal(loc=mean, scale=std, size=[num, ACTION_DIM])
        # Get sample's pdf
        pdf = (1/np.sqrt(2*np.pi*var)) * np.exp((-1/(2*var))*np.square(sample - mean))
        if pdf <= 0:
            pdf = 1e-06
        elif pdf >= 1:
            pdf = 1
        # Return sample, mean, var and std
        return sample, pdf
    
    # Predict Value Function Method (Numpy Format)
    def getValueFunction(self, input) -> float:
        input_tensor = my_utils.tensorConversion(input)
        return self.critic(input_tensor).numpy()

    # Action Performing Method
    def actionPerform(self, state, action=None):
        dummy = False
        if action == None:
            # Get Action (Predict from the policy network)
            action_sample, action_policy = self.getActionPolicy(state, num=1)
        else:
            # If the action is pre-defined, It means that this is the warm-up step.
            action_sample = action.numpy()
            action_policy = tf.constant([1], dtype=tf.float32).numpy()

        duration = int(1/self.sumo_time_step)  # 1s/sumo_time_step(0.1s)
        action_long = np.clip(action_sample, a_min=MAX_DECEL, a_max=MAX_ACCEL)

        # Set a derised speed from (current speed + accel)
        desired_speed = np.clip(a=(state[0]+action_long), a_min=MIN_SPEED, a_max=MAX_SPEED)
        # Perform longitudinal action --> Use 'slowDown' command
        traci.vehicle.slowDown(vehID=self.vehicle_id, speed=desired_speed, duration=duration)
        # Note: Return the action and policy from the model prediction before clipping
        return action_sample, dummy, action_policy
    
    # Reward Function Method
    def rewardFunction(self, noise=False, n_veh=44):
        # Define hyperparameters
        sum_speed = 0  # Reset sum_speed (For global objective)
        num_convolute = 10  # Convolution kernel size (1-D) (Used for computing running mean/std)
        # Get next state first
        next_state = self.getState(noise)
        
        # Check for the reward configuration,
        # ------ Alex Bayen's Paper ------
        if self.reward_config == 'original':
            # Derive Avg Speed
            for c in range(n_veh-1):
                sum_speed += traci.vehicle.getSpeed(vehID=str(c))
            avg_speed = (sum_speed + next_state[0]) / n_veh
            # Check reward objective
            if self.reward_objective == 'greedy':
                # Using the greedy objective since it might be feasible in real-world. (Just my assumption)
                # Reward = AV's speed (Greedy objective)
                reward = next_state[0]  
            elif self.reward_objective == 'global': 
                # Reward = All vehicles average speed (Global objective)
                reward = avg_speed
            # Append Reward
            self.reward_append = np.append(self.reward_append, reward)
            # Append Accumalated Reward
            self.accum_reward = (GAMMA*self.accum_reward) + reward
            self.accum_reward_append = np.append(self.accum_reward_append, self.accum_reward)
            # Compute running mean/std
            if len(self.reward_append) <= num_convolute:
                mean_t = np.mean(self.reward_append[0:num_convolute])
                std_t = np.std(self.accum_reward_append[0:num_convolute])
            else:
                m = len(self.reward_append)-num_convolute
                mean_t = np.mean(self.reward_append[0+m:num_convolute+m])
                std_t = np.std(self.accum_reward_append[0+m:num_convolute+m])
            # Normalise the reward (To reduce the variance of PG)
            if std_t != 0:
                norm_reward = (reward - mean_t) / std_t
            else:
                norm_reward = 0
        # ------ My Modified Reward Fn ------
        elif self.reward_config == 'my_reward':
            # ------ Greed Objective -------
            if self.reward_objective == 'greedy':
                reward = next_state[0]  # Get AV's speed.
            # # ------ Global Objective ------
            elif self.reward_objective == 'global':
                # reward = avg_speed
                reward = (next_state[0]+next_state[2]+next_state[4]+next_state[6]+next_state[8])/5
            # Append Reward
            self.reward_append = np.append(self.reward_append, reward)
            # Derive running mean
            kernel_pos = len(self.reward_append) - num_convolute
            if kernel_pos <= num_convolute:
                mean_t = np.mean(self.reward_append[0:num_convolute])
            else:
                mean_t = np.mean(self.reward_append[0+kernel_pos : num_convolute+kernel_pos])
            # Get distance penalty
            # This is the penalty when the av is too far behind from the leading vehicle.
            # dis_penalty = [0.2, 1]  ('1' means that the av is in the perfect clearance.)
            # dis_penalty = 1 - ((next_state[3] - self.min_gap)/self.lead_dist)
            dis_penalty = 1  # Try fix value first (No distance penalty).
            # Normalised the reward
            norm_reward = (mean_t / self.max_speed) * dis_penalty
        else:
            print('Unknown reward configuration.({:s}) Please try again.'.format(self.reward_config))
            exit()  # Terminate the script.
        # Return tuples
        return norm_reward, next_state

    # Observation storing method (Observation)
    def observationStoring(self, index, state, action, policy, reward, next_state) -> None:
        self.state_obs[index] = state  # S_t 
        self.action_obs[index] = action  # a_t
        self.policy_obs[index] = policy  # \pi(a_t|s_t)
        self.reward_obs[index] = reward  # r_{t+1}
        self.next_state_obs[index] = next_state  # S_{t+1}

    # Prepare Training Batch Method (after finish the trajectory)
    def prepareTrainingBatch(self, size:int) -> dict:
        # Get expected sum return (v_s_t) and advantage function (a_hat)
        v_s_t, a_hat = self.getValueAndAdvantage(size, self.state_obs, self.reward_obs, self.next_state_obs)
        # Create training data dictionary
        batch_dict = {'batch_size': size,
                       'state': self.state_obs,
                       'action': self.action_obs,
                       'policy': self.policy_obs,
                       'reward': self.reward_obs,
                       'next_state': self.next_state_obs,
                       'sum_return': v_s_t,
                       'advantage': a_hat}
        return batch_dict

    # Batch Append Method
    # This method will append new batch from last trajectory to the total dictionary that
    # is collecting all batches for training.
    def trainingBuffersAppends(self, _num_batch:int, _batch_dict:dict) -> None:
        _new_dict = {_num_batch: _batch_dict}
        self.total_buffers_dict.update(_new_dict)

    # Minibatch Sampling Method
    def minibatchSampling(self, key:int, pos:int) -> dict:
        _batch = self.total_buffers_dict[key]
        # Sampling for a minibatch size from the selected batch.
        minibatch = {'batch_size': self.minibatch_size,
                          'state': _batch['state'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'action': _batch['action'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'policy': _batch['policy'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'reward': _batch['reward'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'next_state': _batch['next_state'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'sum_return': _batch['sum_return'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)],
                          'advantage': _batch['advantage'][0 + (pos * self.minibatch_size):self.minibatch_size + (pos * self.minibatch_size)]}
        return minibatch

    # Get Expected Sum Return (Value Function) and Advantage Method
    def getValueAndAdvantage(self, num:int, batch_state, batch_reward, batch_next_state) -> float:
        v_s_t = np.zeros([num, 1], dtype=float)
        a_hat = np.zeros([num, 1], dtype=float)
        # Compute TD error
        _delta = (batch_reward + (GAMMA * self.getValueFunction(batch_state))) - self.getValueFunction(batch_next_state)

        # !!! This loop takes around 140 seconds to process 10,000 data.
        # for i in range(num):
        #     for j in range(num - i):
        #         _v_s_t[i] += (GAMMA**j) * batch_reward[i+j]
        #         _a_hat[i] += ((GAMMA * LAMBDA)**j) * _delta[i+j]  # GAE

        # This loop takes only 1.379 seconds for 10,000 data. Thanks to ChatGPT.
        # Error compared to above is only around 1e-18.
        for i in range(num):
            v_discount = np.reshape(np.power((GAMMA), np.arange(num-i)), [num-i, 1])
            a_discount = np.reshape(np.power((GAMMA*LAMBDA), np.arange(num-i)), [num-i, 1])
            v_s_t[i] = np.sum(v_discount * batch_reward[i:num])
            a_hat[i] = np.sum(a_discount * _delta[i:num])  # GAE

        return v_s_t, a_hat

    # ========== Tensorflow methods start ==========
    # It means that the input and output for these methods must be in tensor format
    # Predict Action and Policy Method (Tensor Format)
    @tf.function
    def getActionPolicyTensor(self, input, old_sample):
        # Get mean and variance
        _mean_tensor, _var_tensor = self.actor(input)
        # Get sample
        sample_tensor = tf.random.normal(shape=input.shape, mean=_mean_tensor, stddev=tf.sqrt(_var_tensor))
        # Get sample PDF
        pdf_tensor = (1/tf.sqrt(2*np.pi*_var_tensor)) * tf.exp((-1/(2*_var_tensor))*tf.square(old_sample - _mean_tensor))
        pdf_tensor = tf.clip_by_value(pdf_tensor, 1e-06, 1)
        # Return sample, policy, mean, and var
        return sample_tensor, pdf_tensor
    
    # Predict Value Function Method (Tensor Format)
    @tf.function
    def getValueFunctionTensor(self, input_tensor):
        return self.critic(input_tensor)
    # =========== Tensorflow methods end ===========

    # Training Method
    def training(self, minibatch:dict) -> bool:
        _nan_flag = False  # Initialise/Reset nan flag.

        # Get training batch from buffers management methods
        _batch_state_tensor = my_utils.tensorConversion(minibatch['state'])
        _batch_action_tensor = my_utils.tensorConversion(minibatch['action'])
        _batch_policy_tensor = my_utils.tensorConversion(minibatch['policy'])
        _batch_v_s_t_tensor = my_utils.tensorConversion(minibatch['sum_return'])
        _batch_a_hat_tensor = my_utils.tensorConversion(minibatch['advantage'])

        # ******************************************* Compute for the Gradient ***************************************************
        # Tensorflow gradient tape
        # ------------------ Tape_p: Updating the Policy Network (Actor) start ------------------
        with tf.GradientTape() as tape_p:
            _, new_sample_pdf = self.getActionPolicyTensor(_batch_state_tensor, _batch_action_tensor)
            # Probability Ratio
            prob_ratio = tf.math.exp(tf.math.log(new_sample_pdf) - tf.math.log(_batch_policy_tensor))
            # Unclipped term
            un_clipped = prob_ratio * _batch_a_hat_tensor
            # Clipped term
            clipped = tf.clip_by_value(prob_ratio, 1-EPSILON, 1+EPSILON) * _batch_a_hat_tensor
            # Cost Function
            policy_cost = -tf.reduce_mean(tf.minimum(un_clipped, clipped))
        # ------------------- Tape_p: Updating the Policy Network (Actor) end --------------------

        # --------------- Tape_q: Updating the Value-Function Network (Critic) start ------------------
        with tf.GradientTape() as tape_q:
            # Predict V
            v_pred = self.getValueFunctionTensor(_batch_state_tensor)
            # Loss
            critic_loss = tf.reduce_mean(keras.losses.mean_squared_error(_batch_v_s_t_tensor, v_pred))
        # ------------------- Tape_q: Updating the Value-Function Network (Critic) end ----------------

        # Check for NaN
        if np.isnan(policy_cost).any():
            print('NaN detected in the Actor! Skip the update for this epoch.')
            _nan_flag = True
            self.actor_glob_gradient = 0
            self.critic_glob_gradient = 0
        elif np.isnan(critic_loss).any():
            print('NaN detected in the Critic! Skip the update for this epoch')
            _nan_flag = True
            self.actor_glob_gradient = 0
            self.critic_glob_gradient = 0
        else:
            # If there is no NaN detected, ...
            # Compute for the actor's gradients
            actor_gradients = tape_p.gradient(policy_cost, self.actor.trainable_variables)
            # Compute for the critic's gradients
            critic_gradients = tape_q.gradient(critic_loss, self.critic.trainable_variables)
            # Apply Gradients
            self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            # Get global gradients norm
            self.actor_glob_gradient = tf.linalg.global_norm(actor_gradients).numpy()
            self.critic_glob_gradient = tf.linalg.global_norm(critic_gradients).numpy()
        return _nan_flag
    
    # Training Status Printing Method
    def statusPrinting(self, episode, trajectory, compute_time):
        print('Episode: ', self.training_record+1, ' / ', episode,
              ', Trajectory: ', self.trajectory, ' / ', trajectory,
              ', Episode_Reward: ', self.episode_reward, 
              ', Avg_Episode_Reward: ', self.avg_reward,
              ', Actor_Grad: ', self.actor_glob_gradient,
              ', Critic_Grad: ', self.critic_glob_gradient, 
              ', Computation_time: ', compute_time)
    
    # Trained models saving Method
    def trainingSaving(self, training_save=True, model_save=True):
        # ----- Save the training record
        if training_save:
            np.savez(self.TRAINING_PATH,
                    self.sum_reward_append,
                    self.training_record)
        if model_save:
            self.actor.save(self.ACTOR_PATH)
            self.critic.save(self.CRITIC_PATH)
    
    # Testing record saving method
    def testSaving(self):
        # ----- Save the testing record for evaluation
        np.savez(self.TEST_PATH,
                 self.av_speed_append,
                 self.avg_speed_append)

# ******************************************************************** PPO Agent Class End *******************************************************************************
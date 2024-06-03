# Agent Script for Autonomous Vehicle Control (AV) (Single-Agent)
# Version: 2
# Algorithm: PPO (Continuous)
# Remark: No critic, Use sum reward instead of advantages function (Same configuration as the original paper (Alex Bayen))

# ****************************************************************** SUMO related headers ********************************************************************************
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary
# ****************************************************************** SUMO related headers ********************************************************************************

# ********************************************************************** Other headers ***********************************************************************************
import tensorflow as tf  # Tensorflow
import keras
import numpy as np   # Numpy
import tensorflow_probability as tfp
import random
# ********************************************************************** Other headers ***********************************************************************************

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
        self.revision = 2  # Revision
        self.vehicle_id = id  # Get the vehicle'id
        self.vehicle_type = 'rl_agent'  # vehicle_type in the sumo config.
        self.sumo_time_step = time_step  # sumo time step
        self.collector_config = collector  # collector
        self.action_config = action_config  # action configuration (discrete or continuous)
        self.reward_objective = reward_objective  # reward objective (greedy or global)
        self.reward_config = reward_config  # reward configuration (original or my_reward)
        self.policy_objective = policy_objective  # policy objective (advantages or sum_reward)
        # State 
        self.state = np.zeros([3], dtype=float)  # State (S_t)
        self.next_state = self.state  # Next state (S_{t+1})
        # Action Space Parameters
        self.decel = -0.5  # Define decel of this vehicle.
        self.accel = 0.5  # Define accel of this vehicle.
        self.max_speed = 10  # Define max speed of this vehicle.
        self.min_speed = 0  # Define min speed of this vehicle.
        self.action = [0]  # Action (a_t)
        self.action_policy = [0]  # Policy of Action (Probability) (\pi_t)
        self.lead_dist = 100  # Observe distance of leading vehicle
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
        self.index_counter = 0  # Agent index counter (used for storing experiences.)
        self.av_speed_append = []  # Storage for av's speed (For visualisation)
        self.avg_speed_append = []  # Storage for avg's speed of all vehs (For visualisation)
        self.trajectory = 0  # Trajectory counter.

        # Initialise observation buffers (For storing the trajectories)
        self.observation_size = int(observation_budget*horizon)
        self.state_obs = np.zeros([self.observation_size, len(self.state)], dtype=float)
        self.action_obs = np.zeros([self.observation_size], dtype=float)  # Store longitudinal actions
        self.action_policy_obs = np.zeros([self.observation_size, len(self.action_policy)], dtype=float)  # Store \pi_t|a_t
        self.reward_obs = np.zeros([self.observation_size], dtype=float) 
        self.termination_obs = np.zeros([self.observation_size], dtype=bool)
        self.next_state_obs = np.zeros([self.observation_size, len(self.next_state)], dtype=float)
        # Initialise batch observation
        # Use batch buffer because we want to collect a long horizon
        self.batch_proportion = 1
        self.batch_observation_size = int(self.observation_size / self.batch_proportion)
        self.batch_state_obs = np.zeros([self.batch_observation_size, len(self.state)], dtype=float)
        self.batch_action_obs = np.zeros([self.batch_observation_size], dtype=float)
        self.batch_action_policy_obs = np.zeros([self.batch_observation_size, len(self.action_policy)], dtype=float)
        self.batch_reward_obs = np.zeros([self.batch_observation_size], dtype=float)
        self.batch_termination_obs = np.zeros([self.batch_observation_size], dtype=bool)
        self.batch_next_state_obs = np.zeros([self.batch_observation_size, len(self.next_state)], dtype=float)
        
        # Check if this is the first time training or not ?
        # If load == False --> this is the first time training --> create new array for these information
        if not load:
            self.sum_reward_append = []
            self.action_policy_append = []  # Append of policy (To visualise the training curve)
            self.training_record = 0  # For on-policy, the training record will be started from zero(0).
        # If load == True --> Continue training --> Load the existing training_record.
        else:
            saved_data = np.load('src/single_ring/training_record/{:s}{:d}_{:s}_single_ring_training_record.npz'.format(self.alg, self.revision, self.reward_objective))
            self.sum_reward_append = np.squeeze(saved_data['arr_0'])
            self.action_policy_append = np.squeeze(saved_data['arr_1'])
            self.training_record = np.squeeze(saved_data['arr_2'])

        # Initialise networks
        # It is the actor-critic framework in PPO.
        # Creating the function approximator (neural networks)
        if not load:
            # If load == False --> Create new networks.
            self.actor = self.network_creation()
        else:
            # If load == True --> load the existing networks.
            self.actor = keras.models.load_model('src/single_ring/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective))

        # Intialise network parameters
        self.actor_glob_gradient = 0  # Global gradient of the actor network (For visualisation only).
        self.policy_cost = 0  # Policy_cost (for visualisation only).

    # Network creation method
    # Same network's architecture as the original paper.
    def network_creation(self):
        # Defining the hyperparameters
        num_layer = 64
        learning_rate = 3e-4

        # Input layer
        input = keras.layers.Input(shape=(len(self.state),))

        # Fully Connected layers
        dense_1 = keras.layers.Dense(num_layer, activation='linear')(input)
        dense_2 = keras.layers.Dense(num_layer, activation='linear')(dense_1)
        dense_3 = keras.layers.Dense(num_layer, activation='linear')(dense_2)

        # Output layer
        actor_output_1 = keras.layers.Dense(1, activation='linear')(dense_3)  # Mean of actor's output
        actor_output_2 = keras.layers.Dense(1, activation='softplus')(dense_3)  # Variance of actor's output

        # Determine the model
        # Actor -- Policy Network
        actor = keras.models.Model(inputs=input, outputs=[actor_output_1, actor_output_2], name='policy_network')
        actor.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        
        # Return the model
        return actor

    # State Retrieving Method
    def get_state(self, noise=False):
        # ----- State condition 1 (Follow Alex Bayen's paper):
        # state[0] = # Speed of the AV (m/s)
        # state[1] = # Speed of vehicle in front (m/s)
        # state[2] = # Distance between AV and the vehicle in the front. (m)
        state = np.zeros([3], dtype=float)
        state[0] = traci.vehicle.getSpeed(vehID=self.vehicle_id)  # Get It's speed (m/s)
        # Get Leader
        leader_info = traci.vehicle.getLeader(vehID=self.vehicle_id, dist=self.lead_dist)
        # Check if there is a vehicle in front ?
        if leader_info != None:
            # Yes -- There is
            state[1] = traci.vehicle.getSpeed(vehID=leader_info[0])
            state[2] = leader_info[1]
        else:
            # No -- There is not
            state[1] = self.max_speed  # Assign the maximum speed (Assume that the leader is faster than AV.)
            state[2] = self.lead_dist  # Assign distance = maximum observeable distance.
        
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
    
    # Action Performing Method
    def action_perform(self, state, action=None, mode='training'):
        dummy = False
        if action == None:
            # Get Action (Predict from the policy network)
            action_mu, action_var = self.actor(tf.convert_to_tensor(np.expand_dims(state, axis=0)))
            action_dist = tfp.distributions.Normal(loc=action_mu, scale=tf.math.sqrt(action_var))
            action_sample = tf.clip_by_value(action_dist.sample(), self.decel, self.accel)
            action_policy = action_dist.prob(action_sample)
        else:
            # If the action is pre-defined, It means that this is the warm-up step.
            action_sample = action
            action_policy = tf.constant([1], dtype=tf.float32)

        # Store action_policy (For visualisation)
        self.action_policy_append = np.append(self.action_policy_append, action_policy.numpy()[0])

        # Initialise Duration for speed changing
        duration = int(1/self.sumo_time_step)  # 1s/sumo_time_step(0.1s)
        # Convert to numpy
        # action_long = longitudinal_action_tensor.numpy()[0]
        action_long = action_sample.numpy()[0]
        # Retrieve Desired Speed
        desired_speed = np.clip(a=(state[0]+action_long), a_min=0, a_max=self.max_speed)
        # Perform longitudinal action --> Use 'slowDown' command
        traci.vehicle.slowDown(vehID=self.vehicle_id, speed=desired_speed, duration=duration)
        # Perform lateral action (If lane changing is not permitted.)
        # Return action and policy
        # Note: Return the action and policy from the model prediction before clipping
        return action_sample.numpy()[0], action_policy.numpy()[0], dummy
    
    # Observation storing method (Observation)
    def observation_storing(self, index, state, action, policy, reward, termination, next_state):
        # ------------- Store the tuples (S_t, a_t, death_flag, r_{t+1}, S_{t+1}) in the Experience Relay Buffer
        # Check if the step count is over the buffer size, reset to zero and set the full flag.
        self.state_obs[index] = state  # S_t
        self.action_obs[index] = action  # a_t
        self.action_policy_obs[index] = policy  # \pi_t|a_t
        self.reward_obs[index] = reward  # r_{t+1}
        self.termination_obs[index] = termination  # death_flag
        self.next_state_obs[index] = next_state  # S_{t+1}

    # Reward Function Method
    def reward_function(self, noise=False, n_veh=44):
        # Define hyperparameters
        gamma = 0.99  # Define discount factor
        sum_speed = 0  # Reset sum_speed (For global objective)
        num_convolute = 10  # Convolution kernel size (1-D) (Used for computing running mean/std)
        # Get next state first
        next_state = self.get_state(noise)
        
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
            self.accum_reward = (gamma*self.accum_reward) + reward
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

    # Training Method
    def training(self, gradient_steps):
        for _ in range(gradient_steps):
            # Define hyperparameters
            gamma = 0.99  # Discount factor
            epsilon = 0.2  # Epsilon used for limit the moving of probability ratio

            # Check for the policy objective (Use advantages function or sum_reward)
            if self.policy_objective == 'advantages':
                # Create the advantage function storage.
                a_hat = np.zeros([self.batch_observation_size], dtype=float)  # Advantages function
                v_s_t = np.zeros([self.batch_observation_size], dtype=float)  # First term of a_hat
                # Third term
                v_s_T = self.reward_obs[self.batch_observation_size-1]  # Getting the last element in the observations.
                # Compute for the advantages function and the expected future return
                # Assign the expected future return by the bellman's equation.
                for j in range(self.batch_observation_size):
                    # Derived V_{s_t} (First term) for t = {0, 1, 2, ..., T-j} ; T = experience_size
                    for l in range(self.batch_observation_size - j):
                        v_s_t += (gamma ** l) * self.reward_obs[l + j]
                    # Advantages Function (A_hat)
                    for k in range(self.batch_observation_size - 1):
                        if k + j > (self.batch_observation_size - 1):
                            a_hat[j] = a_hat[j]
                        else:
                            a_hat[j] = a_hat[j] + (gamma ** k) * self.reward_obs[k + j]
                    # Derive A_hat
                    a_hat[j] = -v_s_t + a_hat[j] + (gamma**(self.batch_observation_size - j)) * v_s_T
            
            # Use sum_reward (Same as the original paper)
            elif self.policy_objective == 'sum_reward':
                v_s_t = np.zeros([self.batch_observation_size], dtype=float)  # First term of a_hat
                # Derive Expected Future Return (v_s_t)
                for t in range(self.observation_size):
                    for k in range(self.observation_size-t):
                        v_s_t[t] += (gamma**k) * self.reward_obs[t+k]
            else:
                print('No reward objective found. Please try again.')
                exit()  # Terminate the script.

            # ******************************************* Compute for the Gradient ***************************************************
            # **** Prepare the data *****
            state_obs_tensor = tf.convert_to_tensor(self.state_obs, dtype=tf.float32)
            old_policy_tensor = tf.convert_to_tensor(self.action_policy_obs, dtype=tf.float32)
            # a_hat_tensor = tf.convert_to_tensor(a_hat, dtype=tf.float32)
            # v_s_t_tensor = tf.convert_to_tensor(v_s_t, dtype=tf.float32)

            # Tensorflow gradient tape
            # ------------------ Tape_1: Updating the Policy Network (Actor) start ------------------
            with tf.GradientTape() as tape_1:
                # Bring the neccessary variables into the tape.
                tape_1.watch(state_obs_tensor)
                tape_1.watch(old_policy_tensor)
                # tape_1.watch(v_s_t_tensor)
                # Predict policies from the current actor model.
                new_goal_mu, new_goal_var = self.actor(state_obs_tensor)
                new_goal_dist = tfp.distributions.Normal(loc=new_goal_mu, scale=tf.math.sqrt(new_goal_var))
                new_policy_tensor = new_goal_dist.prob(new_goal_dist.sample())  # Get the policies
                # Probability Ratio
                prob_ratio = tf.math.divide(new_policy_tensor, old_policy_tensor)
                # prob_ratio = tf.math.exp(tf.math.log(new_policy_tensor) - tf.math.log(old_policy_tensor))
                # Unclipped Term
                un_clipped = prob_ratio * v_s_t
                # Clipped Term
                clipped = tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon) * v_s_t
                # Cost Function
                policy_cost = -tf.reduce_mean(tf.math.minimum(un_clipped, clipped))
            # ------------------- Tape_1: Updating the Policy Network (Actor) end --------------------

            # Assign cost to the numpy format (For visualisation only)
            self.policy_cost = policy_cost.numpy()
            # Check for NaN value
            if np.isnan(self.policy_cost).any():
                print('NaN detected! Skip the update for this epoch.')
                break
            else:
                # If there is no NaN ...
                # Compute for the gradient
                policy_cost_gradient = tape_1.gradient(policy_cost, self.actor.trainable_variables)
                # Apply Gradient
                self.actor.optimizer.apply_gradients(zip(policy_cost_gradient, self.actor.trainable_variables))
                # Compute for the global gradient (for visualisation only)
                policy_glob_gradient = tf.linalg.global_norm(policy_cost_gradient)
                self.actor_glob_gradient = policy_glob_gradient.numpy()

            # Update training record
            self.training_record += 1
            
    # Training Status Printing Method
    def status_printing(self, episode, trajectory):
        print('Episode: ', self.training_record+1, ' / ', episode,
              ', Trajectory: ', self.trajectory+1, ' / ', trajectory,
              ', Episode_Reward: ', self.episode_reward, 
              ', Avg_Episode_Reward: ', self.avg_reward,
              ', Policy_Cost: ', self.policy_cost,
              ', Actor_Grad: ', self.actor_glob_gradient)
    
    # Trained models saving Method
    def training_saving(self, training_save=True, model_save=True):
        # ----- Save the training record
        if training_save:
            np.savez('src/single_ring/training_record/{:s}{:d}_{:s}_single_ring_training_record'.format(self.alg, self.revision, self.reward_objective),
                    self.sum_reward_append,
                    self.action_policy_append,
                    self.training_record)
        if model_save:
            self.actor.save('src/single_ring/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective))

    # Testing record saving method
    def testing_saving(self):
        # ----- Save the testing record for evaluation
        np.savez('src/single_ring/testing_record/{:s}{:d}_{:s}_single_ring_testing_record'.format(self.alg, self.revision, self.reward_objective),
                 self.av_speed_append,
                 self.avg_speed_append,
                 self.action_policy_append)

# ******************************************************************** PPO Agent Class End *******************************************************************************

# ********************************************************************* Main Program Start *******************************************************************************
# if __name__ == "__main__":
#     # Initial command for opening the sumo interface.
#     sumoBinary = checkBinary("sumo-gui")
#     sumoCmd = [sumoBinary, "-c", "C:/Amornyos/PhD/Traffic_Flow_SUMO/Networks/single_ring/single_ring.sumocfg.xml", 
#             '--no-warnings', 
#             '--no-step-log']

#     # Start the program
#     traci.start(sumoCmd)
#     # Initialise agent
#     agent = Autonomous_Vehicle_Agent(load=None, file_address=None, id='AV_1')
#     step = 0
    
#     # Pre-define simulation environment
#     route_id = np.squeeze(traci.route.getIDList())  # Get all route's ID
#     veh_type = np.squeeze(traci.vehicletype.getIDList()[0])  # Get the vehicle type
    
#     # Generate AV in the simulation
#     traci.vehicle.add(vehID=agent.vehicle_id, routeID=route_id, typeID=veh_type)
    
#     # Generate Non-Controlled Vehicles in the simulation
#     for i in range(22):
#         # random_route = random.choice(route_id)
#         # random_veh_type = random.choice(veh_type)
#         traci.vehicle.add(vehID=i, routeID=route_id, typeID=veh_type)

#     # Simulation Step
#     while step < 1000:
#         traci.simulationStep()
#         # Get State
#         agent.state = agent.get_state()
#         # Get Action
#         action_mu, action_var = agent.actor(tf.convert_to_tensor(np.expand_dims(agent.state, axis=0)))
#         action_dist = tfp.distributions.Normal(loc=action_mu, scale=tf.math.sqrt(action_var))
#         action_sample = action_dist.sample()  # Tensor format
#         action_policy = action_dist.prob(action_sample)  # Tensor format
#         # Perform Action
#         agent.action_perform(longitudinal_action_tensor=action_sample)
#         # Print Action
#         print('Action: ', action_sample.numpy()[0], 'Policy: ', action_policy.numpy()[0])
#         # Step increment
#         step += 1

#     # Stop the simulation
#     traci.close()
# ********************************************************************** Main Program End ********************************************************************************
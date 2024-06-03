# Agent Script for Autonomous Vehicle Control (AV) (Single-Agent)
# Version: 1
# Algorithm: TRPO (Discrete)
# Remark: Same as rev3 but it is a discrete action space.

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
# ********************************************************************** Other headers ***********************************************************************************

# ******************************************************************* PPO Agent Class Start ******************************************************************************
class Autonomous_Vehicle_Agent():
    # Initialise Variables
    def __init__(self, 
                 load, 
                 time_step,
                 observation_budget, 
                 collector=False,
                 action_config='discrete',
                 reward_config='original',
                 reward_objective='greedy',
                 policy_objective='sum_reward',
                 id='AV'):
        self.alg = 'TRPO'  # Algorithm
        self.revision = 1  # Revision
        self.vehicle_id = id  # Get the vehicle'id
        self.vehicle_type = traci.vehicletype.getIDList()[0]  # In this case, AV is the first one so the index is 0.
        self.sumo_time_step = time_step  # sumo time step
        self.collector_config = collector  # collector (Use non-updated policy to collect trajectories.)
        self.action_config = action_config  # action configuration (discrete or continuous)
        self.reward_objective = reward_objective  # reward objective (greedy or global)
        self.reward_config = reward_config  # reward configuration (original or my_reward)
        self.policy_objective = policy_objective  # policy objective (advantages or sum_reward)
        # State 
        self.state = np.zeros([3], dtype=float)  # The state will be the same as controller (S_t)
        self.next_state = self.state  # Next state (S_{t+1})
        # Action Space Parameters
        self.decel = 0.5  # Define decel of this vehicle.
        self.accel = 0.5  # Define accel of this vehicle.
        self.max_speed = 10  # Define max speed of this vehicle.
        self.min_speed = 0  # Define min speed of this vehicle.
        self.min_gap = 2.0  # Define minimum clearance between vehicles.
        # Define action space (For discrete actions) --> A_long x A_lateral
        self.action_space = [-self.decel,  # Deceralation
                             0,  # Maintain current speed
                             self.accel]  # Acceleration
        self.action = 0  # Action (a_t)
        self.action_policy = np.zeros([len(self.action_space)], dtype=float)  # Policy of Action (Probability) (\pi_t)
        self.action_append = []  # Append of action (To visualise the policy function.)
        self.action_policy_append = []  # Append of policy (To visualise the training curve)
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
        self.avg_speed_append = []  # Storage for avg speed of all vehicle in the environment. (For visualisation only)

        # Initialise observation buffers
        self.observation_size = observation_budget  # 40 <= B <= 45
        # If load == False --> this is the first time training --> create new array for these information
        self.state_obs = np.zeros([self.observation_size, len(self.state)], dtype=float)
        self.action_obs = np.zeros([self.observation_size], dtype=float)  # Store actions
        self.action_policy_obs = np.zeros([self.observation_size, len(self.action_policy)], dtype=float)  # Store \pi_t|a_t
        self.reward_obs = np.zeros([self.observation_size], dtype=float) 
        self.termination_obs = np.zeros([self.observation_size], dtype=bool)
        self.next_state_obs = np.zeros([self.observation_size, len(self.next_state)], dtype=float)
        
        # Initialise batch observation buffers to train the model
        # Use batch buffer because we want to collect a long horizon.
        self.batch_proportion = 1
        self.batch_size = int(self.observation_size / self.batch_proportion)
        self.batch_state_obs = np.zeros([self.batch_size, len(self.state)], dtype=float)
        self.batch_action_obs = np.zeros([self.batch_size], dtype=float)
        self.batch_action_policy_obs = np.zeros([self.batch_size, len(self.action_policy)], dtype=float)
        self.batch_reward_obs = np.zeros([self.batch_size], dtype=float)
        self.batch_termination_obs = np.zeros([self.batch_size], dtype=bool)
        self.batch_next_state_obs = np.zeros([self.batch_size, len(self.next_state)], dtype=float)

        # Check if this is the first time training or not ?
        # If load == False --> this is the first time training --> create new array for these information
        if not load:
            self.sum_reward_append = []
            self.training_record = 0  # For on-policy, the training record will be started from zero(0).
        else:
            saved_data = np.load('src/single_ring/training_record/{:s}{:d}_{:s}_single_ring_training_record.npz'.format(self.alg, self.revision, self.reward_objective))
            self.sum_reward_append = np.squeeze(saved_data['arr_0'])
            self.training_record = np.squeeze(saved_data['arr_1'])
            
        # No Experience Replay Buffers required for the on-policy methods.

        # Initialise networks
        # It is the actor-critic framework in PPO.
        # Creating the function approximator (neural networks)
        if not load:
            # If load == False --> Create new networks.
            self.actor = self.network_creation()  # Create actor
            self.collector = self.network_creation()  # Create collector (non-update actor)
        else:
            # If load == True --> load the existing networks.
            self.actor = keras.models.load_model('src/single_ring/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective))
            self.collector = self.actor # Assign the collector's weights same as the trained actor.

        # Intialise network parameters
        self.collector.set_weights(self.actor.get_weights())
        self.actor_glob_gradient = 0  # Global gradient of the actor (For visualisation only).
        self.trajectory = 0  # Trajectory counter
        # ========  Hyperparameters for the TRPO (Use the same as original paper) 
        self.cg_damping = 0.1        # ??? something related to the conjugate gradient method.
        self.cg_iters = 10           # ??? something related to the conjugate gradient method.
        self.backtrack_iters = 10    # ??? something related to the conjugate gradient method.
        self.backtrack_coeff = 0.6   # ??? something related to the conjugate gradient method.
        self.delta = 0.01  # delta coefficient (constraint).
        self.residual_tol = 1e-5
        self.temp_actor = self.network_creation()  # Temporary actor (Used for updating gradient only.)

    # Load trained models Method
    def load_models(self, agent):
        self.actor = keras.models.load_model('src/single_ring/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective))
        self.collector.set_weights(self.actor.get_weights())

    # Network creation method
    def network_creation(self):
        # Defining the hyperparameters
        num_layer = 64
        # learning_rate = 3e-4

        # Input layer
        input = keras.layers.Input(shape=(len(self.state),))
        # Fully Connected layers
        dense_1 = keras.layers.Dense(num_layer)(input)
        dense_2 = keras.layers.Dense(num_layer)(dense_1)
        dense_3 = keras.layers.Dense(num_layer)(dense_2)
        # Output layer
        actor_output = keras.layers.Dense(len(self.action_space), 
                                          activation='softmax')(dense_3)  # Mean of actor's output
        # Determine the model
        # Actor -- Policy Network
        actor = keras.models.Model(inputs=input, outputs=actor_output, name='policy_network')
        # actor.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003))   
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
            state[1] = self.min_speed  # Assign the maximum speed (Assume that the leader is faster than AV.)
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
    def action_perform(self, state, action=None):
        dummy = None
        if action == None:
            # Get Action (Predict from the policy network)
            # Network return probabilities of all outputs.
            # Check if using the old or new policy to collect trajectories.
            if self.collector_config:
                # Use old policy to collect trajectories (Base training)
                policies = self.collector(tf.convert_to_tensor(np.expand_dims(state, axis=0)))
            else:
                # Use new policy to collect trajectories (Online training or Test)
                policies = self.actor(tf.convert_to_tensor(np.expand_dims(state, axis=0)))
            action_index = np.random.choice(np.arange(len(self.action_space)), p=policies.numpy()[0])
            action_long = self.action_space[action_index]  # Get longitudinal action (Acceleration)
            action_policy = policies.numpy()[0] # ****** Use this (in numpy) to store policy (prob)
        else:
            # If the action is pre-defined, It means that this is the warm-up step.
            action_long = action.numpy()
            action_policy = np.zeros([len(self.action_policy)], dtype=bool)
            action_index = 0

        # Initialise Duration for speed changing
        duration = int(1/self.sumo_time_step)  # 1s/sumo_time_step(0.1s)
        # Retrieve Desired Speed
        desired_speed = np.clip(a=(state[0]+action_long), a_min=self.min_speed, a_max=self.max_speed)
        # Perform longitudinal action --> Use 'slowDown' command
        traci.vehicle.slowDown(vehID=self.vehicle_id, speed=desired_speed, duration=duration)
        # Perform lateral action (If lane changing is not permitted.)
        # Return action and policy
        # Note: Return the action and policy from the model prediction before clipping
        return action_index, dummy, action_policy
    
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
    def reward_function(self, noise=False, n_veh=22):
        # Define hyperparameters
        gamma = 0.99  # Define discount factor
        sum_speed = 0  # Reset sum_speed (For global objective)
        num_convolute = 10  # Convolution kernel size (1-D) (Used for computing running mean/std)
        # Get next state first
        next_state = self.get_state(noise)

        # Check reward configurations (Original or My own design)
        # ------ Alex Bayen's Paper ------
        if self.reward_config == 'original':
            if self.reward_objective == 'greedy':
                # Using the greedy objective since it might be feasible in real-world. (Just my assumption)
                # Reward = AV's speed (Greedy policy)
                reward = next_state[0]  
            else:
                # Reward = All vehicles average speed (Global policy)
                for c in range(n_veh-1):
                  sum_speed += traci.vehicle.getSpeed(vehID=str(c))
                  avg_speed = (sum_speed + next_state[0]) / n_veh
                reward = avg_speed
                # reward = (next_state[0]+next_state[2]+next_state[4]+next_state[6]+next_state[8])/5
            # Append Reward
            self.reward_append = np.append(self.reward_append, reward)
            # Accumalated Reward
            self.accum_reward = (gamma * self.accum_reward) + reward 
            self.accum_reward_append = np.append(self.accum_reward_append, self.accum_reward)
            # Compute the running mean/std
            if len(self.reward_append) <= num_convolute:
                mean = np.mean(self.reward_append[0:num_convolute])
                std = np.std(self.accum_reward_append[0:num_convolute])
            else:
                m = len(self.reward_append) - num_convolute
                mean = np.mean(self.reward_append[0+m:num_convolute+m])
                std = np.std(self.accum_reward_append[0+m:num_convolute+m])
            # Normalise the reward (To reduce the variance of PG)
            if std != 0:
                norm_reward = (reward - mean) / std
            else:
                norm_reward = 0
        # ------ My Reward Fn ------
        elif self.reward_config == 'my_reward':
            # ------ Greed Objective -------
            if self.reward_objective == 'greedy':
                reward = next_state[0]  # Get AV's speed.
            # # ------ Global Objective ------
            elif self.reward_objective == 'global':
                reward = (next_state[0]+next_state[2]+next_state[4]+next_state[6]+next_state[8])/5
            else:
                print('Reward objective is unknown or not provided. Please chose one either greedy or global and try again.')
                exit()  # The script is terminated.
            # Append Reward
            self.reward_append = np.append(self.reward_append, reward)
            # Derive mean
            if len(self.reward_append) <= num_convolute:
                mean_t = np.mean(self.reward_append)
            else:
                m = len(self.reward_append)-num_convolute
                mean_t = np.mean(self.reward_append[0+m:num_convolute+m])
            # Derive distance penalty
            # This is the penalty when the av is too far behind from the leading vehicle.
            dis_penalty = 1 - ((next_state[2] - self.min_gap)/self.lead_dist)
            # dis_penalty = 1
            # Normalised the reward
            # norm_reward = np.mean(self.reward_append) / self.max_speed
            norm_reward = (mean_t / self.max_speed) * dis_penalty
        else:
            print('No reward configuration is provided or wrong identification, Please try again.')
            exit()
        # Return tuples
        return norm_reward, next_state

    # Flatten model's variables function
    def flatvars(self, model):
        return tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

    # Assign weigth function
    def assign_vars(self, model, theta):
        shapes = [v.shape.as_list() for v in model.trainable_variables]
        size_theta = np.sum([np.prod(shape) for shape in shapes])
        start = 0
        for i, shape in enumerate(shapes):
            size = np.prod(shape)
            param = tf.reshape(theta[start:start + size], shape)
            model.trainable_variables[i].assign(param)
            start += size
        assert start == size_theta, 'messy shapes'

    # Training Method
    def training(self, experience_size, epoch):
        # Define hyperparameters
        gamma = 0.99  # Discount factor
        my_lambda = 0.95
        # epsilon = 0.2  # Epsilon used for limit the moving of probability ratio
        dummy = 0  # Dummy parameter (To common code)
        # Indices array stores taken actions (Used for slicing indices in the new policy during policy gradient updating.)
        slice_indices = np.zeros([self.batch_size, 2])
        slice_indices[:, 0] = np.arange(self.batch_size)

        # Define batch size (1/10th of the buffers)
        for b in range(self.batch_proportion):
            # Sampling for 1/10th of the whole observation buffers.
            self.batch_state_obs = self.state_obs[self.batch_size*(b+0) : self.batch_size*(b+1)]
            self.batch_action_obs = self.action_obs[self.batch_size*(b+0) : self.batch_size*(b+1)]
            self.batch_action_policy_obs = self.action_policy_obs[self.batch_size*(b+0) : self.batch_size*(b+1)]
            self.batch_reward_obs = self.reward_obs[self.batch_size*(b+0) : self.batch_size*(b+1)]
            self.batch_next_state_obs = self.next_state_obs[self.batch_size*(b+0) : self.batch_size*(b+1)]

            # Derive the expected sum reward (sum_reward or advantages function depended on the policy_objective argument.)
            # Check the policy objective.
            # If the policy objective is advantages, Derive Advantages Function.
            if self.policy_objective == 'advantages':
                # Advantages function storage
                a_hat = np.zeros([self.batch_size], dtype=float)  # Advantages function array.
                v_s_t = np.zeros([self.batch_size], dtype=float) # First term of a_hat
                td_term = np.zeros([self.batch_size], dtype=float)  # TD term
                v_s_T = self.batch_reward_obs[-1]  # Last term of a_hat

                # Compute for the advantages function rev1
                # # Assign the expected future return by the bellman's equation.
                # for j in range(self.batch_size):
                #     # Derived V_{s_t} (First term) for t = {0, 1, 2, ..., T-j} ; T = experience_size
                #     for l in range(self.batch_size - j):
                #         v_s_t += (gamma ** l) * self.batch_reward_obs[l + j]
                #     # Advantages Function (A_hat)
                #     for k in range(self.batch_size - 1):
                #         if k + j > (self.batch_size - 1):
                #             a_hat[j] = a_hat[j]
                #         else:
                #             a_hat[j] = a_hat[j] + (gamma ** (k-j+1)) * self.batch_reward_obs[k + j]
                #     # Derive A_hat
                #     a_hat[j] = -v_s_t + a_hat[j] + (gamma**(self.batch_size - j)) * v_s_T
            
                # Compute Advantage function rev2 (Generalised Advantages Estimation)
                # Derive td_term
                for t in range(self.batch_size):
                    for k in range(self.batch_size-t):
                        v_s_t[t] += (gamma ** k) * self.batch_reward_obs[t+k]
                    if t+1 >= self.batch_size:
                        td_term[t] = self.batch_reward_obs[-1]
                    else:
                        td_term[t] = self.batch_reward_obs[t] + (gamma*self.batch_reward_obs[t+1]) - v_s_t[t]
                # Derive Advantages Function
                for l in range(self.batch_size):
                    for m in range(self.batch_size-l):
                        a_hat[l] += ((gamma*my_lambda)**l)*td_term[l+m]
                # Convert to tensor 
                sum_reward_tensor = tf.convert_to_tensor(a_hat, dtype=tf.float32)

            # If policy objective is 'sum_reward', Use sigma(gamma*r) (Same as original paper)
            elif self.policy_objective == 'sum_reward':
                # Sum of Expected Return storage
                v_s_t = np.zeros([self.batch_size], dtype=float) 
                # Derive v_s_t
                for t in range(self.batch_size):
                    for k in range(self.batch_size-t):
                        v_s_t[t] += (gamma ** k) * self.batch_reward_obs[t+k]            
                # Convert to tensor
                sum_reward_tensor = tf.convert_to_tensor(v_s_t, dtype=tf.float32)
            else:
                # No policy objective is provided or unknown.
                print('Policy objective is unknown or not provided. Please chose one either advantages or sum_reward and try again.')
                exit()

            # ******************************************* Gradient Updating Start ***************************************************
            # ---------- Prepare the data 
            state_obs_tensor = tf.convert_to_tensor(self.batch_state_obs, dtype=tf.float32)  # Convert state observation to tensor format
            old_policy_tensor = tf.convert_to_tensor(self.batch_action_policy_obs, dtype=tf.float32)  # Convert policy observation to tensor format
            slice_indices[:, 1] = self.batch_action_obs  # Store action observation in the slice_indices variable (Will be used to find surrogate cost)

            # ++++++++++ Relevant Methods for Gradients updating 
            # ----- Derive surrogate cost
            def surrogate_loss(theta=None):
                # If theta is not provided, predict the current policy.
                if theta is None:
                    model = self.actor
                else:
                    model = self.temp_actor
                    self.assign_vars(model, theta)
                # new_policy = model(state_obs_tensor)
                # new_policy = tf.reduce_sum(new_policy, axis=1)
                new_policy = tf.gather_nd(model(state_obs_tensor), indices=slice_indices.astype(int))
                old_policy = self.actor(state_obs_tensor)
                # old_policy = tf.math.add(tf.reduce_sum(old_policy, axis=1), 1e-8)
                old_policy = tf.math.add(tf.gather_nd(old_policy, indices=slice_indices.astype(int)), 1e-8)
                prob_ratio = new_policy / old_policy
                # loss = tf.reduce_mean(prob_ratio * a_hat_tensor)
                loss = tf.reduce_mean(prob_ratio * sum_reward_tensor).numpy()
                return loss
            
            # ----- Derive the kl-divergence
            def kl_fn(theta=None):
                if theta is None:
                    model = self.actor
                else:
                    model = self.temp_actor
                    self.assign_vars(model, theta)
                # new_policy = tf.gather_nd(model(state_obs_tensor), indices=slice_indices.astype(int))
                new_policy = model(state_obs_tensor)
                new_policy = tf.math.add(new_policy, 1e-8)
                old_policy = self.actor(state_obs_tensor)
                # old_policy = tf.gather_nd(old_policy, indices=slice_indices.astype(int))
                return tf.reduce_mean(tf.reduce_sum(old_policy * tf.math.log(old_policy/new_policy), axis=1)).numpy()
            
            # ----- Compute hessian_vector_product (To solve H^(-1) matrix)
            def hessian_vector_product(p):
                with tf.GradientTape() as tape_2:
                    with tf.GradientTape() as tape_3:
                        # Get new policy 
                        new_policy_tensor_2 = self.actor(state_obs_tensor) + (1e-8)
                        kl_fn = tf.reduce_sum(old_policy_tensor * tf.math.log(old_policy_tensor/new_policy_tensor_2), axis=1)
                    kl_grad_vector = tape_3.gradient(kl_fn, self.actor.trainable_variables)
                    kl_grad_vector = tf.concat([tf.reshape(g, [-1]) for g in kl_grad_vector], axis=0)
                    # grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
                    grad_vector_product = tf.transpose(kl_grad_vector) * p
                fisher_vector_product = tape_2.gradient(grad_vector_product, self.actor.trainable_variables)
                fisher_vector_product = tf.concat([tf.reshape(g, [-1]) for g in fisher_vector_product], axis=0).numpy()
                # return fisher_vector_product + (self.cg_damping * p)
                return fisher_vector_product

            # ----- Compute conjugate gradient (To solve H^(-1) matrix)
            def conjugate_grad(Ax, b):
                x = np.zeros_like(b)
                r = b.copy()
                p = r.copy()
                old_p = p.copy()
                r_dot_old = np.dot(r, r)
                for _ in range(self.cg_iters):
                    z = Ax(p)
                    alpha = r_dot_old / (np.dot(p, z) + 1e-8)
                    old_x = x
                    x += alpha * p
                    r -= alpha * z
                    r_dot_new = np.dot(r, r)
                    beta = r_dot_new / (r_dot_old + 1e-8)
                    r_dot_old = r_dot_new
                    if r_dot_old < self.residual_tol:
                        break
                    old_p = p.copy()
                    p = r + beta*p
                    if np.isnan(x).any():
                        print("x is nan")
                        print("z", np.isnan(z))
                        print("old_x", np.isnan(old_x))
                        print("kl_fn", np.isnan(kl_fn()))
                return x
            
            # Compute linesearch
            def linesearch(x, beta, step):
                fval = surrogate_loss(x)
                improve_thresh = np.dot(np.mean(policy_gradient.numpy()), beta) * 0.1
                max_step = beta * step
                for (_n_backtracks, stepfrac) in enumerate(self.backtrack_coeff**np.arange(self.backtrack_iters)):
                    xnew = x + stepfrac * max_step
                    newfval = surrogate_loss(xnew)
                    kl_div = kl_fn(xnew)
                    if kl_div <= self.delta and newfval - fval >= improve_thresh :
                    # if kl_div <= self.delta and newfval > fval:
                        print("Linesearch worked at ", _n_backtracks, ' kl_div = ', kl_div, ' surrogate = ', newfval)
                        return xnew
                    if _n_backtracks == self.backtrack_iters-1:
                        print('Linesearch failed.', ' kl_div = ', kl_div, 'surrogate = ', newfval)
                    improve_thresh /= 2  # Reduce maximum step length exponentially
                    max_step /= 2
                return x

            # Derive policy gradient and kl_gradient
            with tf.GradientTape() as tape_1:
                # Predict new policy (\theta_{new})
                new_policy_tensor_slice = tf.gather_nd(self.actor(state_obs_tensor), indices=slice_indices.astype(int))
                # Slice Old policy
                old_policy_tensor_slice = tf.gather_nd(old_policy_tensor, indices=slice_indices.astype(int))
                # Derive prob ratio
                prob_ratio = new_policy_tensor_slice / old_policy_tensor_slice
                # surrogate = tf.reduce_mean(prob_ratio * a_hat_tensor)  # surrogate_cost
                surrogate = tf.reduce_mean(prob_ratio * sum_reward_tensor)
            # Policy gradient
            policy_gradient = tape_1.gradient(surrogate, self.actor.trainable_variables)
            policy_gradient = tf.concat([tf.reshape(g, [-1]) for g in policy_gradient], axis=0)  # Flatten the gradient to 1-D
            # Derive the A^(-1)g term by using the conjugate gradient technique.
            step_direction = conjugate_grad(hessian_vector_product, policy_gradient.numpy())  # s = A^(-1)g
            # Derive max-step-length (beta) rev1
            # shs = 0.5 * step_direction.dot(hessian_vector_product(step_direction).T)
            # lm = np.sqrt(shs / self.delta) + 1e-8
            # fullstep = step_direction / lm
            # Derive max-step-length (beta) rev2
            beta = np.sqrt(2*self.delta/np.dot(np.transpose(step_direction), hessian_vector_product(step_direction)))
            # fullstep = beta * step_direction

            if np.isnan(beta).any():
                print('fullstep is NaN')
                # print('lm ', lm)
                print('step_direction ', step_direction)
                print('policy_gradient ', policy_gradient.numpy())
            # Get the weigths of current approximator (old policy)
            oldtheta = self.flatvars(self.actor).numpy()
            # Derive for the new weights (new policy) using linesearch
            theta = linesearch(oldtheta, beta, step_direction)
            if np.isnan(theta).any():
                print('NaN detected. Skipping update...')
            else:
                self.assign_vars(self.actor, theta)
            kl = kl_fn(oldtheta)
        # Return global gradient value (for visualisation only)
        return self.actor_glob_gradient, dummy

    # Training Status Printing Method
    def status_printing(self, episode, trajectory=1):
        print('Episode: ', self.training_record+1, ' / ', episode,
              ', Trajectory: ', self.trajectory+1, ' / ', trajectory,
              ', Episode_Reward: ', self.episode_reward, 
              ', Avg_Episode_Reward: ', self.avg_reward,
              ', Actor_Grad: ', self.actor_glob_gradient)
    
    # Trained models saving Method
    def training_saving(self, training_save=True, model_save=True):
        if training_save:    
            # ----- Save the training record
            np.savez('src/single_ring/training_record/{:s}{:d}_{:s}_single_ring_training_record'.format(self.alg, self.revision, self.reward_objective),
                    self.sum_reward_append,  # 0
                    self.training_record)    # 1
        if model_save:
            self.actor.save('src/single_ring/trained_model/{:s}{:d}_{:s}_single_ring_actor.keras'.format(self.alg, self.revision, self.reward_objective))

    # Testing record saving method
    def testing_saving(self):
        np.savez('src/single_ring/testing_record/{:s}{:d}_{:s}_single_ring_testing_record'.format(self.alg, self.revision, self.reward_objective),
                 self.av_speed_append,       # 0
                 self.avg_speed_append,      # 1
                 self.action_policy_append,) # 2

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
    #     # random_veh_type = random.choice(veh_type)
    #     traci.vehicle.add(vehID=i, routeID=route_id, typeID=veh_type)

    # # Simulation Step
    # while step < 1000:
    #     traci.simulationStep()
    #     # Get State
    #     agent.state = agent.get_state()
    #     # Get Action
    #     action_mu, action_var = agent.actor(tf.convert_to_tensor(np.expand_dims(agent.state, axis=0)))
    #     action_dist = tfp.distributions.Normal(loc=action_mu, scale=tf.math.sqrt(action_var))
    #     action_sample = action_dist.sample()  # Tensor format
    #     action_policy = action_dist.prob(action_sample)  # Tensor format
    #     # Perform Action
    #     agent.action_perform(longitudinal_action_tensor=action_sample)
    #     # Print Action
    #     print('Action: ', action_sample.numpy()[0], 'Policy: ', action_policy.numpy()[0])
    #     # Step increment
    #     step += 1

    # # Stop the simulation
    # traci.close()
# ********************************************************************** Main Program End ********************************************************************************
# Training Script for SUMO
# Environment: Single-Ring
# Traning method: On-policy trajectories collection (Same as reference paper and original PG.)

# *********************** SUMO related headers ***********************
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary

# *********************** Other headers ***********************
import tensorflow as tf  
import numpy as np 
from time import time

# *********************** Agent scripts ***********************
from agents.PPO_agent_3 import Autonomous_Vehicle_Agent


# **************************** Local Methods *********************************
def runinSteps(agent, step, n_veh):
    _NONE_CONTROLLED_VEH = "human"
    _FIXED_ACCEL = 0.5
    # Generate Non-Controlled Vehicles in the simulation
    for i in range(n_veh-1):
        traci.vehicle.add(vehID=i, routeID=route_id, typeID=_NONE_CONTROLLED_VEH, departLane="allowed")  # Add vehicle into the simulation in another lane
        traci.vehicle.setSpeed(vehID=i, speed=0.5)  # Slow every vehicle in the run-in steps
        traci.vehicle.setLaneChangeMode(i, 0b000000000000)  # Prohibit lane change to the IDM vehicles
    traci.simulationStep()  # Take 1st simulation step to prevent below condition to be ignored.

    # Limit the speed of agent to wait for the other vehicles to depart.
    while len(traci.simulation.getPendingVehicles()) != 0:
        # >>>>>>>>>> In run-in state, AV speed is limited to slow (In order to wait for all the vehicle to depart)
        run_in_action = tf.constant([_FIXED_ACCEL], dtype=tf.float32)  # Fixed Acceleration = self.max_speed/10
        _, _, _ = agent.actionPerform(state=agent.state, action=run_in_action)  # Perform action
        # >>>>>>>>>> Step Increment
        traci.simulationStep()  # Simulation Step
        step += 1
    
    # `````````` After all vehicle have been departed, set the speed controll back to SUMO configuration.
    for i in range(n_veh-1):
        traci.vehicle.setSpeed(vehID=i, speed=-1)
    traci.vehicle.setSpeed(vehID=agent.vehicle_id, speed=-1)

    # `````````` Run-in steps end
    return step  # Return total run-in steps used

def getNumVehicles(trajectory, iter_per_config):
    if trajectory == (iter_per_config-1):
        n_veh = 22
    elif trajectory > (iter_per_config-1) and trajectory <= (2*iter_per_config-1):
        n_veh = 20
    elif trajectory > (2*iter_per_config-1) and trajectory <= (3*iter_per_config-1):
        n_veh = 24
    elif trajectory > (3*iter_per_config-1) and trajectory <= (4*iter_per_config-1):
        n_veh = 21
    elif trajectory > (4*iter_per_config-1) and trajectory <= (5*iter_per_config-1):
        n_veh = 23
    else:
        n_veh = np.random.randint(20, 25)
    return n_veh

def sumoInteraction(agent, evaluation=False) -> float:
    # `````````` Start the program again
    traci.start(sumoCmd)

    # `````````` Initialise/Reset observation parameters
    step = 0  # Initialise step counter
    agent.step = 0  # Reset agent's step
    trajectory_reward = 0
    agent.reward_append = []  # Reset reward append (used in reward function --> for reduce memory load)
    agent.accum_reward = 0  # Reset accumulative reward (used in reward function)
    agent.accum_reward_append = []  # Reset accumulative reward append (used in reward function --> for reduce memory load)

    # `````````` Random the number of vehicle generated in the environment.
    # random from the density from 85veh/km(42veh) to 92 veh/km(46veh)
    # Equally-spaced density configuration
    if evaluation:
        n_veh = 22  # Fixed the number of vehicles to 22 for the evaluation phase.
    else:
        n_veh = getNumVehicles(agent.trajectory, ITER_PER_CONFIG)

    # `````````` Generate Autonomous Vehicle in the simulation (Agent)
    traci.vehicle.add(vehID=agent.vehicle_id, routeID=route_id, typeID=CONTROLLED_VEH)

    # `````````` Run-in steps
    # Run-in steps until all vehicles have been deployed.
    run_in_time = runinSteps(agent, step, n_veh)
    
    # `````````` Warmup Steps
    # Warm up the traffic to get a steady state before learning (Follow the paper.)
    while step < (run_in_time + WARMUP_TIME):
        traci.simulationStep()
        step += 1

    # `````````` Get initial state
    agent.state = agent.getState()  

    # `````````` Observation Loop Start 
    while agent.step < TRAJECTORY_BUDGET:
        # Get Action and Policy
        agent.action, _, agent.policy = agent.actionPerform(state=agent.state)

        # Execute Simulation Step (Perform Action)
        traci.simulationStep()

        # Get Transitions (Reward, Next State)
        agent.reward, agent.next_state = agent.rewardFunction(noise=False, n_veh=n_veh)
        trajectory_reward += agent.reward  # Accumulate Reward

        # Store Observation
        agent.observationStoring(index=agent.step, 
                                 state=agent.state,
                                 action=agent.action,
                                 policy=agent.policy,
                                 reward=agent.reward,
                                 next_state=agent.next_state)
        # Step Increment
        step += 1
        agent.step += 1
        # Assign New state
        agent.state = agent.next_state
    # `````````` Observation Loop End

    # `````````` Stop the simulation
    traci.close()
    return trajectory_reward

# ************************** Main Program Start ******************************
# ========== Initial command for opening the sumo interface.
sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-c", "C:/Amornyos/PhD/Traffic_Flow_SUMO_shared/networks/single_ring/single_ring.sumocfg.xml", 
        '--no-warnings', 
        '--no-step-log']
SUMO_TIME_STEP = 0.1  # SUMO time step setting is 0.1 s (Follow Alex Bayen's paper)

# ========== Start the program
traci.start(sumoCmd)

# ========== Pre-define simulation environment
route_id = np.squeeze(traci.route.getIDList())  # Get all route's ID
veh_type = np.squeeze(traci.vehicletype.getIDList()[0])  # Get the vehicle type
lane_id = traci.lane.getIDList()  # Get all lane ids in the environment.
# Note: minGap has been changed to 2.0 (Not default value 2.5)
# Note: sigma of "rl_agent" has been set to 0 (Not default value 0.2) <-- which means the AV's driving is perfect.
CONTROLLED_VEH = "rl_agent"

# ========== Initialise simulation parameters
GRADIENTS_STEP = 200  # Define total required gradient updating step
EPOCH = 5
EVALUATION_TIMES = 1
BATCH_TRAJECTORY = 40  # 40 <= B <= 45 in original paper.
TRAJECTORY_BUDGET = int(1000/SUMO_TIME_STEP)  # Define trajectories budget for the agent (batch size of the observation) (horizon)
WARMUP_TIME = 100/SUMO_TIME_STEP # Get warmup time
ONE_HOUR = 3600/SUMO_TIME_STEP  # Define one hour time. (s)
# Single-ring configuration 
NUM_CONFIG = 5  # Density from 42veh to 46 veh [42, 43, 44, 45 ,46] (5 configurations)
ITER_PER_CONFIG = int(BATCH_TRAJECTORY / NUM_CONFIG)  # Define iteration per configuration (How many trajectory to collect from the particular configuration.)

# ========== Initialise agent
agent = Autonomous_Vehicle_Agent(load=True, 
                                 time_step=SUMO_TIME_STEP, 
                                 observation_budget=BATCH_TRAJECTORY,
                                 horizon=TRAJECTORY_BUDGET,
                                 collector=False,
                                 action_config='continuous',
                                 reward_config='original',
                                 reward_objective='greedy',
                                 policy_objective='sum_reward')

# ========== Close the program
traci.close()

# ========== Training Loop Start

while agent.training_record < GRADIENTS_STEP:
    # ----------  Initialise/Reset
    start_time = time()  # Get start training time
    agent.trajectory = 0
    agent.episode_reward = 0  # Reset episode reward
    agent.avg_reward = 0  # Reset average episode reward
    nan_flag = False
    accum_eva_score = 0
    
    # ---------- Trajectories Collection Loop Start
    while agent.trajectory < BATCH_TRAJECTORY:

        # `````````` Interact with SUMO
        trajectory_reward = sumoInteraction(agent)

        # `````````` Accumulate Episode Reward
        agent.episode_reward += trajectory_reward

        # `````````` Compute ExpectedSumReturn (Value_Function) and Advantage (a_hat)
        training_batch_dict = agent.prepareTrainingBatch(size=agent.step)
        # `````````` Add new trajectory data to a current training buffers
        agent.trainingBuffersAppends(agent.trajectory, training_batch_dict)

        # `````````` Observation Loop End 
        agent.trajectory += 1  # Update the trajectory counter.
        print('Episode: ', agent.training_record+1, 
                ', Trajectory: ', agent.trajectory, '/', BATCH_TRAJECTORY, 
                ', Trajectory_Reward: ', trajectory_reward)

    # ---------- Average the sum episode_rewards
    agent.episode_reward = agent.episode_reward / BATCH_TRAJECTORY

    
    # ---------- Train the agent
    # Train the agent with the gradient steps setting.
    for b in range(BATCH_TRAJECTORY):
        for c in range(int(agent.observation_size/agent.minibatch_size)):
            minibatch = agent.minibatchSampling(key=b, pos=c)
            for _ in range(EPOCH):
                agent.training(minibatch=minibatch)
    end_time = time()  # Get end training time
    
    # ---------- Record Episode Reward
    agent.sum_reward_append = np.append(agent.sum_reward_append, agent.episode_reward)  # Append the episode reward.
    agent.avg_reward = np.mean(agent.sum_reward_append)  # Compute the average episode reward.

    # ---------- Print the training status
    agent.statusPrinting(episode=GRADIENTS_STEP, 
                         trajectory=BATCH_TRAJECTORY, 
                         compute_time=(end_time - start_time))

    # ---------- After all 'B' trajectories have been collected and trained,
    # Increase the training counter and save the model
    agent.training_record += 1  # Update the training_record.
    for _ in range(EVALUATION_TIMES):
        # Evaluate the trained model before saving
        accum_eva_score += sumoInteraction(agent, evaluation=True)
    agent.evaluation_score = accum_eva_score/EVALUATION_TIMES

    improvement = True if agent.evaluation_score >= agent.prev_evaluation_score else False
    if improvement:
        agent.trainingSaving(training_save=True, model_save=True)  # save the mo
        agent.prev_evaluation_score = agent.evaluation_score
    print('Evaluation_Score: ', agent.evaluation_score, ', (Better -> saved)' if improvement else '(Worse -> Not saved)')

# ========== Training Loop End

# ========== Print Status
print('Training_Finished !!')

# ========== Save the models
agent.trainingSaving(training_save=True, model_save=True)

    # ********************************************************************** Main Program End ********************************************************************************
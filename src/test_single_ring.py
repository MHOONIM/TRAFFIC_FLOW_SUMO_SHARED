# Test Script for SUMO
# Environment: Single-Ring

# *********************** SUMO related headers ***********************
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
from sumolib import checkBinary

# *********************** Other headers ***********************
import tensorflow as tf  # Tensorflow
import numpy as np   # Numpy
from time import time

# *********************** Agent scripts ***********************
from agents.PPO_agent_3 import Autonomous_Vehicle_Agent

# **************************** Local Methods *********************************
def runinSteps(step, agent):
    _NONE_CONTROLLED_VEH = "human"
    # Generate Non-Controlled Vehicles in the simulation
    for i in range(n_veh-1):
        traci.vehicle.add(vehID=i, routeID=route_id, typeID=_NONE_CONTROLLED_VEH, departLane="allowed")  # Add vehicle into the simulation in another lane
        traci.vehicle.setSpeed(vehID=i, speed=0.5)  # Slow every vehicle in the run-in steps
        traci.vehicle.setLaneChangeMode(i, 0b000000000000)  # Prohibit lane change to the IDM vehicles
    traci.simulationStep()  # Take 1st simulation step to prevent below condition to be ignored.

    # Limit the speed of agent to wait for the other vehicles to depart.
    while len(traci.simulation.getPendingVehicles()) != 0:
        # >>>>>>>>>> In run-in state, AV speed is limited to slow (In order to wait for all the vehicle to depart)
        run_in_action = tf.constant([agent.accel], dtype=tf.float32)  # Fixed Acceleration = self.max_speed/10
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

# ************************** Main Program Start ******************************
if __name__ == "__main__":
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
    TRAINING_EPISODE = 200  # Define total required gradient updating step
    BATCH_TRAJECTORY = 1  # 40 <= B <= 45 in original paper.
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

    # `````````` Initialise/Re-initialise observation parameters
    step = 0  # Initialise step counter
    agent.step = 0  # Reset agent's step
    nan_flag = False
    trajectory_reward = 0
    agent.reward_append = []  # Reset reward append (used in reward function --> for reduce memory load)
    agent.accum_reward = 0  # Reset accumulative reward (used in reward function)
    agent.accum_reward_append = []  # Reset accumulative reward append (used in reward function --> for reduce memory load)
    noise_inj = False  # Noise injection parameters, Set 'True' if desire some noise in the observation.
    noise_imp = False  # Noise imposter parameters, Set 'True' if desire some IDM car to become noisy.

    # `````````` Random the number of vehicle generated in the environment.
    # random from the density from 85veh/km(42veh) to 92 veh/km(46veh)
    # Equally-spaced density configuration
    n_veh = getNumVehicles(agent.trajectory, ITER_PER_CONFIG)

    # `````````` Generate Autonomous Vehicle in the simulation (Agent)
    traci.vehicle.add(vehID=agent.vehicle_id, routeID=route_id, typeID=CONTROLLED_VEH)

    # `````````` Run-in steps start
    # Run-in steps until all vehicles have been deployed.
    run_in_time = runinSteps(step, agent)
    # `````````` Run-in steps end
    
    # `````````` Warmup Steps
    # Warm up the traffic to get a steady state before learning (Follow the paper.)
    while step < (run_in_time + WARMUP_TIME):
        traci.simulationStep()
        step += 1

    # `````````` Get initial state
    agent.state = agent.getState()  
    s_time = time()
    # `````````` Observation Loop Start 
    while agent.step < TRAJECTORY_BUDGET:
        # >>>>>>>>> Get Action and Policy
        agent.action, _, agent.policy = agent.actionPerform(state=agent.state)
        # >>>>>>>>> Execute Simulation Step (Perform Action)
        traci.simulationStep()
        # >>>>>>>>> Get Reward
        agent.reward, agent.next_state = agent.rewardFunction(noise=False, n_veh=n_veh)
        # >>>>>>>>> Accumulate Episode Reward
        trajectory_reward += agent.reward

        # >>>>>>>>> Assign New state
        agent.state = agent.next_state
        # >>>>>>>>> Step Increment
        step += 1
        agent.step += 1
    e_time = time()
    print(e_time - s_time)
    # `````````` Accumulate Episode Reward
    agent.episode_reward += trajectory_reward

    # `````````` Stop the simulation
    traci.close()

    # `````````` Print the training status
    agent.statusPrinting(episode=TRAINING_EPISODE, trajectory=BATCH_TRAJECTORY, compute_time=0)

    # ========== Test Loop End

    # ========== Print Status
    print('Test_Finished !!')

    # ========== Save the models
    agent.testSaving()

    # ********************************************************************** Main Program End ********************************************************************************
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
import tensorflow as tf  # Tensorflow
import numpy as np   # Numpy

# *********************** Agent scripts ***********************
from PPO_agent_2 import Autonomous_Vehicle_Agent
# from PPO_agent_double_ring_3 import Autonomous_Vehicle_Agent  # PPO (Continuous, No Critic)
# from PPO_agent_4 import Autonomous_Vehicle_Agent  # PPO (Discrete, No Critic)
# from PPO_agent_double_ring_5 import Autonomous_Vehicle_Agent  # PPO (My Mapped Continuous, No Critic)
# from TRPO_agent_1 import Autonomous_Vehicle_Agent
# from TRPO_agent_2 import Autonomous_Vehicle_Agent
# from TRPO_agent_3 import Autonomous_Vehicle_Agent
# from DQL_agent_3 import Autonomous_Vehicle_Agent

# ********************************************************************* Main Program Start *******************************************************************************
if __name__ == "__main__":
    # ========== Initial command for opening the sumo interface.
    sumoBinary = checkBinary("sumo")
    sumoCmd = [sumoBinary, "-c", "C:/Amornyos/PhD/Traffic_Flow_SUMO_shared/networks/single_ring/single_ring.sumocfg.xml", 
            '--no-warnings', 
            '--no-step-log']
    sumo_time_step = 0.1  # SUMO time step setting is 0.1 s (Follow Alex Bayen's paper)
    
    # ========== Start the program
    traci.start(sumoCmd)

    # ========== Pre-define simulation environment
    route_id = np.squeeze(traci.route.getIDList())  # Get all route's ID
    veh_type = np.squeeze(traci.vehicletype.getIDList()[0])  # Get the vehicle type
    lane_id = traci.lane.getIDList()  # Get all lane ids in the environment.
    # Note: minGap has been changed to 2.0 (Not default value 2.5)
    # Note: sigma of "rl_agent" has been set to 0 (Not default value 0.2) <-- which means the AV's driving is perfect.
    non_controlled_veh = "human"
    controlled_veh = "rl_agent"

    # ========== Initialise simulation parameters
    training_episode = 200  # Define total required gradient updating step
    batch_trajectories = 5  # 40 <= B <= 45 in original paper.
    trajectory_budget = int(100/sumo_time_step)  # Define trajectories budget for the agent (batch size of the observation) (horizon)
    warmup_time = 100/sumo_time_step # Get warmup time
    one_hour = 3600/sumo_time_step  # Define one hour time. (s)
    # Single-ring configuration 
    num_config = 5  # Density from 42veh to 46 veh [42, 43, 44, 45 ,46] (5 configurations)
    iter_per_config = batch_trajectories / num_config  # Define iteration per configuration (How many trajectory to collect from the particular configuration.)

    # ========== Initialise agent
    agent = Autonomous_Vehicle_Agent(load=False, 
                                     time_step=sumo_time_step, 
                                     observation_budget=batch_trajectories,
                                     horizon=trajectory_budget,
                                     collector=False,
                                     action_config='continuous',
                                     reward_config='original',
                                     reward_objective='greedy',
                                     policy_objective='sum_reward')
    
    # ========== Close the program
    traci.close()

    # ========== Training Loop Start
    while agent.training_record < training_episode:
        # ---------- Load the new policy of actor to the collector (on-policy) (When this is not the first time learning.)
        if agent.training_record > 0 and agent.collector_config == True:
            agent.collector.set_weights(agent.actor.get_weights())
        
        # ---------- Reset the trajectory counter
        agent.trajectory = 0

        # ---------- Trajectories Collection Loop Start
        while agent.trajectory < batch_trajectories:

            # `````````` Start the program again
            traci.start(sumoCmd)

            # `````````` Initialise/Re-initialise observation parameters
            step = 0  # Initialise step counter
            agent.step = 0  # Reset agent's step
            agent.episode_reward = 0  # Reset episode reward
            agent.avg_reward = 0  # Reset average episode reward
            agent.reward_append = []  # Reset reward append (used in reward function --> for reduce memory load)
            agent.accum_reward = 0  # Reset accumulative reward (used in reward function)
            agent.accum_reward_append = []  # Reset accumulative reward append (used in reward function --> for reduce memory load)
            noise_inj = False  # Noise injection parameters, Set 'True' if desire some noise in the observation.
            noise_imp = False  # Noise imposter parameters, Set 'True' if desire some IDM car to become noisy.

            # `````````` Random the number of vehicle generated in the environment.
            # random from the density from 85veh/km(42veh) to 92 veh/km(46veh)
            # Equally-spaced density configuration
            if agent.trajectory <= (iter_per_config-1):
                n_veh = 22
            elif agent.trajectory > (iter_per_config-1) and agent.trajectory <= (2*iter_per_config-1):
                n_veh = 20
            elif agent.trajectory > (2*iter_per_config-1) and agent.trajectory <= (3*iter_per_config-1):
                n_veh = 24
            elif agent.trajectory > (3*iter_per_config-1) and agent.trajectory <= (4*iter_per_config-1):
                n_veh = 21
            elif agent.trajectory > (4*iter_per_config-1) and agent.trajectory <= (5*iter_per_config-1):
                n_veh = 23

            # `````````` Generate Autonomous Vehicle in the simulation (Agent)
            traci.vehicle.add(vehID=agent.vehicle_id, routeID=route_id, typeID=controlled_veh)
            # traci.vehicle.setLaneChangeMode(agent.vehicle_id, 0b000000000000)

            # `````````` Run-in steps start
            # Run-in steps until all vehicles have been deployed.
            # Generate Non-Controlled Vehicles in the simulation
            for i in range(n_veh-1):
                traci.vehicle.add(vehID=i, routeID=route_id, typeID=non_controlled_veh, departLane="allowed")  # Add vehicle into the simulation in another lane
                traci.vehicle.setSpeed(vehID=i, speed=0.5)  # Slow every vehicle in the run-in steps
                traci.vehicle.setLaneChangeMode(i, 0b000000000000)  # Prohibit lane change to the IDM vehicles
            traci.simulationStep()  # Take 1st simulation step to prevent below condition to be ignored.
            while len(traci.simulation.getPendingVehicles()) != 0:
                # >>>>>>>>>> In run-in state, AV speed is limited to slow (In order to wait for all the vehicle to depart)
                run_in_action = tf.constant([agent.accel], dtype=tf.float32)  # Fixed Acceleration = self.max_speed/10
                _, _, _ = agent.action_perform(state=agent.state, action=run_in_action)  # Perform action
                # >>>>>>>>>> Step Increment
                traci.simulationStep()  # Simulation Step
                step += 1
            # `````````` After all vehicle have been departed, set the speed controll back to SUMO configuration.
            for i in range(n_veh-1):
                traci.vehicle.setSpeed(vehID=i, speed=-1)
            traci.vehicle.setSpeed(vehID=agent.vehicle_id, speed=-1)
            # `````````` Run-in steps end
            run_in_time = step  # Get the run_in times

            # # `````````` Define the simulation time, run_in + warmup + training_time + testing_time(one_hour)
            # simulation_time = run_in_time + warmup_time + (trajectories_budget*training_episode) + one_hour

            # # `````````` Initialise imposter vehicle parameters (If noise_inj == True)
            # impos_lane = lane_id[int(agent.state[1])]  # Gather current lane of AV. (0=outer, 1=inner)
            # if traci.lane.getLastStepVehicleIDs(laneID=impos_lane) == ():  # Check if there is any vehicle on the current lane of AV
            #     # ++++++ If there is no vehicle on the current lane of AV, there must be vehicle on the opposite site (left-right).
            #     if agent.state[1] == 0:  # If AV is currently on the outer lane.
            #         impos_lane = lane_id[2]  # check imposter lane on 
            #     elif agent.state[1] == 1:
            #         impos_lane = lane_id[3]
            # # `````````` Retrieve the imposter vehicle id.
            # impos_veh_id = random.choice(traci.lane.getLastStepVehicleIDs(laneID=impos_lane))
            # impos_veh_speed = -1  # Initialise imposter's speed with SUMO controlled first.

            # `````````` Warmup Steps
            # Warm up the traffic to get a steady state before learning (Follow the paper.)
            while step < (run_in_time + warmup_time):
                traci.simulationStep()
                step += 1

            # `````````` Get initial state
            agent.state = agent.get_state()  

            # `````````` Observation Loop Start 
            while agent.step < trajectory_budget:
                # >>>>>>>>> Get Action and Policy
                agent.action, _, agent.action_policy = agent.action_perform(state=agent.state)
                # >>>>>>>>> Execute Simulation Step (Perform Action)
                traci.simulationStep()
                # >>>>>>>>> Get Reward
                agent.reward, agent.next_state = agent.reward_function(noise=False, n_veh=n_veh)
                # >>>>>>>>> Accumulate Episode Reward
                agent.episode_reward += agent.reward
                # >>>>>>>>> Store Observation
                agent.observation_storing(index=agent.step, 
                                          state=agent.state,
                                          action=agent.action,
                                          policy=agent.action_policy,
                                          reward=agent.reward,
                                          termination=agent.termination,
                                          next_state=agent.next_state)
                # >>>>>>>>> Assign New state
                agent.state = agent.next_state
                # >>>>>>>>> Step Increment
                step += 1
                agent.step += 1

                # # >>>>>>>>> Noise imposter process (Omitted if noise_imp == False) ++++++
                # # If noise_inj == True, Randomly reduce the speed or stop one of the vehicle 
                # if step % (250/sumo_time_step) == 0 and noise_imp and agent.training_record > training_episode//10: 
                #     impos_lane = lane_id[int(agent.state[1])]  # Gather current lane of AV. (0=outer, 1=inner)
                #     if traci.lane.getLastStepVehicleIDs(laneID=impos_lane) == ():  # Check if there is any vehicle on the current lane of AV
                #         # If there is no vehicle on the current lane of AV, there must be vehicle on the opposite site (left-right).
                #         if agent.state[1] == 0:  # If AV is currently on the outer lane.
                #             impos_lane = lane_id[2]  # check imposter lane on 
                #         elif agent.state[1] == 1:
                #             impos_lane = lane_id[3]
                #     # traci.lane.getLastStepVehicleIDs(laneID=agent.state[1])
                #     prev_impos_veh_id = impos_veh_id
                #     traci.vehicle.setSpeed(vehID=prev_impos_veh_id, speed=-1)  # Reset current imposter speed (back to SUMO)
                #     impos_veh_id = random.choice(traci.lane.getLastStepVehicleIDs(laneID=impos_lane))  # Generate new imposter vehicle id.
                #     impos_veh_speed = 0.5  # Slow the imposter vehicle down.
                #     traci.vehicle.setSpeed(vehID=impos_veh_id, speed=impos_veh_speed)  
                # # >>>>>>>>> Set imposter speed 
                # traci.vehicle.setSpeed(vehID=impos_veh_id, speed=impos_veh_speed)  

            # `````````` Observation Loop End 

            # `````````` Train the agent
            # Train the agent with the gradient steps setting.
            # agent.training(gradient_steps=int(agent.observation_size/agent.batch_observation_size))
            agent.training(gradient_steps=1)

            # `````````` Record Episode Reward
            agent.sum_reward_append = np.append(agent.sum_reward_append, agent.episode_reward)  # Append the episode reward.
            agent.avg_reward = np.mean(agent.sum_reward_append)  # Compute the average episode reward.

            # `````````` Print the training status
            agent.status_printing(episode=training_episode, trajectory=batch_trajectories)
            agent.trajectory += 1  # Update the trajectory counter.
                
            # `````````` Stop the simulation
            traci.close()
    
        # ---------- After all 'B' trajectories have been collected and trained,
        # Increase the training counter and save the model
        agent.training_record += 1  # Update the training_record.
        agent.training_saving(training_save=True, model_save=True)  # save the models.

    # ========== Training Loop End

    # ========== Print Status
    print('Training_Finished !!')

    # ========== Save the models
    agent.training_saving(training_save=True, model_save=True)

    # ********************************************************************** Main Program End ********************************************************************************
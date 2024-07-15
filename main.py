import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from pid_controller import PIDController
from utils import Params
from particle_swarm_optimization import ParticleSwarmOptimization
from constants import SAMPLE_TIME, NUM_EPISODES
from math import inf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

RENDER = False  # If the Mountain Car environment should be rendered
fig_format = 'png'  # Format used for saving matplotlib's figures
# fig_format = 'eps'
# fig_format = 'svg'
# Defining PSO hyperparameters
hyperparams = Params()
hyperparams.num_particles = 60
hyperparams.inertia_weight = 0.2
hyperparams.cognitive_parameter = 0.6
hyperparams.social_parameter = 0.8
lower_bound = np.array([30, 0.0, 0.0])
upper_bound = np.array([100, 10, 10.0])
# lower_bound = np.array([10.0, 0.0, 0.0])
# upper_bound = np.array([200.0, 1300.0, 30.0])
pso = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
pso_theta = ParticleSwarmOptimization(hyperparams, lower_bound, upper_bound)
def plot_results():
    """
    Plots the results of the optimization.
    """
    fig_format = 'png'
    plt.figure()
    plt.plot(position_x_history)
    plt.legend(['Kp', 'Ki', 'Kd'])
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameters Convergence For Horizontal Control')
    plt.grid()
    plt.savefig('parameters_convergence_x.%s' % fig_format, format=fig_format)

    plt.figure()
    plt.plot(position_theta_history)
    plt.legend(['Kp', 'Ki', 'Kd'])
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Parameters Convergence for Theta Control')
    plt.grid()
    plt.savefig('parameters_convergence_theta.%s' % fig_format, format=fig_format)
    
    
    plt.figure()
    plt.plot(quality_history)
    plt.xlabel('Iteration')
    plt.ylabel('Simulation Time')
    plt.title('Simulation Time Convergence')
    plt.grid()
    plt.savefig('line_quality_convergence.%s' % fig_format, format=fig_format)
    best_history = []
    best = -inf
    for q in quality_history:
        if q > best:
            best = q
        best_history.append(best)
    plt.figure()
    plt.plot(best_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Time')
    plt.title('Best Time Convergence')
    plt.grid() 
    plt.savefig('line_best_convergence.%s' % fig_format, format=fig_format)
    plt.show()

def convert_particle_position_to_params(position):
	"""
	Converts a particle position into controller params.
	:param position: particle position.
	:type position: numpy array.
	:return: controller params.
	"""
	params = Params()
	params.kp = position[0]
	params.ki = position[1]
	params.kd = position[2]
	return params


# Initiating the Mountain Car environment
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initializing history
position_x_history = []  # history of evaluated particle positions
position_theta_history = []
quality_history = []  # history of evaluated qualities

control_theta = PIDController(0,0, 0, SAMPLE_TIME)
control_posic = PIDController(0, 0, 0, SAMPLE_TIME)


for episodes in range(1, NUM_EPISODES + 1):
	# Reset the environment
	position = pso.get_position_to_evaluate()
	controller_params = convert_particle_position_to_params(position)
	control_posic.set_gains(controller_params.kp, controller_params.ki, controller_params.kd)

	position_theta = pso_theta.get_position_to_evaluate()
	controller_params_theta = convert_particle_position_to_params(position_theta)
	control_theta.set_gains(controller_params_theta.kp, controller_params_theta.ki, controller_params_theta.kd)


	state = env.reset()
	# This reshape is needed to keep compatibility with Keras
	# Cumulative reward is the return since the beginning of the episode
	cumulative_reward = 0
	for time in range(1, 500):
		if RENDER:
			env.render()  # Render the environment for visualization
		# Select action
		action = PIDController.decide_action(state,control_posic,control_theta)

		# Take action, observe reward and new state
		next_state, reward, done, _ = env.step(action)
		# Accumulate reward
		#cumulative_reward = agent.gamma * cumulative_reward + reward
		cumulative_reward += reward
		state = next_state
		if done:
			print("episode: {}/{}, time(score): {:.6}"
				  .format(episodes, NUM_EPISODES, cumulative_reward))
			break
		# We only update the policy if we already have enough experience in memory
	quality_history.append(cumulative_reward)

	pso_theta.notify_evaluation(cumulative_reward)
	pso.notify_evaluation(cumulative_reward)
	if episodes % hyperparams.num_particles == 0:
		pso.advance_generation()
		pso_theta.advance_generation()
		best_posic = pso.get_best_position()
		best_theta = pso_theta.get_best_position()
		position_x_history.append(np.array(best_posic))
		position_theta_history.append(np.array(best_theta))
		# print("posic kp: {} ki: {} kd: {} ".format(best_posic.kp,best_posic.kd, best_posic.ki))
		# print("theta kp: {} ki: {} kd: {} ".format(best_theta.kp, best_theta.kd, best_theta.ki))
	# Every 10 episodes, update the plot for training monitoring
	# if episodes % 50 == 0:
		# plt.plot(return_history, 'b')
		# plt.xlabel('Episode')
		# plt.ylabel('Return')
		# plt.show(block=False)
		# plt.pause(0.1)
		# plt.savefig('dqn_training.' + fig_format, format=fig_format)
		# Saving the model to disk
		# agent.save("cart_pole.h5")
plot_results()
plt.pause(1.0)

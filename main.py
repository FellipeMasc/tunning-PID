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
hyperparams.num_particles = 40
hyperparams.inertia_weight = 0.4
hyperparams.cognitive_parameter = 0.6
hyperparams.social_parameter = 0.8
lower_bound_x = np.array([ 0, 100.0])
upper_bound_x = np.array([ 3, 150.0])
lower_bound_theta = np.array([ 0, 100.0])
upper_bound_theta = np.array([ 5, 150.0])
# lower_bound = np.array([10.0, 0.0, 0.0])
# upper_bound = np.array([200.0, 1300.0, 30.0])
pso = ParticleSwarmOptimization(hyperparams, lower_bound_x, upper_bound_x)
pso_theta = ParticleSwarmOptimization(hyperparams, lower_bound_theta,upper_bound_theta)
def plot_results():
	"""
	Plots the results of the optimization.
	"""
	fig_format = 'png'
	plt.figure()
	plt.plot(position_x_history)
	plt.legend(['Kp', 'Kd'])
	plt.xlabel('Iteration')
	plt.ylabel('Parameter Value')
	plt.title('Parameters Convergence For Horizontal Control')
	plt.grid()
	plt.savefig('parameters_convergence_x.%s' % fig_format, format=fig_format)

	plt.figure()
	plt.plot(position_theta_history)
	plt.legend(['Kp', 'Kd'])
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
	params.kd = position[1]
	# params.ki = position[1]
	return params


# Initiating the Mountain Car environment
env = gym.make('CartPole-v1')
# Initializing history
position_x_history = []  # history of evaluated particle positions
position_theta_history = []
quality_history = []  # history of evaluated qualities

control_theta = PIDController(5,0, 100, 1)
control_posic = PIDController(1, 0, 100, 1)


for episodes in range(1, NUM_EPISODES + 1):

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
	pso.advance_generation()
	pso_theta.advance_generation()
	best_posic = pso.get_best_position()
	best_theta = pso_theta.get_best_position()
	position_x_history.append(np.array(best_posic))
	position_theta_history.append(np.array(best_theta))
  
	position = pso.get_position_to_evaluate()
	controller_params = convert_particle_position_to_params(position)
	control_posic.set_gains(controller_params.kp, 0, controller_params.kd)

	position_theta = pso_theta.get_position_to_evaluate()
	controller_params_theta = convert_particle_position_to_params(position_theta)
	control_theta.set_gains(controller_params_theta.kp, 0, controller_params_theta.kd)
plot_results()
plt.pause(1.0)

#visualize 
# class PD:
#     def __init__(self, kp, kd, goal):
#         self.kp = kp
#         self.kd = kd
#         self.goal = goal
#         self.last_error = 0

#     def observe(self, x):
#         error = self.goal - x
#         d_error = error - self.last_error
#         self.last_error = error
#         return self.kp * error + self.kd * d_error
# controller = PD(kp=5, kd=100, goal=0)

env = gym.make("CartPole-v1")
observation = env.reset()

for _ in range(1000):
	# control_output = controller.observe(pole_angle)
	env.render()
	action = PIDController.decide_action(observation,control_posic,control_theta)
	
	observation, reward, terminated, truncated = env.step(action)
	if terminated or truncated:
		observation = env.reset()

env.close()

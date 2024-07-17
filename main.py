import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from pid_controller import PIDController
from utils import Params
from particle_swarm_optimization import ParticleSwarmOptimization
from constants import SAMPLE_TIME, NUM_EPISODES, INITIAL_INERTIA_WEIGHT,INITIAL_SOCIAL_PARAMETER, FINAL_INERTIA_WEIGHT,FINAL_SOCIAL_PARAMETER, INITIAL_COGNITIVE_PARAMETER, FINAL_COGNITIVE_PARAMETER
from math import inf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

fig_format = 'png'
# Defining PSO hyperparameters
hyperparams = Params()
hyperparams.num_particles = 40
hyperparams.inertia_weight = 0.4
hyperparams.cognitive_parameter = 0.4
hyperparams.social_parameter = 0.6
lower_bound_x = np.array([ 0.8, 0.1, 100.0])
upper_bound_x = np.array([ 2, 0.3, 120.0])
lower_bound_theta = np.array([ 4, 0.3, 100.0])
upper_bound_theta = np.array([ 6, 0.5, 110.0])
pso = ParticleSwarmOptimization(hyperparams, lower_bound_x, upper_bound_x)
pso_theta = ParticleSwarmOptimization(hyperparams, lower_bound_theta,upper_bound_theta)

def adjust_inertia_weight(iteration, max_iterations):
	"""
    Adjusts the inertia weight of the PSO algorithm over iterations.

    Parameters:
        iteration (int): Current iteration number.
        max_iterations (int): Total number of iterations.

    Returns:
        float: The adjusted inertia weight.
    """
	return INITIAL_INERTIA_WEIGHT - (INITIAL_INERTIA_WEIGHT - FINAL_INERTIA_WEIGHT) * (0.1*iteration / max_iterations)

def adjust_cognitive_social_parameters(iteration, max_iterations):
	"""
    Adjusts the cognitive and social parameters of the PSO algorithm over iterations.

    Parameters:
        iteration (int): Current iteration number.
        max_iterations (int): Total number of iterations.

    Returns:
        tuple: Tuple containing adjusted cognitive and social parameters.
    """
	cognitive = INITIAL_COGNITIVE_PARAMETER - (INITIAL_COGNITIVE_PARAMETER - FINAL_COGNITIVE_PARAMETER) * (iteration / max_iterations)
	social = INITIAL_SOCIAL_PARAMETER - (INITIAL_SOCIAL_PARAMETER - FINAL_SOCIAL_PARAMETER) * (iteration / max_iterations)
	return cognitive,social

def plot_results():
	"""
	Plots the results of the optimization.
	"""
	fig_format = 'png'
	plt.figure()
	plt.plot(position_x_history)
	plt.legend(['Kp','Ki', 'Kd'])
	plt.xlabel('Iteration')
	plt.ylabel('Parameter Value')
	plt.title('Parameters Convergence For Horizontal Control')
	plt.grid()
	plt.savefig('parameters_convergence_x.%s' % fig_format, format=fig_format)

	plt.figure()
	plt.plot(position_theta_history)
	plt.legend(['Kp','Ki', 'Kd'])
	plt.xlabel('Iteration')
	plt.ylabel('Parameter Value')
	plt.title('Parameters Convergence for Theta Control')
	plt.grid()
	plt.savefig('parameters_convergence_theta.%s' % fig_format, format=fig_format)
	
	
	plt.figure()
	plt.semilogx(quality_history)
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
	plt.semilogx(best_history)
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


# Initiating the CartPole Environment
env = gym.make('CartPole-v1')
# Initializing history
position_x_history = []  # history of evaluated particle positions for horizontal control
position_theta_history = [] # history of evaluated particle positions for theta control
quality_history = []  # history of evaluated qualities

control_theta = PIDController(5,0.3, 100, SAMPLE_TIME)
control_posic = PIDController(1, 0.1, 100, SAMPLE_TIME)

#loop for encounter the best solution using PSO
for episodes in range(1, NUM_EPISODES + 1):
    #adjust hyperparams
	hyperparams.inertial_weight = adjust_inertia_weight(episodes,NUM_EPISODES)
	hyperparams.cognitive_parameter,hyperparams.social_parameter = adjust_cognitive_social_parameters(episodes,NUM_EPISODES)
	position = pso.get_position_to_evaluate()
	controller_params = convert_particle_position_to_params(position)
	pso.reset_hyperparams(hyperparams)
	pso_theta.reset_hyperparams(hyperparams)
	control_posic.set_gains(controller_params.kp, controller_params.ki, controller_params.kd)
	position_theta = pso_theta.get_position_to_evaluate()
	controller_params_theta = convert_particle_position_to_params(position_theta)
	control_theta.set_gains(controller_params_theta.kp, controller_params_theta.ki, controller_params_theta.kd)
	state = env.reset()
	cumulative_reward = 0
	for time in range(1, 500):
		action = PIDController.decide_action(state,control_posic,control_theta)
		next_state, reward, done, truncated = env.step(action)
		cumulative_reward += reward
		state = next_state
		if done or truncated:
			print("episode: {}/{}, time(score): {:.6}"
				  .format(episodes, NUM_EPISODES, cumulative_reward))
			break
	quality_history.append(cumulative_reward)

	pso_theta.notify_evaluation(cumulative_reward)
	pso.notify_evaluation(cumulative_reward)
	pso.advance_generation()
	pso_theta.advance_generation()
	best_posic = pso.get_best_position()
	best_theta = pso_theta.get_best_position()
	position_x_history.append(np.array(best_posic))
	position_theta_history.append(np.array(best_theta))

plot_results()
plt.pause(1.0)

def plot_results_optimized():
	"""
	Plots the results of simulation pos optmization.
	"""
	fig_format = 'png'
	plt.figure()
	plt.plot(posic)
	plt.xlabel('Passo de simulação')
	plt.ylabel('Posição (m)')
	plt.title('Gráfico da posição do carrinho em cada período da simulação')
	plt.grid()
	plt.savefig('posic_step.%s' % fig_format, format=fig_format)
	fig_format = 'png'
	plt.figure()
 
	plt.plot(angle)
	plt.xlabel('Passo de simulação')
	plt.ylabel('Ângulo (rad)')
	plt.title('Gráfico do ângulo do poste em cada período da simulação')
	plt.grid()
	plt.savefig('angle_step.%s' % fig_format, format=fig_format)
	plt.figure()
 
	plt.plot(angle_velocity)
	plt.xlabel('Passo de simulação')
	plt.ylabel('Velocidade angular (rad/s)')
	plt.title('Gráfico da velocidade angular do poste em cada período da simulação')
	plt.grid()
	plt.savefig('angle_velocity_step.%s' % fig_format, format=fig_format)
	plt.figure()
 
	plt.plot(angle)
	plt.xlabel('Passo de simulação')
	plt.ylabel('Velocidade Linear (m/s)')
	plt.title('Gráfico da velocidade linear do carrinho em cada período da simulação')
	plt.grid()
	plt.savefig('posic_velocity_step.%s' % fig_format, format=fig_format)
	plt.show()

#initializing the best controller for real simulation
position = pso.get_best_position()
controller_params = convert_particle_position_to_params(position)
control_posic.set_gains(controller_params.kp, controller_params.ki, controller_params.kd)

position_theta = pso_theta.get_best_position()
controller_params_theta = convert_particle_position_to_params(position_theta)
control_theta.set_gains(controller_params_theta.kp, controller_params_theta.ki, controller_params_theta.kd)
#arrays for graphics
posic = []
angle = []
posic_velocity = []
angle_velocity = []

state = env.reset()

for _ in range(2000):
	posic.append(state[0])
	angle.append(state[2])
	angle_velocity.append(state[3])
	posic_velocity.append(state[1])
	env.render()
	action = PIDController.decide_action(state,control_posic,control_theta)
	
	state, reward, done, truncated = env.step(action)
	if done or truncated:
		state = env.reset()

plot_results_optimized()
env.close()

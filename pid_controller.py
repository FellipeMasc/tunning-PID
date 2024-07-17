from constants import THETA_REF,POSITION_REF


class PIDController:
	"""
	PID Controller for controlling a dynamic system.

	Attributes:
		kp (float): Proportional gain.
		ki (float): Integral gain.
		kd (float): Derivative gain.
		sample_time (float): Sampling time for the controller.
		e1 (float): Previous error for derivative calculation.
		integral (float): Accumulated integral of the error.
	"""
	def __init__(self, kp, ki, kd, sample_time):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.sample_time = sample_time
		self.e1 = 0.0
		self.integral = 0.0

	def control(self, error):
		"""
			Computes the control action from a given error using PID logic.

			Parameters:
				error (float): The current error between desired and actual position.

			Returns:
				float: The control output.
		"""
		self.integral += error * self.sample_time
		derivative = (error - self.e1) / self.sample_time
		self.e1 = error
		u = self.kp * error + self.ki * self.integral + self.kd * derivative
		return u

	def set_gains(self,kp,ki,kd):
		"""
			Sets the PID gains.
			Parameters:
				kp (float): Proportional gain.
				ki (float): Integral gain.
				kd (float): Derivative gain.
		"""
		self.kp = kp
		self.ki = ki
		self.kd = kd

	@staticmethod
	def decide_action(state, control_posic, control_theta):
		"""
			Decides the control action based on the position and angle errors.

			Parameters:
				state (list): The current state of the system [position, _, angle, _].
				control_posic (PIDController): PID controller for position.
				control_theta (PIDController): PID controller for angle.

			Returns:
				int: The control action (0 or 1) based on the PID outputs.
		"""
		error_posic = POSITION_REF - state[0]
		error_theta = THETA_REF - state[2]
		action = 1 if control_theta.control(error_theta) + control_posic.control(error_posic) < 0  else 0
		return action
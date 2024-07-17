from constants import THETA_REF,POSITION_REF


class PIDController:
	
	def __init__(self, kp, ki, kd, sample_time):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.sample_time = sample_time
		self.e1 = 0.0
		self.integral = 0.0
		
	def control(self, error):
		# Calculando a integral do erro
		self.integral += error * self.sample_time
		# Calculando a derivada do erro
		derivative = (error - self.e1) / self.sample_time
		# Atualizando o erro anterior
		self.e1 = error
		# Calculando a saída do PID
		# u = self.kp * error + self.ki * self.integral + self.kd * derivative
		u = self.kp * error + self.kd * derivative
		return u
	
	def set_gains(self,kp,ki,kd):
		self.kp = kp
		self.ki = ki
		self.kd = kd
  
	@staticmethod
	def decide_action(state, control_posic, control_theta):
		error_posic = POSITION_REF - state[0]
		error_theta = THETA_REF - state[2]
		
		action = 1 if control_theta.control(error_theta) + control_posic.control(error_posic) < 0  else 0
		return action
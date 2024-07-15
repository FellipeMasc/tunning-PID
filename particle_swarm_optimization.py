import numpy as np
import random
from math import inf

import utils


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.pos = np.random.uniform(lower_bound, upper_bound)
        vel_min = lower_bound - upper_bound
        vel_max = upper_bound - lower_bound
        self.vel = np.random.uniform(vel_min, vel_max)
        self.best_pos = self.pos
        self.best_value = -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.vel_min = lower_bound - upper_bound
        self.vel_max = upper_bound - lower_bound
        self.cont = 0
        self.global_best_pos = np.array(np.size(self.lower_bound))
        self.global_best_value = -inf
        self.particles = []
        self.num = hyperparams.num_particles
        self.w = hyperparams.inertia_weight
        self.phip = hyperparams.cognitive_parameter
        self.phig = hyperparams.social_parameter
        for i in range(self.num):
            self.particles.append(Particle(lower_bound, upper_bound))

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.global_best_pos  # Remove this line

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.global_best_value  # Remove this line

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.particles[self.cont].pos  # Remove this line

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # Todo: implement
        for i in range(self.num):
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)
            self.particles[i].vel = self.w*self.particles[i].vel + self.phip*rp*(self.particles[i].best_pos - self.particles[i].pos) + self.phig*rg*(self.global_best_pos - self.particles[i].pos)
            for k in range(len(self.particles[i].vel)):
                if self.particles[i].vel[k] > self.vel_max[k]:
                    self.particles[i].vel[k] = self.vel_max[k]
                elif self.particles[i].vel[k] < self.vel_min[k]:
                    self.particles[i].vel[k] = self.vel_min[k]
            self.particles[i].pos += self.particles[i].vel
            for j in range(len(self.particles[i].pos)):
                if self.particles[i].pos[j] > self.upper_bound[j]:
                    self.particles[i].pos[j] = self.upper_bound[j]
                elif self.particles[i].pos[j] < self.lower_bound[j]:
                    self.particles[i].pos[j] = self.lower_bound[j]


    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        if value > self.particles[self.cont].best_value:
            self.particles[self.cont].best_value = value
            self.particles[self.cont].best_pos = self.particles[self.cont].pos
        if value > self.global_best_value:
            self.global_best_value = value
            self.global_best_pos = self.particles[self.cont].pos

        self.cont += 1

        if self.cont == self.num:
            self.cont = 0
            ParticleSwarmOptimization.advance_generation(self)


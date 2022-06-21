import numpy as np
import time
import os


class QuantumEvAlgorithm:
    """This object encapsules the necessary methods to make a Quantum-Inspired
    optimization algorithm. For a thorough description of the algorithm please visit:
    URL"""

    def __init__(self, f, n_dims, sigma_scaler=1.00001, mu_scaler=100, elitist_level=2):
        """The QuantumEvAlgorithm class admits a (scalar) function to be optimized. The function
        must be able to generate multiple outputs for multiple inputs of shape (n_samples,n_dimensions).
        The n_dims attribute is to be placed as an input of the class"""
        self.cost_function = f
        self.n_dims = n_dims
        self.sigma_scaler = sigma_scaler
        self.mu_scaler = mu_scaler
        self.elitist_level = elitist_level

    def quantum_individual_init(self):
        """Creates a Quantum individual of n_dims features. For each feature mu and sigma are created
         (normal distribution).
         First row: mean (mu)
         Second row: std deviation (sigma)
         In this case, the mu creation is random. The initialization of the std deviation is done so that
         a significant part of the domain is covered. (-32<x_i<32) Domain information must be an input of the class.
         (PENDING)"""
        # First row: mean (mu)
        # Second row: std deviation (sigma)
        np.random.seed(4)
        Q = 1 * np.random.rand(2, self.n_dims)
        Q[1, :] = 1 * np.ones(self.n_dims)
        self.best_of_best = Q[0:1, :]  # Initial definition of best_of_best
        # print(Q)
        return Q

    def quantum_sampling(self, Q, n_samples):
        """This method generates n_samples from Q (each sample feature is generated with its correspondent
        mu_i and sigma_i)"""
        samples = np.random.normal(Q[0, :], Q[1, :], size=(n_samples, self.n_dims))

        return samples

    def sample_evaluation(self, samples):
        """Each of the generated samples is evaluated against the cost function. In this case, the best
        performing sample is the output of the method."""
        cost = self.cost_function(samples)
        sort_order = np.argsort(cost, axis=0)
        best_performing_sample = samples[sort_order[0]]

        return best_performing_sample

    def elitist_sample_evaluation(self, samples):
        """This function is analog to the previous one. Instead of choosing the best, it computes the mean of the
        10 best samples."""
        cost = self.cost_function(samples)
        sort_order = np.argsort(cost, axis=0)
        elitist_level = 3
        best_performing_sample = np.mean(samples[sort_order[0:elitist_level]], axis=0)[None]

        return best_performing_sample

    def elitist_sample_evaluation_2(self, samples):
        """This function is analog to the previous one. Instead of choosing the best, it computes the mean of the
        10 best samples."""
        cost = self.cost_function(samples)
        sort_order = np.argsort(cost, axis=0)
        elitist_level = self.elitist_level
        best_performing_sample = np.mean(samples[sort_order[0:elitist_level]], axis=0)[None]

        if self.cost_function(self.best_of_best) > self.cost_function(best_performing_sample):
            self.best_of_best = best_performing_sample

        return self.best_of_best

    def quantum_update(self, Q, best_performing_sample):
        """This method updates the Quantum individual with the criteria explained in: URL (PENDING).
        The update mainly depends in two hyper-parameters as defined below:

        scaling: It controls the transformation of mu_(j+1)
        sigma_scaler: It controls the transformation of sigma_(j+1).

        It is PENDING to put them as inputs to the class (Global parameters of the algorithm)"""

        mu = Q[0:1, :]
        sigma = Q[1:2, :]

        scaling = self.mu_scaler
        mu_delta = best_performing_sample - mu

        updated_mu = mu + mu_delta / scaling

        sigma_decider = mu_delta / sigma

        sigma_scaler = self.sigma_scaler

        updated_sigma = (sigma_decider < 1) * sigma / sigma_scaler + (sigma_decider > 1) * sigma * sigma_scaler
        # if self.cost_function(updated_mu)>10e-10:
        #     updated_sigma[updated_sigma < 0.01] = updated_sigma[updated_sigma < 0.01] * 1.5

        Q[0:1] = updated_mu
        Q[1:2] = updated_sigma

        return Q

    def training(self, N_iterations=100000, sample_size=5, sample_increaser_factor=0):
        j = 0

        Q = self.quantum_individual_init()
        saving_interval = 50

        Q_history = np.zeros((int(N_iterations / saving_interval), 2, self.n_dims))
        best_performer_marker = np.zeros((int(N_iterations / saving_interval), 1))
        function_evaluations = np.zeros(int(N_iterations / saving_interval))
        print('Beginning of the iteration process')
        beginning = time.time()
        for i in range(N_iterations):
            adapted_sample_size = int(sample_size * (1 + sample_increaser_factor * (i / N_iterations)))
            samples = self.quantum_sampling(Q, adapted_sample_size)
            best_performer = self.elitist_sample_evaluation(samples)
            Q = self.quantum_update(Q, best_performer)

            if np.mod(i, saving_interval) == 0:
                Q_history[j, :, :] = Q
                output = self.cost_function(best_performer)
                best_performer_marker[j, :] = output
                function_evaluations[j] = i * (sample_size + (sample_increaser_factor * i) / 2)
                j += 1

            if np.mod(i, 50000) == 0:
                print(f'Iteration {i}, Best cost = {output}')
        end = time.time()

        print(f'The algorithm took {end - beginning} seconds')
        print(f' min is  = {self.cost_function(best_performer)}')
        print(f'The min is IN = {best_performer}')

        results_path = 'Results'

        np.savez(os.path.join(results_path, "testing_evl.npz"), best_performer_marker, Q_history, function_evaluations,
                 cost_h=best_performer_marker,
                 pos_history=Q_history, time=function_evaluations)

        print('End of training')

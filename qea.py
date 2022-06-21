import numpy as np


class QuantumEvAlgorithm:
    """This object encapsules the necessary methods to make a Quantum-Inspired
    optimization algorithm. For a thorough description of the algorithm please visit:
    URL"""

    def __init__(self, f, n_dims):
        """The QuantumEvAlgorithm class admits a (scalar) function to be optimized. The function
        must be able to generate multiple outputs for multiple inputs of shape (n_samples,n_dimensions).
        The n_dims attribute is to be placed as an input of the class"""
        self.cost_function = f
        self.n_dims = n_dims

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
        Q = - 2.04 + 4 * np.random.rand(2, self.n_dims)
        Q[1, :] = 2 * np.ones(self.n_dims)
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
        elitist_level = 2
        best_performing_sample = np.mean(samples[sort_order[0:elitist_level]], axis=0)[None]

        if self.cost_function(self.best_of_best) > self.cost_function(best_performing_sample):
            self.best_of_best = best_performing_sample

        return self.best_of_best

    def quantum_update(self, Q, best_performing_sample,i):
        """This method updates the Quantum individual with the criteria explained in: URL (PENDING).
        The update mainly depends in two hyper-parameters as defined below:

        scaling: It controls the transformation of mu_(j+1)
        sigma_scaler: It controls the transformation of sigma_(j+1).

        It is PENDING to put them as inputs to the class (Global parameters of the algorithm)"""

        mu = Q[0:1, :]
        sigma = Q[1:2, :]

        scaling =  200
        mu_delta = best_performing_sample - mu

        updated_mu = mu + mu_delta / scaling

        sigma_decider =  mu_delta / sigma

        sigma_scaler = 1.00001

        updated_sigma = (sigma_decider < 1) * sigma / sigma_scaler + (sigma_decider > 1) * sigma * sigma_scaler
        if self.cost_function(updated_mu)>10e-10:
            updated_sigma[updated_sigma < 0.01] = updated_sigma[updated_sigma < 0.01] * 1.5


        Q[0:1] = updated_mu
        Q[1:2] = updated_sigma

        return Q

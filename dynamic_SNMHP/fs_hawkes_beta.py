import numpy as np
from scipy.stats import beta
from scipy.special import expit
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import invgauss
from scipy.stats import dirichlet
from scipy.special import psi
from numpy.polynomial import legendre
from pypolyagamma import PyPolyaGamma
import copy
import time
from tqdm import trange

class FSHawkesBeta:
    """
    This class implements flexible state-switching Hawkes processes with Beta densities as basis functions.
    The main features it provides include simulation and statistical inference. 
    """
    def __init__(self, number_of_states, number_of_dimensions, number_of_basis):
        r"""
        Initialises an instance.

        :type number_of_states: int
        :param number_of_states: number of states
        :type number_of_dimensions: int
        :param number_of_dimensions: number of dimensions
        :type number_of_basis: int
        :param number_of_basis: number of basis functions (beta densities)
        """
        self.number_of_states = number_of_states
        self.number_of_dimensions = number_of_dimensions
        self.number_of_basis = number_of_basis
        self.beta_ab = np.zeros((number_of_basis, 3))
        self.T_phi = 0

        self.lamda_ub = np.zeros(number_of_dimensions)
        self.lamda_ub_estimated = None
        self.base_activation = np.zeros((number_of_states, number_of_dimensions))
        self.base_activation_estimated = None
        self.weight = np.zeros((number_of_states, number_of_dimensions, number_of_dimensions, number_of_basis))
        self.weight_estimated = None
        self.P = np.zeros((number_of_dimensions, number_of_states, number_of_states))
        self.P_estimated = None

    def set_hawkes_hyperparameters(self, beta_ab, T_phi):
        r"""
        Fix the hyperparameters : parameters a, b, shift and scale (T_phi) for basis functions (Beta densities). 

        :type beta_ab: numpy array
        :param beta_ab: [[a,b,shift],[a,b,shift]...] for basis functions.
        :type T_phi: float
        :param T_phi: the support of influence functions (the scale of basis functions)
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(beta_ab) != (self.number_of_basis, 3):
            raise ValueError('given basis functions have incorrect shape')
        if np.shape(T_phi) != ():
            raise ValueError('given T_phi has incorrect shape')
        self.beta_ab = copy.copy(beta_ab)
        self.T_phi = copy.copy(T_phi)

    def set_hawkes_parameters(self, lamda_ub, base_activation, weight, P):
        r"""
        Fix the parameters: intensity upperbound, base activation, influence weight and state-transition matrix. 
        It is used in the simulation. 

        :type lamda_ub: 1D numpy array
        :param lamda_ub: :math:`\lambda_i`.
        :type base_activation: 2D numpy array
        :param base_activation: :math:`\mu_i^{z(t)}`.
        :type weight: number_of_states * number_of_dimensions * number_of_dimensions * number_of_basis numpy array
        :param weight: :math:`w_{ijb}^{z(t)}`.
        :type P: number_of_dimensions * number_of_states * number_of_states numpy array
        :param P: :math:`\Phi_i`.
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(lamda_ub) != (self.number_of_dimensions,):
            raise ValueError('given intensity upperbounds have incorrect shape')
        if np.shape(base_activation) != (self.number_of_states, self.number_of_dimensions):
            raise ValueError('given base activations have incorrect shape')
        if np.shape(weight) != (self.number_of_states, self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis):
            raise ValueError('given weight have incorrect shape')
        if np.shape(P) != (self.number_of_dimensions, self.number_of_states, self.number_of_states):
            raise ValueError('given transition matrixes have incorrect shape')
        self.lamda_ub = copy.copy(lamda_ub)
        self.base_activation = copy.copy(base_activation)
        self.weight = copy.copy(weight)
        self.P = copy.copy(P)

    def set_hawkes_parameters_estimated(self, lamda_ub_estimated, W_estimated, P_estimated):
        r"""
        Set the estimated intensity upperbound, base activation and influence weight, and state-transition matrix. 
        It is used in the visualization. 

        :type lamda_ub_estimated: 1D numpy array
        :param lamda_ub_estimated: :math:`\hat{\lambda_i}`.
        :type W_estimated: number_of_states * number_of_dimensions * (number_of_dimensions * number_of_basis + 1) numpy array
        :param W_estimated: `W[:,:,0]` is the estimated base activation of all dimensions at all states, `W[:,:,1:]` is the estimated influence weight of all dimensions at all states
        :type P_estimated: number_of_dimensions * number_of_states * number_of_states numpy array
        :param P_estimated: :math:`\hat{\Phi_i}`.
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(lamda_ub_estimated) != (self.number_of_dimensions,):
            raise ValueError('given estimated intensity upperbounds have incorrect shape')
        if np.shape(W_estimated) != (self.number_of_states, self.number_of_dimensions, self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('given estimated weights have incorrect shape')
        if np.shape(P_estimated) != (self.number_of_dimensions, self.number_of_states, self.number_of_states):
            raise ValueError('given estimated transition matrixes have incorrect shape')
        self.lamda_ub_estimated = copy.copy(lamda_ub_estimated)
        self.base_activation_estimated = copy.copy(W_estimated[:,:,0].reshape((self.number_of_states, self.number_of_dimensions)))
        self.weight_estimated = copy.copy(W_estimated[:,:,1:].reshape((self.number_of_states, self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)))
        self.P_estimated = copy.copy(P_estimated)

    def intensity(self, t, target_state, target_dimension, timestamps_history, estimation = False):
        """
        Given the historical timestamps, evaluate the conditional intensity of t at target state on the target dimension.
        It is used in the simulation and visualization. If `estimation` is False, the intensity function is using 
        the ground truth parameters; if `estimation` is True, the intensity function is using the estimated parameters. 

        :type t: float
        :param t: the target time
        :type target_state: int
        :param target_state: the target state
        :type target_dimension: int
        :param target_dimension: the target dimension
        :type timestamps_history: list
        :param timestamps_history: [[t_1,t_2,...,t_N_1],[t_1,t_2,...,t_N_2],...], the historical timestamps before t
        :type estimation: bool
        :param estimation: indicate to use whether the ground-truth or estimated parameters

        :rtype: float
        :return: the conditional intensity of t at target state on the target dimension
        """
        # Raise ValueError if the given historical timestamps do not have the right shape
        if len(timestamps_history) != self.number_of_dimensions:
            raise ValueError('given historical timestamps have incorrect shape')
        if estimation == False:
            lamda_ub_target_dimension = self.lamda_ub[target_dimension]
            base_activation_target_state_dimension = self.base_activation[target_state, target_dimension]
            weight_target_state_dimension = self.weight[target_state, target_dimension]
        else:
            lamda_ub_target_dimension = self.lamda_ub_estimated[target_dimension]
            base_activation_target_state_dimension = self.base_activation_estimated[target_state, target_dimension]
            weight_target_state_dimension = self.weight_estimated[target_state, target_dimension]
        intensity = 0
        for n in range(self.number_of_dimensions):
            for i in range(len(timestamps_history[n])):
                if timestamps_history[n][i] >= t:
                    break
                elif t - timestamps_history[n][i] > self.T_phi: 
                    continue
                for b in range(self.number_of_basis):
                    intensity += weight_target_state_dimension[n][b] * beta.pdf(t - timestamps_history[n][i], a = self.beta_ab[b][0], b = self.beta_ab[b][1], loc = self.beta_ab[b][2], scale = self.T_phi)
        return lamda_ub_target_dimension * expit(base_activation_target_state_dimension + intensity)

    def simulation(self, T, P_0):
        r"""
        Simulate a sample path of the flexible state-switching Hawkes processes with Beta densities as basis functions.
        
        :type T: float
        :param T: time at which the simulation ends.
        :type P_0: 1-D numpy array, (number_of_states)
        :param P_0: the probability with which to draw the initial state. 

        :rtype: list
        :return: the timestamps when events occur on each dimension, the overall state sequence, the state of timestamps on each dimension. 
        """
        t = 0
        points_hawkes = []
        states=[list(multinomial.rvs(n=1,p=P_0)).index(1)]
        states_n=[]
        for i in range(self.number_of_dimensions):
            points_hawkes.append([])
            states_n.append([])
        intensity_sup = sum(self.lamda_ub)
        while(t < T):
            r = expon.rvs(scale = 1 / intensity_sup)
            t += r
            sum_intensity = sum(self.intensity(t,states[-1],m,points_hawkes) for m in range(self.number_of_dimensions))
            assert sum_intensity <= intensity_sup, "intensity exceeds the upper bound"
            D = uniform.rvs(loc = 0,scale = 1)
            if D * intensity_sup <= sum_intensity:
                k = list(multinomial.rvs(1,[self.intensity(t,states[-1],m,points_hawkes) / sum_intensity for m in range(self.number_of_dimensions)])).index(1)
                points_hawkes[k].append(t)
                states_n[k].append(states[-1])
                states.append(list(multinomial.rvs(n=1,p=self.P[k][states[-1]])).index(1))
        if points_hawkes[k][-1] > T:
            del points_hawkes[k][-1]
            del states_n[k][-1]
            del states[-1]
        return points_hawkes, states, states_n


    # "Inference for transition matrix"
    # def P_estimation(self, states_n, states, points_hawkes):
    #     if len(points_hawkes) != self.number_of_dimensions:
    #         raise ValueError('given timestamps has incorrect shape')
    #     if len(states_n) != self.number_of_dimensions:
    #         raise ValueError('given states_n has incorrect shape')
    #     if any([len(points_hawkes[i]) != len(states_n[i]) for i in range(self.number_of_dimensions)]):
    #         raise ValueError('the dimension of timestamps and states_n mismatch')
    #     if len(states) != sum([len(points_hawkes[i]) for i in range(self.number_of_dimensions)])+1:
    #         raise ValueError('given states has incorrect shape')
    #     states_uni=np.unique(states)
    #     P=np.zeros((self.number_of_dimensions,self.number_of_states,self.number_of_states))
    #     points_all=np.sort(sum(points_hawkes,[]))
    #     for d in range(self.number_of_dimensions):
    #         for i in range(self.number_of_states):
    #             for j in range(self.number_of_states):
    #                 numerator=sum([[states_n[d][l],states[np.where(points_all==points_hawkes[d][l])[0][0]+1]]==[states_uni[i],states_uni[j]] for l in range(len(states_n[d]))])
    #                 denominator=states_n[d].count(states_uni[i])
    #                 P[d][i][j]=numerator/denominator
    #     return P


    "Inference: Gibbs Sampler"


    PG = PyPolyaGamma() # we use PyPolyaGamma to sample from PolyaGamma distribution
    # @staticmethod
    # def PG(b,c): ## sampling from a PG distribution by truncation (default 2000 samples). It is not efficient. 
    #     g=gamma.rvs(b,size=2000)
    #     d=np.array(range(1,2001))
    #     d=(d-0.5)**2+c**2/4/np.pi/np.pi
    #     return sum(g/d)/2/np.pi/np.pi

    def Phi_t(self, t, points_hawkes):
        r"""
        Evaluate \Phi(t)=[1,\Phi_{11}(t),...,\Phi_{MB}(t)] where \Phi_{jb}(t) is the cumulative influence on t
        of the j-th dimensional observation by the b-th basis function

        :type t: float
        :param t: the target time
        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension

        :rtype: 1D numpy array
        :return: \Phi(t)=[1,\Phi_{11}(t),...,\Phi_{MB}(t)]
        """
        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('given timestamps have incorrect shape')
        Phi_t = [1]
        for i in range(self.number_of_dimensions):
            for j in range(self.number_of_basis):
                index = (np.array(points_hawkes[i]) < t) & ((t - np.array(points_hawkes[i])) <= self.T_phi)
                Phi_t.append(sum(beta.pdf(t - np.array(points_hawkes[i])[index], a=self.beta_ab[j][0], b=self.beta_ab[j][1], loc=self.beta_ab[j][2], scale=self.T_phi)))
        return np.array(Phi_t)

    def Phi_n_g_States_g(self, points_hawkes, states, points_g):
        r"""
        Evaluate \Phi(t) on all observed points and grid (Gaussian quadrature) nodes, and the state of grid (Gaussian quadrature) nodes

        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension
        :type states: 1D numpy array
        :param states: the overall state sequence 
        :type points_g: list
        :param points_g: the timestamps of grid or Gaussian quadrature nodes on [0,T]

        :rtype: number_of_dimensions*N_i*(number_of_dimensions*number_of_basis+1), num_g*(number_of_dimensions*number_of_basis+1), 1-D numpy array
        :return: list of \Phi(t_n), \Phi(t_g), state on grid or quadrature nodes
        """
        points_hawkes_sum = np.sort(sum(points_hawkes,[]))
        N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)])
        num_g = len(points_g)
        Phi_n = [np.zeros((N[d],self.number_of_dimensions*self.number_of_basis+1)) for d in range(self.number_of_dimensions)]
        for d in range(self.number_of_dimensions):
            for n in range(N[d]):
                Phi_n[d][n] = self.Phi_t(points_hawkes[d][n],points_hawkes)
        Phi_g = np.zeros((num_g, self.number_of_dimensions*self.number_of_basis+1))
        for g in range(num_g):
            Phi_g[g] = self.Phi_t(points_g[g],points_hawkes)
        states_g = [states[sum(~(points_hawkes_sum>=t))] for t in points_g]
        return Phi_n, Phi_g, states_g

    def inhomo_simulation(self, intensity, T):
        r"""
        Simulate an inhomogeneous Poisson process with a discrete intensity function (vector)

        :type intensity: 1-D numpy array
        :param intensity: the discrete intensity function (vector)
        :type T: float
        :param T: the observation windown [0,T]

        :rtype: 1D list
        :return: [t_1, t_2, ..., t_N]
        """
        delta_t = T/len(intensity)
        Lambda = np.max(intensity)*T
        r = poisson.rvs(Lambda)
        x = uniform.rvs(size = r)*T
        measures = intensity[(x/delta_t).astype(int)]
        ar = measures/np.max(intensity)
        index = bernoulli.rvs(ar)
        points = x[index.astype(bool)]
        return points

    def loglikelyhood_gibbs(self, W, lamda, Phi_n, states_n, Phi_g, states_g, points_hawkes, T):
        r"""
        Evaluate the log-likelihood for the given timestamps in the iterative update of Gibbs sampler
        
        :type W: number_of_states * number_of_dimensions * (number_of_dimensions*number_of_basis +1) numpy array
        :param W: the input weight which includes the base activation
        :type lamda: 1D numpy array
        :param lamda: the input intensity upperbound for each dimension
        :type Phi_n: list of 1D numpy arrays
        :param Phi_n: the cumulative influence \Phi on each observed timestamp
        :type states_n: list of 1D numpy arrays
        :param states_n: the state of each observed timestamp
        :type Phi_g: numpy array (Q, number_of_dimensions * number_of_basis + 1)
        :param Phi_g: the cumulative influence \Phi on each grid node
        :type states_g: 1D numpy array
        :param states_g: the state of each grid node
        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension
        :type T: float
        :param T: the observation windown [0,T]

        :rtype: float
        :return: the log-likelihood for the given timestamps
        """
        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('given timestamps have incorrect shape')
        if np.shape(Phi_g) != (len(states_g), self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('the dimension of Phi_g or states_g is incorrect')
        for i in range(self.number_of_dimensions):
            if len(Phi_n[i]) != len(points_hawkes[i]):
                raise ValueError('the dimension of Phi_n is incorrect')
        N_g=len(states_g)
        logl = 0
        for i in range(self.number_of_dimensions):
            N_i = len(points_hawkes[i])
            for n in range(N_i):
                logl += np.log(expit(W[states_n[i][n]][i].dot(Phi_n[i][n])))+np.log(lamda[i])
            for n in range(N_g):
                logl -= lamda[i]*expit(W[states_g[n]][i].dot(Phi_g[n]))*T/N_g
        return logl

    def Gibbs_inference(self, points_hawkes, states, states_n, points_hawkes_test, states_test, states_n_test, \
        T, T_test, b, eta, num_grid, num_grid_test, num_iter, initial_W = None): 
        r"""
        Gibbs sampler which is used to sample from the posterior of 
        lamda_ub, weight at each state (base_activation is included in the weight) and state-transition matrix on each dimension
        
        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type states: 1D numpy array
        :param states: the overall state sequence in training data
        :type states_n: list of 1D numpy arrays
        :param states_n: the state of each observed timestamp in training data
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type states_test: 1D numpy array
        :param states_test: the overall state sequence in test data
        :type states_n_test: list of 1D numpy arrays
        :param states_n_test: the state of each observed timestamp in test data
        :type T: float
        :param T: time at which the simulation ends.
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type eta: float
        :param eta: the parameter of Dirichlet prior, default is 1
        :type num_grid: int
        :param num_grid: the number of grid nodes on [0,T]
        :type num_grid_test: int
        :param num_grid_test: the number of grid nodes on [0,T_test]
        :type num_iter: int
        :param num_iter: the number of Gibbs loop
        :type initial_W: numpy array
        :param initial_W: the initial value for W 

        :rtype: numpy array
        :return: the samples of lamda_ub (lamda), weight (W), state_transitionn matrix (P), the training (logl) and test log-likelihood (logl_test)
        by those samples. 
        """
        # number of points on each dimension 
        N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) 
        N_test = np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)])
        points_hawkes_sum=np.sort(sum(points_hawkes,[]))
        states_uni=np.unique(states)
        #initial W and lamda
        if initial_W is None:
            W = np.random.uniform(-1,1,size=(self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            W = copy.copy(initial_W)
        lamda = N / T
        beta = np.zeros((self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        w_n=[np.zeros(N[d]) for d in range(self.number_of_dimensions)]
        Phi_n,Phi_g,states_g=self.Phi_n_g_States_g(points_hawkes, states, np.linspace(0,T,num_grid))
        intensity_g=np.zeros(num_grid)
        t_m=[[] for d in range(self.number_of_dimensions)]
        states_m=[[] for d in range(self.number_of_dimensions)]
        w_m=[[] for d in range(self.number_of_dimensions)]
        Phi_m=[[] for d in range(self.number_of_dimensions)] # precompute Phi(t) and state on the thinned points

        P=np.zeros((self.number_of_dimensions,self.number_of_states,self.number_of_states)) # the transition matrix
        counts=np.zeros((self.number_of_dimensions,self.number_of_states,self.number_of_states))
        for d in range(self.number_of_dimensions): # precompute the counts on i-th dimension, from k to k'
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    counts[d][i][j]=sum([[states_n[d][n],states[np.where(points_hawkes_sum==points_hawkes[d][n])[0][0]+1]]\
                                   ==[states_uni[i],states_uni[j]] for n in range(N[d])])

        lamda_list=[]
        W_list=[]
        P_list=[] 

        logl=[]
        logl_test=[]
        Phi_n_test,Phi_g_test,states_g_test=self.Phi_n_g_States_g(points_hawkes_test,states_test,np.linspace(0,T_test,num_grid_test))
        
        for ite in trange(num_iter):
            for d in range(self.number_of_dimensions):
                # sample P_d
                for z in range(self.number_of_states):
                    P[d][z]=dirichlet.rvs(counts[d][z]+eta)
                
                # sample w_n
                for n in range(N[d]):
                    w_n[d][n]=self.PG.pgdraw(1,W[states_n[d][n]][d].dot(Phi_n[d][n]))
                
                # sample t_m and w_m
                for g in range(num_grid):
                    intensity_g[g]=lamda[d]*expit(-W[states_g[g]][d].dot(Phi_g[g]))
                t_m[d]=self.inhomo_simulation(intensity_g,T)
                states_m[d]=[states[sum(~(points_hawkes_sum>=t))] for t in t_m[d]]
                Phi_m[d]=np.array([self.Phi_t(t,points_hawkes) for t in t_m[d]])
                w_m[d]=np.array([self.PG.pgdraw(1,W[states_m[d][m]][d].dot(Phi_m[d][m])) for m in range(len(t_m[d]))])
                
                # sample lamda
                lamda[d]=gamma(a=N[d]+len(t_m[d]),scale=1/T).rvs()
                
                # sample W
                for z in range(self.number_of_states):
                    index_z_n=np.where(np.array(states_n[d])==z)
                    index_z_m=np.where(np.array(states_m[d])==z)
                    v_z=np.array([0.5]*len(index_z_n[0])+[-0.5]*len(index_z_m[0]))
                    Sigma_z_inv=np.diag(list(w_n[d][index_z_n])+list(w_m[d][index_z_m]))
                    Phi_z=np.concatenate((Phi_n[d][index_z_n],Phi_m[d][index_z_m]))
                    Sigma_W_z=np.linalg.inv((Phi_z.T).dot(Sigma_z_inv).dot(Phi_z)+np.diag(beta[z][d]/b/b))
                    mean_W_z=Sigma_W_z.dot(Phi_z.T).dot(v_z)
                    W[z][d]=multivariate_normal(mean=mean_W_z,cov=Sigma_W_z).rvs()
                    # for numerical stability, we truncate W if it is too close to 0
                    W[z][d][np.abs(W[z][d])<1e-200]=1e-200*np.sign(W[z][d][np.abs(W[z][d])<1e-200])

                # sample beta
                for z in range(self.number_of_states):
                    beta[z][d] = invgauss.rvs(b / np.abs(W[z][d]))
            lamda_list.append(lamda.copy())
            W_list.append(W.copy())
            P_list.append(P.copy())
            
            # compute the loglikelihood
            logl.append(self.loglikelyhood_gibbs(W,lamda,Phi_n,states_n,Phi_g,states_g,points_hawkes,T))
            logl_test.append(self.loglikelyhood_gibbs(W,lamda,Phi_n_test,states_n_test,Phi_g_test,states_g_test,points_hawkes_test,T_test))
        return lamda_list,W_list,P_list,logl,logl_test

    'tool functions'
    def influence_function_estimated(self, z, i, j, t, weight_estimated, gt = False):
        r"""
        Evaluate the influence function based on the basis functions and the influence weight W.
        It is used to visualize the influence functions. If gt = False, it uses the estimated parameters;
        if gt = True, it uses the ground truth parameters. 
        
        :type z: int
        :param z: the target state. \phi_{ij}^{z}(t)
        :type i: int
        :param i: the target dimension. \phi_{ij}^{z}(t)
        :type j: int
        :param j: the source dimension. \phi_{ij}^{z}(t)
        :type t: float
        :param t: the target time. \phi_{ij}^{z}(t)
        :type weight_estimated: number_of_states*number_of_dimensions*(number_of_dimensions*number_of_basis+1) numpy array
        :param weight_estimated: the estimated weights (including base activation)
        :type gt: bool
        :param gt: indicate to use whether the ground-truth or estimated parameters

        :rtype: float
        :return: the influence function \phi_{ij}^{z}(t)
        """
        if gt == False:
            W_phi = weight_estimated[:,:,1:].reshape(self.number_of_states, self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)
        else:
            W_phi = self.weight
        phi_t=np.array([beta.pdf(t,a=self.beta_ab[i][0],b=self.beta_ab[i][1],loc=self.beta_ab[i][2],scale=self.T_phi) for i in range(self.number_of_basis)])
        return W_phi[z][i][j].dot(phi_t)

    "Inference: EM Algorithm"

    # @staticmethod
    # def gq_points_weights(a,b,Q):
    #     r"""
    #     Generate the Gaussian quadrature nodes and weights for the integral :math:`\int_a^b f(t) dt`

    #     :type a: float
    #     :param a: the lower end of the integral
    #     :type b: float
    #     :param b: the upper end of the integral
    #     :type Q: int
    #     :param Q: the number of Gaussian quadrature nodes (weights)

    #     :rtype: 1D numpy array, 1D numpy array
    #     :return: Gaussian quadrature nodes and the corresponding weights
    #     """
    #     p,w = legendre.leggauss(Q)
    #     c = np.array([0] * Q + [1])
    #     p_new = (a + b + (b - a) * p) / 2
    #     w_new = (b - a) / (legendre.legval(p, legendre.legder(c))**2*(1-p**2))
    #     return p_new, w_new

    def loglikelyhood_em_mf(self, W, lamda, Phi_n, states_n, Phi_gq, states_gq, T, points_hawkes):
        r"""
        Evaluate the log-likelihood for the given timestamps in the iterative update of EM algorithm and mean-field approximation
        
        :type W: number_of_states * number_of_dimensions * (number_of_dimensions*number_of_basis +1) numpy array
        :param W: the input weight which includes the base activation
        :type lamda: 1D numpy array
        :param lamda: the input intensity upperbound for each dimension
        :type Phi_n: list of 1D numpy arrays
        :param Phi_n: the cumulative influence \Phi on each observed timestamp
        :type states_n: list of 1D numpy arrays
        :param states_n: the state of each observed timestamp
        :type Phi_gq: numpy array (gq, number_of_dimensions * number_of_basis + 1)
        :param Phi_gq: the cumulative influence \Phi on each gaussian quadrature node
        :type states_gq: 1D numpy array
        :param states_gq: the state of each gaussian quadrature node
        :type T: float
        :param T: the time window
        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension

        :rtype: float
        :return: the log-likelihood for the given timestamps
        """
        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('given timestamps have incorrect shape')
        if np.shape(Phi_gq) != (len(states_gq), self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('the dimension of Phi_g or states_g is incorrect')
        for i in range(self.number_of_dimensions):
            if len(Phi_n[i]) != len(points_hawkes[i]):
                raise ValueError('the dimension of Phi_n is incorrect')
        N_gq=len(states_gq)
        logl=0
        for i in range(self.number_of_dimensions):
            N_i=len(points_hawkes[i])
            for n in range(N_i):
                logl+=np.log(expit(W[states_n[i][n]][i].dot(Phi_n[i][n])))+np.log(lamda[i])
            for n in range(N_gq):
                logl-=(T/N_gq)*lamda[i]*expit(W[states_gq[n]][i].dot(Phi_gq[n]))
        return logl

    def EM_inference(self, points_hawkes, states, states_n, points_hawkes_test, states_test, states_n_test, \
        T, T_test, b, num_gq, num_gq_test, num_iter, initial_W = None): 
        r"""
        EM algorithm which is used to estimate the MAP estimation of parameters: 
        lamda_ub, weight at each state (base_activation is included in the weight) and state-transition matrix on each dimension
        
        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type states: 1D numpy array
        :param states: the overall state sequence in training data
        :type states_n: list of 1D numpy arrays
        :param states_n: the state of each observed timestamp in training data
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type states_test: 1D numpy array
        :param states_test: the overall state sequence in test data
        :type states_n_test: list of 1D numpy arrays
        :param states_n_test: the state of each observed timestamp in test data
        :type T: float
        :param T: time at which the simulation ends.
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type num_gq: int
        :param num_gq: the number of gaussian quadrature nodes on [0,T]
        :type num_gq_test: int
        :param num_gq_test: the number of gaussian quadrature nodes on [0,T_test]
        :type num_iter: int
        :param num_iter: the number of iterations
        :type initial_W: numpy array
        :param initial_W: the initial value for W

        :rtype: numpy array
        :return: the MAP estimation of lamda_ub (lamda), weight (W) and state-transition matrix (P),
        the training (logl) and test log-likelihood (logl_test) along EM iterations. 
        """
        points_hawkes_sum=np.sort(sum(points_hawkes,[]))
        # number of points on each dimension 
        N=np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) # of points on each dimension 
        N_test = np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)])
        states_uni=np.unique(states)
        #initial W and lamda
        if initial_W is None:
            W = np.random.uniform(-1,1,size=(self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            W = copy.copy(initial_W)
        lamda = N / T
        E_beta = np.zeros((self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        E_w_n = [np.zeros(N[d]) for d in range(self.number_of_dimensions)]
        H_n = [np.zeros(N[d]) for d in range(self.number_of_dimensions)]
        Phi_n,Phi_gq,states_gq=self.Phi_n_g_States_g(points_hawkes, states, np.linspace(0,T,num_gq))
        Phi_n_test,Phi_gq_test,states_gq_test=self.Phi_n_g_States_g(points_hawkes_test, states_test, np.linspace(0,T_test,num_gq_test))
        H_gq = np.zeros((self.number_of_dimensions,num_gq))
        int_intensity = np.zeros(self.number_of_dimensions)

        logl = []
        logl_test = []

        counts=np.zeros((self.number_of_dimensions,self.number_of_states,self.number_of_states))
        for d in range(self.number_of_dimensions): # precompute the counts on i-th dimension, from k to k'
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    counts[d][i][j]=sum([[states_n[d][n],states[np.where(points_hawkes_sum==points_hawkes[d][n])[0][0]+1]]\
                                   ==[states_uni[i],states_uni[j]] for n in range(N[d])])

        P = counts/np.sum(counts,axis=2).reshape(self.number_of_dimensions,self.number_of_states,1)
        
        for ite in trange(num_iter):
            for d in range(self.number_of_dimensions):
                # update w_d
                for n in range(N[d]):
                    H_n[d][n] = W[states_n[d][n]][d].dot(Phi_n[d][n])
                    E_w_n[d][n] = 1/2/H_n[d][n]*np.tanh(H_n[d][n]/2)

                # update Pi_d
                for m in range(num_gq):
                    H_gq[d][m] = W[states_gq[m]][d].dot(Phi_gq[m])
                int_intensity[d] = np.sum(lamda[d]*expit(-H_gq[d])*(T/num_gq))

                # update beta_d
                for z in range(self.number_of_states):
                    E_beta[z][d] = b / np.abs(W[z][d])
                
                # update lamda_d
                lamda[d]=(int_intensity[d]+N[d])/T

                # update W_d
                for z in range(self.number_of_states):
                    index_n_z=np.where(np.array(states_n[d])==z)[0]
                    index_gq_z=np.where(np.array(states_gq)==z)[0]
                    
                    int_A=np.zeros((self.number_of_dimensions*self.number_of_basis+1,self.number_of_dimensions*self.number_of_basis+1))
                    for n in index_n_z:
                        int_A+=E_w_n[d][n]*np.outer(Phi_n[d][n],Phi_n[d][n])
                    for m in index_gq_z:
                        int_A+=(T/num_gq)*(lamda[d]/2/H_gq[d][m]*np.tanh(H_gq[d][m]/2)*expit(-H_gq[d][m])*np.outer(Phi_gq[m],Phi_gq[m]))
                    
                    int_B=np.zeros(self.number_of_dimensions*self.number_of_basis+1)
                    for n in index_n_z:
                        int_B+=0.5*Phi_n[d][n]
                    for m in index_gq_z:
                        int_B+=-(T/num_gq)/2*(lamda[d]*expit(-H_gq[d][m])*Phi_gq[m])

                    W[z][d]=np.linalg.inv(int_A+np.diag(E_beta[z][d]/b/b)).dot(int_B)
                    # for numerical stability, we truncate W if it is too close to 0
                    W[z][d][np.abs(W[z][d])<1e-200]=1e-200*np.sign(W[z][d][np.abs(W[z][d])<1e-200])

            # compute the loglikelihood
            logl.append(self.loglikelyhood_em_mf(W,lamda,Phi_n,states_n,Phi_gq,states_gq,T,points_hawkes))
            logl_test.append(self.loglikelyhood_em_mf(W,lamda,Phi_n_test,states_n_test,Phi_gq_test,states_gq_test,T_test,points_hawkes_test))
        return lamda, W, P, logl, logl_test


    "Inference: Mean-Field Variational Inference"

    @staticmethod
    def a_c_predict_n(z,d,n,Phi,mean_W,cov_W):
        r"""
        Compute the a=E[h_i^z(t)], c=E[h_i^z(t)^2] of observed points

        :type z: int
        :param z: the target state
        :type d: int
        :param d: the target dimension
        :type n: int
        :param n: the target n-th timestamp on d-th dimension
        :type Phi: number_of_dimension*N_i*(number_of_dimension*number_of_basis+1) list
        :param Phi: the cumulative influence on each timestamp
        :type mean_W: number_of_states*number_of_dimension*(number_of_dimension*number_of_basis+1) numpy array
        :param mean_W: the mean of weights
        :type cov_W: number_of_states*number_of_dimension*(number_of_dimension*number_of_basis+1)*(number_of_dimension*number_of_basis+1) numpy array
        :param cov_W: the covariance of weights

        :rtype: float
        :return: c=E[h_i^z(t)^2]
        """
        a = Phi[d][n].dot(mean_W[z][d])
        c = np.sqrt(a**2+(Phi[d][n].T).dot(cov_W[z][d]).dot(Phi[d][n]))
        return c

    @staticmethod
    def a_c_predict_gq(z,d,n,Phi,mean_W,cov_W):
        r"""
        Compute the a=E[h_i^z(t)], c=E[h_i^z(t)^2] of Gaussian quadrature nodes

        :type z: int
        :param z: the target state
        :type d: int
        :param d: the target dimension
        :type n: int
        :param n: the target n-th timestamp on d-th dimension
        :type Phi: N_gq*(number_of_dimension*number_of_basis+1) list
        :param Phi: the cumulative influence on gaussian quadrature nodes
        :type mean_W: number_of_states*number_of_dimension*(number_of_dimension*number_of_basis+1) numpy array
        :param mean_W: the mean of weights
        :type cov_W: number_of_states*number_of_dimension*(number_of_dimension*number_of_basis+1)*(number_of_dimension*number_of_basis+1) numpy array
        :param cov_W: the covariance of weights

        :rtype: float, float
        :return: a=E[h_i^z(t)], c=E[h_i^z(t)^2]
        """
        a = Phi[n].dot(mean_W[z][d])
        c = np.sqrt(a**2+(Phi[n].T).dot(cov_W[z][d]).dot(Phi[n]))
        return a,c

    # def gq_points_weights_states_T(self, points_hawkes, states, T, num_gq_seg):
    #     r"""
    #     Compute the gaussian quadrature nodes and weights on each state-segment, and the state on each gaussian quadrature nodes
    #     :type points_hawkes: list
    #     :param points_hawkes: the timestamps when events occur on each dimension
    #     :type states: 1D numpy array
    #     :param states: the overall state sequence 
    #     :type T: float
    #     :param T: time at which the simulation ends.
    #     :type num_gq_seg: int
    #     :param num_gq_seg: the number of gaussian quadrature nodes on each state-segment
    #     :rtype: list, list, 1-D numpy array
    #     :return: gaussian quadrature nodes, gaussian quadrature weights, state of gaussian quadrature nodes
    #     """
    #     points_hawkes_sum=np.sort(sum(points_hawkes,[]))
    #     changing_points=[0]         
    #     states_changing_points=[states[0]]
    #     for i in range(len(points_hawkes_sum)):
    #         if states[i]!=states[i+1]:
    #             changing_points.append(points_hawkes_sum[i])
    #             states_changing_points.append(states[i+1])
    #     changing_points.append(T)
    #     p_gq_T=[]
    #     w_gq_T=[]
    #     states_gq=[]
    #     for s in range(len(states_changing_points)):
    #         p_gq,w_gq=self.gq_points_weights(changing_points[s],changing_points[s+1],num_gq_seg) # num_gq_seg gauss quadureture points for segment
    #         p_gq_T+=list(p_gq)
    #         w_gq_T+=list(w_gq)
    #         states_gq+=list([states_changing_points[s]]*num_gq_seg)
    #     return p_gq_T,w_gq_T,np.array(states_gq)


    def MF_inference(self, points_hawkes, states, states_n, points_hawkes_test, states_test, states_n_test, \
        T, T_test, b, eta, num_gq, num_gq_test, num_iter, initial_W_mean = None):
        r"""
        Mean-field variational inference which is used to update the posterior of 
        lamda_ub, weight at each state (base_activation is included in the weight) and state-transition matrix on each dimension. 
        
        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type states: 1D numpy array
        :param states: the overall state sequence in training data
        :type states_n: list of 1D numpy arrays
        :param states_n: the state of each observed timestamp in training data
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type states_test: 1D numpy array
        :param states_test: the overall state sequence in test data
        :type states_n_test: list of 1D numpy arrays
        :param states_n_test: the state of each observed timestamp in test data
        :type T: float
        :param T: time at which the simulation ends.
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type eta: float
        :param eta: the parameter of Dirichlet prior, default is 1
        :type num_gq: int
        :param num_gq: the number of gaussian quadrature nodes on [0,T]
        :type num_gq_test: int
        :param num_gq_test: the number of gaussian quadrature nodes on [0,T_test]
        :type num_iter: int
        :param num_iter: the number of iterations
        :type initial_W_mean: numpy array
        :param initial_W_mean: the initial value for W_mean 

        :rtype: numpy array
        :return: the parameter of lamda_ub posterior, the parameter of weight posterior, the parameter of state-transition matrix posterior
        the training (logl) and test log-likelihood (logl_test) by the mean. 
        """
        points_hawkes_sum=np.sort(sum(points_hawkes,[]))
        N=np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) # of points on each dimension 
        states_uni=np.unique(states)
        #initialization
        #q_lamda_i is a gamma distribution gamma(alpha=N_i+E(|Pi_i|),scale=1/T)
        alpha=1.5*N
        lamda_1=np.zeros(self.number_of_dimensions)
        E_beta = np.zeros((self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        #q_W_i_z is a gaussian distribution N(mean_w_i_z,cov_w_i_z)
        if initial_W_mean is None:
            mean_W = np.random.uniform(-1,1,size=(self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            mean_W = copy.copy(initial_W_mean)
        cov_W=np.zeros((self.number_of_states,self.number_of_dimensions,(self.number_of_dimensions*self.number_of_basis+1),(self.number_of_dimensions*self.number_of_basis+1)))
        for i in range(self.number_of_dimensions*self.number_of_basis+1):
            cov_W[:,:,i,i]=1
        tilde_W=np.zeros((self.number_of_states,self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        for z in range(self.number_of_states):
            for i in range(self.number_of_dimensions):
                tilde_W[z][i]=np.sqrt(mean_W[z][i]**2+np.diag(cov_W[z][i]))   

        #precompute the relative variables
        c_n=[np.zeros(N[i]) for i in range(self.number_of_dimensions)] # parameter of wn, D*N_i 
        E_wn=[np.zeros(N[i]) for i in range(self.number_of_dimensions)]  
        a_gq=np.zeros((self.number_of_dimensions,num_gq))
        c_gq=np.zeros((self.number_of_dimensions,num_gq))
        Phi_n, Phi_gq, states_gq = self.Phi_n_g_States_g(points_hawkes, states, np.linspace(0,T,num_gq))
        
        logl=[]
        logl_test=[]

        N_test=np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)]) # of points on each dimension 
        Phi_n_test, Phi_gq_test, states_gq_test = self.Phi_n_g_States_g(points_hawkes_test, states_test, np.linspace(0,T_test,num_gq_test))
        
        counts=np.zeros((self.number_of_dimensions,self.number_of_states,self.number_of_states))
        for d in range(self.number_of_dimensions): # precompute the counts on i-th dimension, from k to k'
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    counts[d][i][j]=sum([[states_n[d][n],states[np.where(points_hawkes_sum==points_hawkes[d][n])[0][0]+1]]\
                                   ==[states_uni[i],states_uni[j]] for n in range(N[d])])
        
        for ite in trange(num_iter):
            for i in range(self.number_of_dimensions):
                #update parameters of q_w_n_i q_w_n_i=P_pg(wn|1,cn)
                for n in range(N[i]):
                    c_n[i][n]=self.a_c_predict_n(states_n[i][n],i,n,Phi_n,mean_W,cov_W)
                    E_wn[i][n]=1/2/c_n[i][n]*np.tanh(c_n[i][n]/2)
                
                #update parameters of q_Pi_i intensity=exp(E(log lamda))sigmoid(-c(x))exp((c(x)-a(x))/2)*P_pg(w|1,c(x))
                lamda_1[i]=np.exp(np.log(1/T)+psi(alpha[i]))

                #update parameters of q_beta_z_i=IG(b/\tilde{W_z_i},1)
                for z in range(self.number_of_states):
                    E_beta[z][i] = b / tilde_W[z][i]
                
                #update parameters of q_lamda_i q_lamda_i=gamma(alpha=N_i+E(|Pi_i|),scale=1/T)
                for n in range(num_gq):
                    a_gq[i][n],c_gq[i][n]=self.a_c_predict_gq(states_gq[n],i,n,Phi_gq,mean_W,cov_W)
                int_intensity=0
                for n in range(num_gq):
                    int_intensity+=(T/num_gq)*lamda_1[i]*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)
                alpha[i]=int_intensity+N[i]
                
                # update parameters of q_W_i_z  q_W_i_z=N(mean_W,cov_W)
                for z in range(self.number_of_states):
                    index_n_z=np.where(np.array(states_n[i])==z)[0]
                    index_gq_z=np.where(np.array(states_gq)==z)[0]
                    
                    int_A=np.zeros((self.number_of_dimensions*self.number_of_basis+1,self.number_of_dimensions*self.number_of_basis+1))
                    for n in index_n_z:
                        int_A+=E_wn[i][n]*np.outer(Phi_n[i][n],Phi_n[i][n])
                    for n in index_gq_z:
                        int_A+=(T/num_gq)*(lamda_1[i]/2/c_gq[i][n]*np.tanh(c_gq[i][n]/2)*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)*np.outer(Phi_gq[n],Phi_gq[n]))
                    
                    int_B=np.zeros(self.number_of_dimensions*self.number_of_basis+1)
                    for n in index_n_z:
                        int_B+=0.5*Phi_n[i][n]
                    for n in index_gq_z:
                        int_B+=-(T/num_gq)/2*(lamda_1[i]*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)*Phi_gq[n])
                    
                    cov_W[z][i]=np.linalg.inv(int_A+np.diag(E_beta[z][i]/b/b))
                    mean_W[z][i]=cov_W[z][i].dot(int_B)
                    tilde_W[z][i]=np.sqrt(mean_W[z][i]**2+np.diag(cov_W[z][i]))
                    # for numerical stability, we truncate tilde_W if it is too close to 0
                    tilde_W[z][i][tilde_W[z][i]<1e-200]=1e-200
            logl.append(self.loglikelyhood_em_mf(mean_W,alpha/T,Phi_n,states_n,Phi_gq,states_gq,T,points_hawkes))
            logl_test.append(self.loglikelyhood_em_mf(mean_W,alpha/T,Phi_n_test,states_n_test,Phi_gq_test,states_gq_test,T_test,points_hawkes_test))
        return alpha, mean_W, cov_W, counts+eta, logl, logl_test











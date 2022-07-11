import numpy as np
import copy
from scipy.stats import beta
from scipy.special import expit
from scipy.special import psi
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import invgauss
from numpy.polynomial import legendre
from pypolyagamma import PyPolyaGamma

class SNMHawkesBeta:
    """
    This class implements sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.
    The main features it provides include simulation and statistical inference. 
    """
    def __init__(self, number_of_dimensions, number_of_basis):
        """
        Initialises an instance.

        :type number_of_dimensions: int
        :param number_of_dimensions: number of dimensions (neurons)
        :type number_of_basis: int
        :param number_of_basis: number of basis functions (beta densities)
        """
        self.number_of_dimensions = number_of_dimensions
        self.number_of_basis = number_of_basis
        self.beta_ab = np.zeros((number_of_basis, 3))
        self.T_phi = 0

        self.lamda_ub = np.zeros(number_of_dimensions)
        self.lamda_ub_estimated = None
        self.base_activation = np.zeros(number_of_dimensions)
        self.base_activation_estimated = None
        self.weight = np.zeros((number_of_dimensions, number_of_dimensions, number_of_basis))
        self.weight_estimated = None

    def set_hawkes_hyperparameters(self, beta_ab, T_phi):
        r"""
        Fix the hyperparameters : parameters a, b, shift and scale for basis functions (Beta densities). 

        :type beta_ab: numpy array
        :param beta_ab: [[a,b,shift],[a,b,shift]...] for basis functions.
        :type T_phi: float
        :param T_phi: the support of influence functions (the scale of basis functions)
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(beta_ab) != (self.number_of_basis, 3):
            raise ValueError('given basis functions have incorrect shape')
        if np.shape(T_phi) != ():
            raise ValueError('given scale parameter has incorrect shape')
        self.beta_ab = copy.copy(beta_ab)
        self.T_phi = copy.copy(T_phi)

    def set_hawkes_parameters(self, lamda_ub, base_activation, weight):
        r"""
        Fix the parameters: intensity upperbound, base activation and influence weight. 
        It is used in the simulation.

        :type lamda_ub: 1D numpy array
        :param lamda_ub: :math:`\bar{\lambda}`.
        :type base_activation: 1D numpy array
        :param base_activation: :math:`\mu`.
        :type weight: number_of_dimensions*number_of_dimensions*number_of_basis numpy array
        :param weight: :math:`w_{ijb}`.
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(lamda_ub) != (self.number_of_dimensions,):
            raise ValueError('given intensity upperbounds have incorrect shape')
        if np.shape(base_activation) != (self.number_of_dimensions,):
            raise ValueError('given base activations have incorrect shape')
        if np.shape(weight) != (self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis):
            raise ValueError('given weight have incorrect shape')
        self.lamda_ub = copy.copy(lamda_ub)
        self.base_activation = copy.copy(base_activation)
        self.weight = copy.copy(weight)

    def set_hawkes_parameters_estimated(self, lamda_ub_estimated, W_estimated):
        r"""
        Set the estimated intensity upperbound, base activation and influence weight. 
        It is used in the visualization.  

        :type lamda_ub_estimated: 1D numpy array
        :param lamda_ub_estimated: :math:`\hat\bar{\lamda}`.
        :type W_estimated: number_of_dimensions * (number_of_dimensions * number_of_basis + 1) numpy array
        :param W_estimated: `W[:,0]` is the estimated base activation, `W[:,1:]` is the estimated influence weight
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(lamda_ub_estimated) != (self.number_of_dimensions,):
            raise ValueError('given estimated intensity upperbounds have incorrect shape')
        if np.shape(W_estimated) != (self.number_of_dimensions, self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('given estimated W have incorrect shape')
        self.lamda_ub_estimated = copy.copy(lamda_ub_estimated)
        self.base_activation_estimated = copy.copy(W_estimated[:,0])
        self.weight_estimated = copy.copy(W_estimated[:,1:])

    def intensity(self, t, target_dimension, timestamps_history, estimation = False):
        """
        Given the historical timestamps, evaluate the conditional intensity at t on the target dimension.
        It is used in the simulation and visualization. If `estimation` is False, the intensity function is using 
        the ground truth parameters; if `estimation` is True, the intensity function is using the estimated parameters. 

        :type t: float
        :param t: the target time
        :type target_dimension: int
        :param target_dimension: the target dimension
        :type timestamps_history: list
        :param timestamps_history: [[t_1,t_2,...,t_N_1],[t_1,t_2,...,t_N_2],...], the historical timestamps before t
        :type estimation: bool
        :param estimation: indicate to use whether the ground-truth or estimated parameters

        :rtype: float
        :return: the conditional intensity at t
        """
        # Raise ValueError if the given historical timestamps do not have the right shape
        if len(timestamps_history) != self.number_of_dimensions:
            raise ValueError('given historical timestamps have incorrect shape')
        if estimation == False:
            lamda_ub_target_dimension = self.lamda_ub[target_dimension]
            base_activation_target_dimension = self.base_activation[target_dimension]
            weight_target_dimension = self.weight[target_dimension]
        else:
            lamda_ub_target_dimension = self.lamda_ub_estimated[target_dimension]
            base_activation_target_dimension = self.base_activation_estimated[target_dimension]
            weight_target_dimension = self.weight_estimated[target_dimension]
        intensity = 0
        for n in range(self.number_of_dimensions):
            for i in range(len(timestamps_history[n])):
                if timestamps_history[n][i] >= t:
                    break
                elif t - timestamps_history[n][i] > self.T_phi: 
                    continue
                for b in range(self.number_of_basis):
                    intensity += weight_target_dimension[n][b] * beta.pdf(t - timestamps_history[n][i], a = self.beta_ab[b][0], b = self.beta_ab[b][1], loc = self.beta_ab[b][2], scale = self.T_phi)
        return lamda_ub_target_dimension * expit(base_activation_target_dimension + intensity)

    def simulation(self, T):
        r"""
        Simulate a sample path of the sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.

        :type T: float
        :param T: time at which the simulation ends.
        :rtype: list
        :return: the timestamps when events occur on each dimension.
        """
        t = 0
        points_hawkes = []
        for i in range(self.number_of_dimensions):
            points_hawkes.append([])
        intensity_sup = sum(self.lamda_ub)
        while(t < T):
            r = expon.rvs(scale = 1 / intensity_sup)
            t += r
            sum_intensity = sum(self.intensity(t,m,points_hawkes) for m in range(self.number_of_dimensions))
            assert sum_intensity <= intensity_sup, "intensity exceeds the upper bound"
            D = uniform.rvs(loc = 0,scale = 1)
            if D * intensity_sup <= sum_intensity:
                k = list(multinomial.rvs(1,[self.intensity(t,m,points_hawkes) / sum_intensity for m in range(self.number_of_dimensions)])).index(1)
                points_hawkes[k].append(t)
        if points_hawkes[k][-1] > T:
            del points_hawkes[k][-1]
        return points_hawkes



    "Inference: Gibbs Sampler"

    PG = PyPolyaGamma() # we use PyPolyaGamma to sample from PolyaGamma distribution
    # @staticmethod
    # def PG(b,c): 
    #     r"""
    #     Sampling from a Polya-Gamma distribution by truncation (default 2000 samples). It is not efficient. 
    #     """
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

    def Phi_n_g(self, points_hawkes, points_g):
        r"""
        Evaluate \Phi(t) on all observed points and grid nodes (Gaussian quadrature nodes). 

        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension
        :type points_g: list
        :param points_g: the timestamps of grid nodes or Gaussian quadrature nodes on [0,T]
        :rtype: number_of_dimensions*N_i*(number_of_dimensions*number_of_basis+1), num_g*(number_of_dimensions*number_of_basis+1)
        :return: list of \Phi(t_n), \Phi(t_g)
        """
        N = np.array([len(points_hawkes[d]) for d in range(self.number_of_dimensions)])
        num_g = len(points_g)
        Phi_n = [np.zeros((N[d],self.number_of_dimensions*self.number_of_basis+1)) for d in range(self.number_of_dimensions)]
        for d in range(self.number_of_dimensions):
            for n in range(N[d]):
                Phi_n[d][n] = self.Phi_t(points_hawkes[d][n],points_hawkes)
        Phi_g = np.zeros((num_g, self.number_of_dimensions*self.number_of_basis+1))
        for g in range(num_g):
            Phi_g[g] = self.Phi_t(points_g[g],points_hawkes)
        return Phi_n, Phi_g

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

    def loglikelyhood_gibbs(self, W, lamda, Phi_n, Phi_g, points_hawkes, T):
        r"""
        Evaluate the log-likelihood for the given timestamps in the iterative update of Gibbs sampler
        
        :type W: number_of_dimensions * (number_of_dimensions*number_of_basis +1) numpy array
        :param W: the input weight which includes the base activation
        :type lamda: 1D numpy array
        :param lamda: the input intensity upperbound for each dimension
        :type Phi_n: list of 1D numpy arrays
        :param Phi_n: the cumulative influence \Phi on each observed timestamp
        :type Phi_g: numpy array (Q, number_of_dimensions * number_of_basis + 1)
        :param Phi_g: the cumulative influence \Phi on each grid node
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
        for i in range(self.number_of_dimensions):
            if len(Phi_n[i]) != len(points_hawkes[i]):
                raise ValueError('the dimension of Phi_n is incorrect')
        N_g=len(Phi_g)
        logl = 0
        for i in range(self.number_of_dimensions):
            N_i = len(points_hawkes[i])
            for n in range(N_i):
                logl += np.log(expit(W[i].dot(Phi_n[i][n])))+np.log(lamda[i])
            for n in range(N_g):
                logl -= lamda[i]*expit(W[i].dot(Phi_g[n]))*T/N_g
        return logl

    def Gibbs_inference(self, points_hawkes, points_hawkes_test, T, T_test, b, num_grid, num_grid_test, num_iter, initial_W = None):
        r"""
        Gibbs sampler is used to sample from the posterior of parameters:
        lamda_ub and weight (base_activation is included in the weight). Along Gibbs loops, we compute the log-likelihood 
        for the training and test data. 

        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type T: float
        :param T: time at which the training timestamps ends
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type num_grid: int
        :param num_grid: the number of grid on [0,T] to represent the intensity
        :type num_grid_test: int
        :param num_grid_test: the number of grid on [0,T_test] to represent the intensity
        :type num_iter: int
        :param num_iter: the number of Gibbs loops
        :type initial_W: numpy array
        :param initial_W: the initial value for W in the Gibbs loops

        :rtype: numpy array
        :return: the posterior samples of lamda_ub (lamda) and weight (W), the training (logl) and test log-likelihood (logl_test)
        along Gibbs loops. 
        """
        # number of points on each dimension 
        N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) 
        N_test = np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)])
        #initial W and lamda
        if initial_W is None:
            W = np.random.uniform(-1,1,size=(self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            W = copy.copy(initial_W)
        lamda = N / T
        beta = np.zeros((self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        w_n = [np.zeros(N[d]) for d in range(self.number_of_dimensions)]
        Phi_n, Phi_g = self.Phi_n_g(points_hawkes, np.linspace(0,T,num_grid))
        Phi_n_test, Phi_g_test = self.Phi_n_g(points_hawkes_test, np.linspace(0,T_test,num_grid_test))
        intensity_g = np.zeros(num_grid)
        t_m = [[] for d in range(self.number_of_dimensions)] # thinned points
        w_m = [[] for d in range(self.number_of_dimensions)]
        Phi_m = [[] for d in range(self.number_of_dimensions)] # precompute Phi(t) on the thinned points

        lamda_list=[]
        W_list=[]     
        logl=[]
        logl_test=[]

        for ite in range(num_iter):
            for d in range(self.number_of_dimensions):             
                # sample w_n
                for n in range(N[d]):    
                    w_n[d][n]=self.PG.pgdraw(1,W[d].dot(Phi_n[d][n]))
                
                # sample t_m and w_m
                for g in range(num_grid):
                    intensity_g[g]=lamda[d]*expit(-W[d].dot(Phi_g[g]))
                t_m[d] = self.inhomo_simulation(intensity_g,T)
                Phi_m[d]=np.array([self.Phi_t(t,points_hawkes) for t in t_m[d]])
                w_m[d]=np.array([self.PG.pgdraw(1,W[d].dot(Phi_m[d][m])) for m in range(len(t_m[d]))])
                
                # sample lamda
                lamda[d] = gamma(a=N[d]+len(t_m[d]),scale=1/T).rvs()
                
                # sample W
                v=np.array([0.5]*N[d]+[-0.5]*len(t_m[d]))
                Sigma_inv=np.diag(list(w_n[d])+list(w_m[d]))
                Phi=np.concatenate((Phi_n[d],Phi_m[d]))
                Sigma_W=np.linalg.inv((Phi.T).dot(Sigma_inv).dot(Phi)+np.diag(beta[d]/b/b))
                mean_W=Sigma_W.dot(Phi.T).dot(v)
                W[d]=multivariate_normal(mean=mean_W,cov=Sigma_W).rvs()
                # for numerical stability, we truncate W if it is too close to 0
                W[d][np.abs(W[d])<1e-200]=1e-200*np.sign(W[d][np.abs(W[d])<1e-200])

                # sample beta
                beta[d] = invgauss.rvs(b / np.abs(W[d]))
            lamda_list.append(lamda.copy())
            W_list.append(W.copy())
            
            # compute the loglikelihood
            logl.append(self.loglikelyhood_gibbs(W,lamda,Phi_n,Phi_g,points_hawkes,T))
            logl_test.append(self.loglikelyhood_gibbs(W,lamda,Phi_n_test,Phi_g_test,points_hawkes_test,T_test))
        return lamda_list,W_list,logl,logl_test

    "Inference: EM Algorithm"

    @staticmethod
    def gq_points_weights(a,b,Q):
        r"""
        Generate the Gaussian quadrature nodes and weights for the integral :math:`\int_a^b f(t) dt`

        :type a: float
        :param a: the lower end of the integral
        :type b: float
        :param b: the upper end of the integral
        :type Q: int
        :param Q: the number of Gaussian quadrature nodes (weights)
        :rtype: 1D numpy array, 1D numpy array
        :return: Gaussian quadrature nodes and the corresponding weights
        """
        p,w = legendre.leggauss(Q)
        c = np.array([0] * Q + [1])
        p_new = (a + b + (b - a) * p) / 2
        w_new = (b - a) / (legendre.legval(p, legendre.legder(c))**2*(1-p**2))
        return p_new,w_new

    def loglikelyhood_em_mf(self, W, lamda, Phi_n, Phi_gq, points_hawkes, w_gq):
        r"""
        Evaluate the log-likelihood for the given timestamps in the iterative update of EM algorithm and mean-field approximation
        
        :type W: numpy array
        :param W: the input weight which includes the base activation
        :type lamda: 1D numpy array
        :param lamda: the input intensity upperbound
        :type Phi_n: list of 1D numpy arrays
        :param Phi_n: the cumulative influence \Phi on each observed timestamp
        :type Phi_gq: numpy array (Q, number_of_dimensions * number_of_basis + 1)
        :param Phi_gq: the cumulative influence \Phi on each Gaussian quadrature node
        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension
        :type w_gq: 1D numpy array 
        :param w_gq: Gaussian quadrature weights
        :rtype: float
        :return: the log-likelihood for the given timestamps
        """
        # Raise ValueError if the given timestamps do not have the right shape
        if len(points_hawkes) != self.number_of_dimensions:
            raise ValueError('given timestamps have incorrect shape')
        if np.shape(Phi_gq) != (len(w_gq), self.number_of_dimensions * self.number_of_basis + 1):
            raise ValueError('the dimension of Phi_gq or w_gq is incorrect')
        for i in range(self.number_of_dimensions):
            if len(Phi_n[i]) != len(points_hawkes[i]):
                raise ValueError('the dimension of Phi_n is incorrect')
        logl = 0
        for i in range(self.number_of_dimensions):
            N_i = len(points_hawkes[i])
            logl += sum(np.log(expit(W[i].dot(Phi_n[i].T))))+np.log(lamda[i])*N_i-(expit(W[i].dot(Phi_gq.T))*lamda[i]).dot(w_gq)
        return logl

    def EM_inference(self, points_hawkes, points_hawkes_test, T, T_test, b, num_gq, num_gq_test, num_iter, initial_W = None): 
        r"""
        EM algorithm which is used to estimate the MAP of parameters: 
        lamda_ub and weight (base_activation is included in the weight). 
        
        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type T: float
        :param T: time at which the training timestamps ends
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type num_gq: int
        :param num_gq: the number of Gaussian quadrature nodes on [0,T]
        :type num_gq_test: int
        :param num_gq_test: the number of Gaussian quadrature nodes on [0,T_test]
        :type num_iter: int
        :param num_iter: the number of EM iterations
        :type initial_W: numpy array
        :param initial_W: the initial value for W in the EM iterations

        :rtype: numpy array
        :return: the MAP estimation of lamda_ub (lamda) and weight (W), the training (logl) and test log-likelihood (logl_test)
        along EM iterations. 
        """
        # number of points on each dimension 
        N = np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) 
        N_test = np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)])
        #initial W and lamda
        if initial_W is None:
            W = np.random.uniform(-1,1,size=(self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            W = copy.copy(initial_W)
        lamda = N / T
        logl = []
        logl_test = []
        E_beta = np.zeros((self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        E_w_n = [np.zeros(N[d]) for d in range(self.number_of_dimensions)] 
        p_gq, w_gq = self.gq_points_weights(0,T,num_gq) 
        p_gq_test, w_gq_test = self.gq_points_weights(0,T_test,num_gq_test) 

        Phi_n, Phi_gq = self.Phi_n_g(points_hawkes, p_gq)
        Phi_n_test, Phi_gq_test = self.Phi_n_g(points_hawkes_test, p_gq_test)

        H_n = [W[d].dot(Phi_n[d].T) for d in range(self.number_of_dimensions)]
        H_gq = W.dot(Phi_gq.T)
        int_intensity = np.zeros(self.number_of_dimensions)
        
        for ite in range(num_iter):
            for d in range(self.number_of_dimensions):
                # update H_n_d,E_w_n_d; H_gq_d,int_intensity_d; E_beta_d
                H_n[d] = W[d].dot(Phi_n[d].T)
                E_w_n[d] = 1/2/H_n[d]*np.tanh(H_n[d]/2)
                H_gq[d] = W[d].dot(Phi_gq.T)
                int_intensity[d] = lamda[d]*expit(-H_gq[d]).dot(w_gq)
                E_beta[d] = b / np.abs(W[d])
                
                # update lamda_d
                lamda[d]=(int_intensity[d]+N[d])/T

                # update W_d
                int_A=np.zeros((self.number_of_dimensions*self.number_of_basis+1,self.number_of_dimensions*self.number_of_basis+1))
                for n in range(N[d]):
                    int_A+=E_w_n[d][n]*np.outer(Phi_n[d][n],Phi_n[d][n])
                for m in range(num_gq):
                    int_A+=w_gq[m]*(lamda[d]/2/H_gq[d][m]*np.tanh(H_gq[d][m]/2)*expit(-H_gq[d][m])*np.outer(Phi_gq[m],Phi_gq[m]))
                int_B=np.zeros(self.number_of_dimensions*self.number_of_basis+1)
                for n in range(N[d]):
                    int_B+=0.5*Phi_n[d][n]
                for m in range(num_gq):
                    int_B+=-w_gq[m]/2*(lamda[d]*expit(-H_gq[d][m])*Phi_gq[m])
                W[d]=np.linalg.inv(int_A+np.diag(E_beta[d]/b/b)).dot(int_B)
                # for numerical stability, we truncate W if it is too close to 0
                W[d][np.abs(W[d])<1e-200]=1e-200*np.sign(W[d][np.abs(W[d])<1e-200])
            # compute the loglikelihood
            logl.append(self.loglikelyhood_em_mf(W,lamda,Phi_n,Phi_gq,points_hawkes,w_gq))
            logl_test.append(self.loglikelyhood_em_mf(W,lamda,Phi_n_test,Phi_gq_test,points_hawkes_test,w_gq_test))
        return lamda, W, logl, logl_test

    "Inference: Mean-Field Variational Inference"

    @staticmethod
    def a_c_predict_n(d,n,Phi,mean_W,cov_W):
        r"""
        Compute the a=E[h_i(t)], c=E[h_i(t)^2] of observed points

        :type d: int
        :param d: the target dimension
        :type n: int
        :param n: the target n-th timestamp on d-th dimension
        :type Phi: number_of_dimension*N_i*(number_of_dimension*number_of_basis+1) list
        :param Phi: the cumulative influence on each timestamp
        :type mean_W: number_of_dimension*(number_of_dimension*number_of_basis+1) numpy array
        :param mean_W: the mean of weights
        :type cov_W: number_of_dimension*(number_of_dimension*number_of_basis+1)*(number_of_dimension*number_of_basis+1) numpy array
        :param cov_W: the covariance of weights
        :rtype: float
        :return: c=E[h_i(t)^2]
        """
        a=Phi[d][n].dot(mean_W[d])
        c=np.sqrt(a**2+(Phi[d][n].T).dot(cov_W[d]).dot(Phi[d][n]))
        return c

    @staticmethod
    def a_c_predict_gq(d,n,Phi,mean_W,cov_W):
        r"""
        Compute the a=E[h_i(t)], c=E[h_i(t)^2] of Gaussian quadrature nodes

        :type d: int
        :param d: the target dimension
        :type n: int
        :param n: the target n-th timestamp on d-th dimension
        :type Phi: N_gq*(number_of_dimension*number_of_basis+1) list
        :param Phi: the cumulative influence on gaussian quadrature nodes
        :type mean_W: number_of_dimension*(number_of_dimension*number_of_basis+1) numpy array
        :param mean_W: the mean of weights
        :type cov_W: number_of_dimension*(number_of_dimension*number_of_basis+1)*(number_of_dimension*number_of_basis+1) numpy array
        :param cov_W: the covariance of weights
        :rtype: float, float
        :return: a=E[h_i(t)], c=E[h_i(t)^2]
        """
        a=Phi[n].dot(mean_W[d])
        c=np.sqrt(a**2+(Phi[n].T).dot(cov_W[d]).dot(Phi[n]))
        return a,c


    def MF_inference(self, points_hawkes, points_hawkes_test, T, T_test, b, num_gq, num_gq_test, num_iter, initial_W_mean = None):
        r"""
        Mean-field variational inference which is used to update the posterior of 
        lamda_ub and weight (base_activation is included in the weight). 
        
        :type points_hawkes: list
        :param points_hawkes: the training timestamps
        :type points_hawkes_test: list
        :param points_hawkes_test: the test timestamps
        :type T: float
        :param T: time at which the simulation ends.
        :type T_test: float
        :param T_test: time at which the test timestamps ends
        :type b: float
        :param b: the scale parameter of Laplace prior; the larger b is, the wider the Laplace prior is
        :type num_gq: int
        :param num_gq: the number of Gaussian quadrature nodes on [0,T]
        :type num_gq_test: int
        :param num_gq_test: the number of Gaussian quadrature nodes on [0,T_test]
        :type num_iter: int
        :param num_iter: the number of MF iterations
        :type initial_W_mean: numpy array
        :param initial_W_mean: the initial value for W_mean 

        :rtype: numpy array
        :return: the parameter of lamda_ub posterior, the parameter of weight posterior, the training (logl) and test (logl_test) by the mean. 
        """
        N=np.array([len(points_hawkes[i]) for i in range(self.number_of_dimensions)]) # of points on each dimension 
        N_test=np.array([len(points_hawkes_test[i]) for i in range(self.number_of_dimensions)]) # of points on each dimension 
        #initialization
        #q_lamda_i is a gamma distribution gamma(alpha=N_i+E(|Pi_i|),scale=1/T)
        alpha=1.5*N
        lamda_1=np.zeros(self.number_of_dimensions)
        E_beta = np.zeros((self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        #q_W_i is a gaussian distribution N(mean_w_i,cov_w_i)
        if initial_W_mean is None:
            mean_W = np.random.uniform(-1,1,size=(self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        else:
            mean_W = copy.copy(initial_W_mean)
        cov_W=np.zeros((self.number_of_dimensions,(self.number_of_dimensions*self.number_of_basis+1),(self.number_of_dimensions*self.number_of_basis+1)))
        for i in range(self.number_of_dimensions*self.number_of_basis+1):
            cov_W[:,i,i]=1
        tilde_W=np.zeros((self.number_of_dimensions,self.number_of_dimensions*self.number_of_basis+1))
        for i in range(self.number_of_dimensions):
            tilde_W[i]=np.sqrt(mean_W[i]**2+np.diag(cov_W[i]))   

        #precompute the relative variables
        c_n=[np.zeros(N[i]) for i in range(self.number_of_dimensions)] # parameter of wn, D*N_i 
        E_wn=[np.zeros(N[i]) for i in range(self.number_of_dimensions)]      
        p_gq, w_gq = self.gq_points_weights(0,T,num_gq) 
        a_gq=np.zeros((self.number_of_dimensions,num_gq))
        c_gq=np.zeros((self.number_of_dimensions,num_gq))
        p_gq_test, w_gq_test = self.gq_points_weights(0,T_test,num_gq_test)
        Phi_n, Phi_gq = self.Phi_n_g(points_hawkes, p_gq)
        Phi_n_test, Phi_gq_test = self.Phi_n_g(points_hawkes_test, p_gq_test)
        
        logl=[]
        logl_test=[]
        
        for ite in range(num_iter):
            for i in range(self.number_of_dimensions):
                #update parameters of q_w_n_i q_w_n_i=P_pg(wn|1,cn)
                for n in range(N[i]):
                    c_n[i][n]=self.a_c_predict_n(i,n,Phi_n,mean_W,cov_W)
                    E_wn[i][n]=1/2/c_n[i][n]*np.tanh(c_n[i][n]/2)
                
                #update parameters of q_Pi_i intensity=exp(E(log lamda))sigmoid(-c(x))exp((c(x)-a(x))/2)*P_pg(w|1,c(x))
                lamda_1[i]=np.exp(np.log(1/T)+psi(alpha[i]))

                #update parameters of q_beta_i=IG(b/\tilde{W_i},1)
                E_beta[i]=b/tilde_W[i]
                
                #update parameters of q_lamda_i q_lamda_i=gamma(alpha=N_i+E(|Pi_i|),scale=1/T)
                for n in range(num_gq):
                    a_gq[i][n],c_gq[i][n]=self.a_c_predict_gq(i,n,Phi_gq,mean_W,cov_W)
                int_intensity=0
                for n in range(num_gq):
                    int_intensity+=w_gq[n]*lamda_1[i]*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)
                alpha[i]=int_intensity+N[i]
                
                # update parameters of q_W_i  q_W_i=N(mean_W,cov_W)    
                int_A=np.zeros((self.number_of_dimensions*self.number_of_basis+1,self.number_of_dimensions*self.number_of_basis+1))
                for n in range(N[i]):
                    int_A+=E_wn[i][n]*np.outer(Phi_n[i][n],Phi_n[i][n])
                for n in range(num_gq):
                    int_A+=w_gq[n]*(lamda_1[i]/2/c_gq[i][n]*np.tanh(c_gq[i][n]/2)*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)*np.outer(Phi_gq[n],Phi_gq[n]))
                
                int_B=np.zeros(self.number_of_dimensions*self.number_of_basis+1)
                for n in range(N[i]):
                    int_B+=0.5*Phi_n[i][n]
                for n in range(num_gq):
                    int_B+=-w_gq[n]/2*(lamda_1[i]*expit(-c_gq[i][n])*np.exp((c_gq[i][n]-a_gq[i][n])/2)*Phi_gq[n])
                
                cov_W[i]=np.linalg.inv(int_A+np.diag(E_beta[i]/b/b))
                mean_W[i]=cov_W[i].dot(int_B)
                tilde_W[i]=np.sqrt(mean_W[i]**2+np.diag(cov_W[i]))
                # for numerical stability, we truncate tilde_W if it is too close to 0
                tilde_W[i][tilde_W[i]<1e-200]=1e-200
            logl.append(self.loglikelyhood_em_mf(mean_W,alpha/T,Phi_n,Phi_gq,points_hawkes,w_gq))
            logl_test.append(self.loglikelyhood_em_mf(mean_W,alpha/T,Phi_n_test,Phi_gq_test,points_hawkes_test,w_gq_test))
        return alpha, mean_W, cov_W, logl, logl_test



    'tool functions'
    def influence_function_estimated(self, i, j, t, gt = False): 
        r"""
        Evaluate the influence function based on the basis functions and the influence weight W.
        It is used to visualize the influence functions. If gt = False, it is using the estimated parameters;
        if gt = True, it is using the ground truth parameters. 
        
        :type i: int
        :param i: the target dimension. \phi_{ij}(t)
        :type j: int
        :param j: the source dimension. \phi_{ij}(t)
        :type t: float
        :param t: the target time. \phi_{ij}(t)
        :type gt: bool
        :param gt: indicate to use whether the ground-truth or estimated parameters

        :rtype: float
        :return: the influence function \phi_{ij}(t)
        """
        if gt == False:
            W_phi = self.weight_estimated.reshape(self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)
        else:
            W_phi = self.weight.reshape(self.number_of_dimensions, self.number_of_dimensions, self.number_of_basis)
        phi_t = np.array([beta.pdf(t, a = self.beta_ab[i][0], b = self.beta_ab[i][1], loc = self.beta_ab[i][2], scale = self.T_phi) for i in range(self.number_of_basis)])
        return W_phi[i][j].dot(phi_t)

    def heat_map(self, gt = False):
        r"""
        Evaluate the heatmap value based on the weight of the instance. 
        (It is assumed that the integral of all basis functions is 1). If gt = False, it is using the estimated parameters;
        if gt = True, it is using the ground truth parameters. 

        :type gt: bool
        :param gt: indicate to use whether the ground-truth or estimated parameters

        :rtype: numpy array
        :return: the estimated heatmap value (self.number_of_dimensions * self.number_of_dimensions)
        """
        phi_heat=np.zeros((self.number_of_dimensions,self.number_of_dimensions))
        if gt == False:
            for i in range(self.number_of_dimensions):
                phi_heat[:,i]=np.sum(self.weight_estimated[:,self.number_of_basis*i:self.number_of_basis*(i+1)],axis=1)
        else:
            for i in range(self.number_of_dimensions):
                phi_heat[:,i]=np.sum(self.weight[:,self.number_of_basis*i:self.number_of_basis*(i+1)],axis=1)
        return phi_heat


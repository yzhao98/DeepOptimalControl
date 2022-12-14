import torch
import numpy as np
from problems.problem import ControlProblem

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

class ProblemSetup(ControlProblem):
    def __init__(self):
        super().__init__()
        self.N_states = 6
        self.N_controls = 3
        self.tT = 20.

        self.Jinv = np.diag([1/2, 1/3, 1/4])
        self.B = np.array([[1., 1/20, 1/10],[1/15, 1., 1/10],[1/10, 1/15, 1.]])
        self.BJ = np.matmul(self.Jinv, self.B).T
        h1, h2, h3 = 1., 1., 1.
        self.H = np.array([[h1],[h2],[h3]])

        self.W1 = 1.
        self.W2 = 10.
        self.W3 = 0.5
        self.W4 = 1.
        self.W5 = 1.

        # Initial condition bounds
        self.X0_ub = np.pi / np.array([[3.],[3.],[3.],[4.],[4.],[4.]])
        self.X0_lb = - self.X0_ub
    
    def p_ref(self, t, p):
        # The desired trajectory for (x, y, z)
        p_d = np.array([np.zeros(t.shape),
                 np.zeros(t.shape),
                 np.zeros(t.shape)])
        
        return(np.reshape(p_d, np.shape(p)))
    
    def eta_ref(self, t, eta):
        # The desired trajectory for yaw angle
        eta_d = np.array([np.zeros(t.shape),
                  np.zeros(t.shape),
                  np.zeros(t.shape)])
        
        return(np.reshape(eta_d, np.shape(eta)))

    def U_star(self, X_aug):
        """ Control as a function of the costate. """
        Aw = X_aug[9:12,:]
        U = np.matmul(self.BJ, Aw) / -self.W3
        return U

    def make_U_NN(self, u_model, t, X):
        U = u_model(t, X)
        return U

    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.N_states]
            XT = X_aug_T[:self.N_states]
            AT = X_aug_T[self.N_states:2*self.N_states]
            vT = X_aug_T[2*self.N_states:]

            # Derivative of the terminal cost with respect to X(T)
            dFdXT = np.concatenate((self.W4 * XT[:3], self.W5 * XT[3:]))

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    def running_cost(self, X, U):
        v = X[:3]
        w = X[3:]
        return np.sum(self.W1/2. * np.sum(v**2, axis=0, keepdims=True) +
                      self.W2/2. * np.sum(w**2, axis=0, keepdims=True) +
                      self.W3/2. * np.sum(U**2, axis=0, keepdims=True),
                      axis=0, keepdims=True)

    def terminal_cost(self, X):
        v = X[:3]
        w = X[3:]
        return np.sum((self.W4/2.) * v**2 + (self.W5/2.) * w**2)

    def dynamics(self, t, X, U_fun):
        """ Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration. """
        U = U_fun([[t]], X.reshape((-1,1))).flatten()

        v = X[:3]
        w = X[3:6]

        sin_v = np.sin(v)
        cos_v = np.cos(v)

        sin_phi = sin_v[0]
        cos_phi = cos_v[0]

        sin_theta = sin_v[1]
        cos_theta = cos_v[1]
        tan_theta = np.tan(v[1])

        sin_psi = sin_v[2]
        cos_psi = cos_v[2]

        E = np.array([[1., sin_phi*tan_theta, cos_phi*tan_theta],
                      [0., cos_phi, -sin_phi],
                      [0., sin_phi/cos_theta, cos_phi/cos_theta]])
        S = np.array([[0., w[2], -w[1]],
                      [-w[2], 0., w[0]],
                      [w[1], -w[0], 0.]])
        R = np.array([[cos_theta*cos_psi, cos_theta*sin_psi, -sin_theta],
                      [sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                        sin_phi*sin_theta*sin_psi + cos_phi*cos_psi,
                        cos_theta*sin_phi],
                      [cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                        cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                        cos_theta*cos_phi]])

        dvdt = np.matmul(E, w)
        dwdt = np.matmul(S, np.matmul(R,self.H)).flatten() + np.matmul(self.B,U)
        dwdt = np.matmul(self.Jinv, dwdt)

        return np.concatenate((dvdt,dwdt))

    def aug_dynamics(self, t, X_aug):
        """ Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP. """
        # Control as a function of the costate
        U = self.U_star(X_aug)

        Nt = t.shape[0]
        zeros = np.zeros_like(t)
        long_zeros = np.zeros(3*Nt)
        ones = np.ones_like(t)

        v = X_aug[:3]
        w = X_aug[3:6]

        # Costate
        Av = X_aug[6:9]
        Aw = X_aug[9:12]

        sin_v = np.sin(v)
        cos_v = np.cos(v)

        sin_phi = sin_v[0]
        cos_phi = cos_v[0]

        sin_theta = sin_v[1]
        cos_theta = cos_v[1]
        tan_theta = np.tan(v[1])

        sin_psi = sin_v[2]
        cos_psi = cos_v[2]

        # State dynamics
        # These are matrices of size (3x(3*Nt)), where data is
        # organized in the form
        # E = [[E_11 E_12 E_13][E_21 E_22 E_23][E_31 E_32 E_33]],
        # where each component E_ij is actually a vector of length Nt evaluating
        # E_ij for the state at each of the Nt time points.
        E = np.vstack((np.concatenate((ones, sin_phi*tan_theta, cos_phi*tan_theta)),
                       np.concatenate((zeros, cos_phi,-sin_phi)),
                       np.concatenate((zeros, sin_phi/cos_theta, cos_phi/cos_theta))))
        S = np.vstack((np.concatenate((zeros, w[2], -w[1])),
                       np.concatenate((-w[2], zeros, w[0])),
                       np.concatenate((w[1], -w[0], zeros))))
        R = np.vstack((np.concatenate((cos_theta*cos_psi,
                                       cos_theta*sin_psi,
                                       -sin_theta)),
                       np.concatenate((sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                                       sin_phi*sin_theta*sin_psi + cos_phi*cos_psi,
                                       cos_theta*sin_phi)),
                       np.concatenate((cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                       cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                                       cos_theta*cos_phi))))

        # Reorganizes matrices into sample-contiguous blocks,
        # E_T = [E^1 E^2 ... E^Nt],
        # where E^i is E for the ith sample. By doing np.sum(E_T*X, axis=0), where
        # X = [x^1 x^1 x^1 x^2 x^2 x^2 ... ],
        # where x^i is the ith sample of x, we get matmul(E^T,X), for every sample x^i.
        idx = np.array([0,Nt,2*Nt])
        idx = np.concatenate(tuple(idx + k for k in range(Nt))).tolist()

        E_T = E[:,idx]
        S_T = S[:,idx]

        # Most matrices need to be organized
        # E = [(E^T)^1 (E^T)^2 ... (E^T)^Nt],
        # where (E^T)^i is E^T for the ith sample (each block is transposed).
        # By doing np.sum(E*X, axis=0), we get matmul(E,X), for every sample x^i.
        E = E.T.reshape(E.shape)
        S = S.T.reshape(S.shape)
        R = R.T.reshape(R.shape)

        # Makes large matrices of copies of w and v:
        # W = [w^1 w^1 w^1 w^2 w^2 w^2 ... ],
        # where w^i is the ith sample of w. See above.
        W = np.repeat(w, 3, axis=1)
        AV = np.repeat(Av, 3, axis=1)
        # Copies H to the appropriate size
        H = np.repeat(self.H, 3*Nt, axis=1)

        RH = np.sum(R*H, axis=0).reshape((3,Nt), order='F')
        RH = np.repeat(RH, 3, axis=1)

        # Matrix-vector multiplication for all samples
        dvdt = np.sum(E*W, axis=0).reshape((3,Nt), order='F')
        dwdt = np.sum(S*RH, axis=0).reshape((3,Nt), order='F') + np.matmul(self.B, U)
        dwdt = np.matmul(self.Jinv, dwdt)

        # Costate dynamics
        DR1 = np.vstack((long_zeros,
                         np.concatenate((cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                         cos_phi*sin_theta*sin_psi - sin_phi*cos_psi,
                                         cos_phi*cos_theta)),
                         np.concatenate((-sin_phi*sin_theta*cos_psi + cos_phi*sin_psi,
                                         -sin_phi*sin_theta*sin_psi - cos_phi*cos_psi,
                                         -sin_phi*cos_theta))))
        DR2 = np.vstack((np.concatenate((-cos_psi*sin_theta,
                                         -sin_psi*sin_theta,
                                         -cos_theta)),
                         np.concatenate((sin_phi*cos_theta*cos_psi,
                                         sin_phi*cos_theta*sin_psi,
                                         -sin_phi*sin_theta)),
                         np.concatenate((cos_phi*cos_theta*cos_psi,
                                         cos_phi*cos_theta*sin_psi,
                                         -cos_phi*sin_theta))))
        DR3 = np.vstack((np.concatenate((-cos_theta*sin_psi,
                                         cos_theta*cos_psi,
                                         zeros)),
                         np.concatenate((-sin_phi*sin_theta*sin_psi - cos_phi*cos_psi,
                                         sin_phi*sin_theta*cos_psi - cos_phi*sin_psi,
                                         zeros)),
                         np.concatenate((-cos_phi*sin_theta*sin_psi + sin_phi*cos_psi,
                                         cos_phi*sin_theta*cos_psi + sin_phi*sin_psi,
                                         zeros))))
        DE1 = np.vstack((np.concatenate((zeros, cos_phi*tan_theta, -sin_phi*tan_theta)),
                         np.concatenate((zeros, -sin_phi, -cos_phi)),
                         np.concatenate((zeros, cos_phi/cos_theta, -sin_phi/cos_theta))))
        DE2 = np.vstack((np.concatenate((zeros,
                                         sin_phi/cos_theta**2,
                                         cos_phi/cos_theta**2)),
                         long_zeros,
                         np.concatenate((zeros,
                                         sin_phi*tan_theta/cos_theta,
                                         cos_phi*tan_theta/cos_theta))))
        DS1 = np.vstack((long_zeros,
                         np.concatenate((zeros, zeros, ones)),
                         np.concatenate((zeros, -ones, zeros))))
        DS2 = np.vstack((np.concatenate((zeros, zeros, -ones)),
                         long_zeros,
                         np.concatenate((ones, zeros, zeros))))
        DS3 = np.vstack((np.concatenate((zeros, ones, zeros)),
                         np.concatenate((-ones, zeros, zeros)),
                         long_zeros))

        # Reorganizes matrices into sample-contiguous blocks
        DR1 = DR1.T.reshape(DR1.shape)
        DR2 = DR2.T.reshape(DR2.shape)
        DR3 = DR3.T.reshape(DR3.shape)
        DE1 = DE1.T.reshape(DE1.shape)
        DE2 = DE2.T.reshape(DE2.shape)
        DS1 = DS1.T.reshape(DS1.shape)
        DS2 = DS2.T.reshape(DS2.shape)
        DS3 = DS3.T.reshape(DS3.shape)

        AwJinvT = np.matmul(self.Jinv.T, Aw)
        AwJinv = np.repeat(AwJinvT, 3, axis=1)
        # Product Aw^T J^-1 S = (S^T J^-1^T Aw)^T
        AwJinvS = np.sum(S_T*AwJinv, axis=0).reshape((3,Nt), order='F')

        dAvdt = - self.W1*v

        # Equivalent to taking inner products
        dAvdt[0] -= np.sum(Av * np.sum(DE1*W, axis=0).reshape((3,Nt), order='F'), axis=0)
        dAvdt[1] -= np.sum(Av * np.sum(DE2*W, axis=0).reshape((3,Nt), order='F'), axis=0)

        dAvdt[0] -= np.sum(AwJinvS * np.sum(DR1*H, axis=0).reshape((3,Nt), order='F'), axis=0)
        dAvdt[1] -= np.sum(AwJinvS * np.sum(DR2*H, axis=0).reshape((3,Nt), order='F'), axis=0)
        dAvdt[2] -= np.sum(AwJinvS * np.sum(DR3*H, axis=0).reshape((3,Nt), order='F'), axis=0)

        dAwdt = - self.W2*w - np.sum(E_T*AV, axis=0).reshape((3,Nt), order='F')

        dAwdt[0] -= np.sum(AwJinvT * np.sum(DS1*RH, axis=0).reshape((3,Nt), order='F'), axis=0)
        dAwdt[1] -= np.sum(AwJinvT * np.sum(DS2*RH, axis=0).reshape((3,Nt), order='F'), axis=0)
        dAwdt[2] -= np.sum(AwJinvT * np.sum(DS3*RH, axis=0).reshape((3,Nt), order='F'), axis=0)

        L = self.running_cost(X_aug[:6], U)

        return np.vstack((dvdt, dwdt, dAvdt, dAwdt, -L))
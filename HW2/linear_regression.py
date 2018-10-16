import numpy as np


class LinReg:

    def __init__(self, x, y, basis='poly', nabla=None):
        self.x = x
        self.y = y
        self.basis = basis
        self.nabla = nabla

        self.phi = None
        self.w = None
        self.L_hist = []

        self.calc_phi()

    def calc_phi(self):
        '''
        calculate phi vector based on indicated basis type
        phi vector depends on scalar input data x and length of phi is M
        :return:
        '''
        basis = self.basis
        phi = None
        if basis == 'poly':
            phi = lambda x, M: np.array([x**i for i in range(M+1)])
        elif basis == 'cosine0':  # first index is 0
            phi = lambda x, M: np.array([np.cos(i*np.pi*x) for i in range(1, M+1)])
        elif basis == 'cosine1':  # first index is 1
            phi = lambda x, M: np.array([np.cos(i*np.pi*x) for i in range(M+1)])
        else:
            raise ValueError('Basis', basis, 'unrecognized')

        self.phi = phi

    def calc_w_plot(self, M):
        x, w, phi = self.x, self.w, self.phi
        n = 100  # number of points to plot
        x_axis = np.linspace(min(x), max(x), n, endpoint=True)
        y_axis = np.array([w.T@phi(xi, M) for xi in x_axis])
        return x_axis, y_axis

    def calc_opt_w(self, M):
        '''
        calcualte optimal closed form length M vector w
        :param M: length of w, phi
        :return:
        '''
        x, y, phi = self.x, self.y, self.phi
        Phi = np.array([phi(xi, M) for xi in x])  # design matrix
        w = np.linalg.inv(Phi.T@Phi)@Phi.T@y
        self.w = w
        return w

    def calc_L(self, M):
        x, y, w, phi = self.x, self.y, self.w, self.phi
        L = .5 * sum([(w.T@phi(x[i], M) - y[i])**2 for i in range(len(x))])
        self.L_hist.append(L)
        return L

    def calc_gd_w(self, M, delta_thresh, b=0, w_init=None):
        '''
        :param M:
        :param L_thresh:
        :param b: batch size
        :param w_init: how to initialize w
        :return:
        '''
        x, y, phi, nabla = self.x, self.y, self.phi, self.nabla
        N = len(x)  # number of data points
        w = None  # weight vector
        B = None  # set of data indices to use for a step
        b = N if b == 0 else b  # b = 0 means default to size of data array
        assert 0 <= b <= N
        assert self.nabla is not None
        # initialize w
        if w_init is None:
            w_init = 'norm1'
        if w_init[:4] == 'norm':
            var = int(w_init[4:])
            w = np.random.normal(scale=var, size=(len(phi(x[0], M)),))
        elif w_init == 'zero':
            w = np.zeros((len(phi(x[0], M)),))

        w0 = w
        self.w = w
        self.calc_L(M)  # get initial loss value

        part = np.arange(N)
        while True:
            # partition into batches
            np.random.shuffle(part)
            Bs = [part[i*b:(i+1)*b] for i in range(N//b)]  # partition into batches
            # perform generalized batch gradient descent
            for B in Bs:
                xB = np.array([x[i] for i in B])
                yB = np.array([y[i] for i in B])
                Phi = np.array([phi(xi, M) for xi in xB])
                dLdw = 1/b * Phi.T@(Phi@w - yB)
                w0 = w
                w = w - nabla * dLdw
                self.w = w
                self.calc_L(M)
                if max(abs(w - w0)) < delta_thresh:
                    return w

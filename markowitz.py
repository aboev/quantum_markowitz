import numpy as np

class Markowitz:
    n = 0
    G = np.zeros((n, n))
    h = np.zeros(n)
    qmatrix = np.zeros((n, n))
    jmatrix = np.zeros((n, n))
    cfactor = 0
    hvector = np.zeros(n)
    qvector = np.zeros(n)
    theta = [0.3, 0.3, 0.3]
    budget = 100.0   
 
    def model(self, file_prices, file_averages, file_covariance):
        f = open(file_prices, 'r')
        price = [float(x) for x in f.readline().split(',')]
        f.close()

        f = open(file_averages, 'r')
        avg = [float(x) for x in f.readline().split(',')]
        f.close()

        n = len(avg)
        cov = np.zeros((n, n))
        f = open(file_covariance, 'r')
        i = 0
        for line in f:
            values = line.split(',')
            for j in range(len(values)):
                cov[i, j] = float(values[j])
            i += 1
        f.close()

        qmatrix = np.zeros((n, n))
        qvector = np.zeros((n))
        for row in range(n):
            for col in range(row + 1, n, 1):
                qmatrix[row][col] = 2.0 * (self.theta[1] * cov[row][col] + self.theta[2] * price[row] * price[col])

        for row in range(n):
            qvector[row] = -1.0 * self.theta[0] * avg[row] - self.theta[2] * 2.0 * self.budget * price[row] + self.theta[1] * cov[row][row] + self.theta[2] * price[row] * price[row]

        cfactor = self.theta[2] * self.budget * self.budget

        return (qmatrix, qvector, cfactor)

    def graph(self, qmatrix, qvector, cfactor):
        n = qmatrix.shape[0]
        jmatrix = 0.25 * qmatrix
        hvector = np.zeros(n)

        jmatrixsum = 0.0
        qvectorsum = 0.0

        for row in range(n):
            rowsum = 0.0
            for col in range(row + 1, n, 1):
                rowsum += jmatrix[row][col]
            hvector[row] = 0.5 * qvector[row] + rowsum
            jmatrixsum += rowsum
            qvectorsum += qvector[row]

        gfactor = cfactor + jmatrixsum + 0.5 * qvectorsum

        return (jmatrix, hvector, gfactor)

    def optval(self, spins):
        optval = 0.0
        for row in range(self.n):
            for col in range(row + 1, self.n , 1):
                optval += spins[row] * self.qmatrix[row][col] * spins[col]
            optval += spins[row] * self.qvector[row]
        optval += self.cfactor
        return optval

    def energy(self, spins):
        return np.dot(np.dot(spins, self.jmatrix), spins) + np.sum(spins * self.hvector)

    def __init__(self, file_prices, file_averages, file_covariance, theta = [0.3, 0.3, 0.3], budget = 100.0):
        self.theta = theta
        self.budget = budget
        (qmatrix, qvector, cfactor) = self.model(file_prices, file_averages, file_covariance)
        (jmatrix, hvector, gfactor) = self.graph(qmatrix, qvector, cfactor)
        self.qmatrix = qmatrix
        self.jmatrix = jmatrix
        self.cfactor = cfactor
        self.n = jmatrix.shape[0]
        self.G = (jmatrix + jmatrix.T).T
        self.h = np.expand_dims(hvector, axis=0).T
        self.hvector = hvector
        self.qvector = qvector

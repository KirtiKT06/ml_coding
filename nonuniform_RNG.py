import numpy as np
import matplotlib.pyplot as plt
class ProbabilityDistribution:
    def pdf(self, x):
        raise NotImplementedError("pdf(x) not implemented")

    def inverse_cdf(self, u):
        raise NotImplementedError("inverse_cdf(u) not implemented")

    def support(self):
        raise NotImplementedError("support() not implemented")

    def theoretical_moments(self):
        raise NotImplementedError("theoretical_moments() not implemented")

class LinearPDF(ProbabilityDistribution):
    def __init__(self):
        self.name = "Linear PDF f(x)=2x on [0,1]"

    def pdf(self, x):
        if 0 <= x <= 1:
            return 2 * x
        return 0.0

    def inverse_cdf(self, u):
        return np.sqrt(u)

    def support(self):
        return (0.0, 1.0)

    def theoretical_moments(self):
        mean = 2 / 3
        variance = 1 / 18
        return mean, variance

class TriangularPDF(ProbabilityDistribution):
    def __init__(self):
        self.name = "Triangular PDF on [-1,1]"

    def pdf(self, x):
        if -1 <= x < 0:
            return 1 + x
        elif 0 <= x < 1:
            return 1-x
        else: 
            return 0.0

    def inverse_cdf(self, u):
        if u < 0.5:
            return (-1 + np.sqrt(2*u))
        else:
            return (1 - np.sqrt(2*(1 - u)))

    def support(self):
        return (-1.0, 1.0)

    def theoretical_moments(self):
        mean = 0
        variance = 1 / 6
        return mean, variance

class InversionSampler:
    def __init__(self, distribution, rng=None):
        self.distribution = distribution
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, n):
        samples = []
        for _ in range(n):
            u = self.rng.uniform(0.0, 1.0)
            x = self.distribution.inverse_cdf(u)
            samples.append(x)
        return np.array(samples)

class RejectionSampler:
    def __init__(self, distribution, M, rng=None):
        self.distribution = distribution
        self.M = M
        self.rng = rng if rng is not None else np.random.default_rng()
        self.trials = 0
        self.accepted = 0

    def sample(self, n):
        samples = []
        xmin, xmax = self.distribution.support()

        while len(samples) < n:
            self.trials += 1
            x = self.rng.uniform(xmin, xmax)
            u = self.rng.uniform(0.0, 1.0)

            # uniform proposal: g(x) = 1 / (xmax - xmin)
            g = 1.0 / (xmax - xmin)

            if u <= self.distribution.pdf(x) / (self.M * g):
                samples.append(x)
                self.accepted += 1

        return np.array(samples)

    def acceptance_rate(self):
        return self.accepted / self.trials

class Experiment:
    def __init__(self, distribution, sampler, n):
        self.distribution = distribution
        self.sampler = sampler
        self.n = n
        self.samples = None

    def run(self):
        self.samples = self.sampler.sample(self.n)

    def compute_statistics(self):
        mean_mc = np.mean(self.samples)
        var_mc = np.var(self.samples)

        mean_th, var_th = self.distribution.theoretical_moments()

        return {
            "mean_mc": mean_mc,
            "var_mc": var_mc,
            "mean_th": mean_th,
            "var_th": var_th,
        }

    def report(self):
        stats = self.compute_statistics()

        print(f"Distribution: {self.distribution.name}")
        print(f"Number of samples: {self.n}")
        print(f"Monte Carlo mean: {stats['mean_mc']}")
        print(f"Theoretical mean: {stats['mean_th']}")
        print(f"Monte Carlo variance: {stats['var_mc']}")
        print(f"Theoretical variance: {stats['var_th']}")

        if hasattr(self.sampler, "acceptance_rate"):
            print(f"Acceptance rate: {self.sampler.acceptance_rate()}")

    def plot(self, bins=100):
        xmin, xmax = self.distribution.support()

        plt.hist(self.samples, bins=bins, density=True,
                 alpha=0.6, label="Estimated PDF")

        x = np.linspace(xmin, xmax, 1000)
        y = [self.distribution.pdf(xi) for xi in x]

        plt.plot(x, y, label="Analytical PDF", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.title(self.distribution.name)
        plt.legend()
        plt.show()

def main():
    n = 1_000_000

    # Choose distribution
    dist = TriangularPDF()
    # dist = LinearPDF()   # <- switch here if needed

    # Choose sampler
    sampler = InversionSampler(dist)
    # sampler = RejectionSampler(dist, M=2)

    # Run experiment
    exp = Experiment(dist, sampler, n)
    exp.run()
    exp.report()
    exp.plot()

if __name__=="__main__":
    main()
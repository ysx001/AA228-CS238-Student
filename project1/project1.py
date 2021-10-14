import sys

import pandas as pd
import networkx as nx
import numpy as np
from scipy.special import loggamma

class Variable():
    def __init__(self, name, r_i):
        self.name = name
        self.r_i = r_i
class K2Search():
    def __init__(self, ordering):
        self.ordering = ordering
    
    def fit(self, vars, D):
        G = nx.DiGraph()
        G.add_nodes_from(range(len(vars)))
        for (k, i) in enumerate(self.ordering[1:]):
            y = bayesian_score(vars, G, D)
            while True:
                y_best, j_best = -np.inf, 0
                for j in self.ordering[:k]:
                    if not G.has_edge(j, i):
                        G.add_edge(j, i)
                        y_prime = bayesian_score(vars, G, D)
                        if y_prime > y_best:
                            y_best, j_best = y_prime, j
                        G.remove_edge(j, i)
                if y_best > y:
                    y = y_best
                    G.add_edge(j_best, i)
                else:
                    break
        return G

class LocalDirectedGraphSearch():
    def __init__(self, k_max):
        self.k_max = k_max
    
    def rand_graph_neighbor(self, G):
        n = len(G)
        i = np.random.randint(low=0, high=n)
        j = (i + np.random.randint(low=1, high=n) - 1) % n
        G_prime = G.copy()
        if G.has_edge(i, j):
            G_prime.remove_edge(i, j)
        else:
            G_prime.add_edge(i, j)
        return G_prime
    
    def fit(self, vars, D):
        G =  nx.DiGraph() # initial graph
        G.add_nodes_from(range(len(vars)))
        y = bayesian_score(vars, G, D)
        for k in range(self.k_max):
            G_prime = self.rand_graph_neighbor(G)
            y_prime = -np.inf if has_cycle(G_prime) else bayesian_score(vars, G_prime, D)
            if y_prime > y:
                y, G = y_prime, G_prime
        return G

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile, method="local", n_iter=10):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    D = pd.read_csv(infile)
    vars = [Variable(name, D.max()[name]) for name in D.columns]

    G = nx.DiGraph()
    G.add_nodes_from(range(len(vars)))
    best_score, best_G = -np.inf, G
    ordering = np.arange(len(vars))
    for i in range(n_iter):
        new_ordering = np.copy(ordering)
        np.random.shuffle(new_ordering)
        if method == 'k2':
            k2_search = K2Search(ordering=new_ordering)
            G = k2_search.fit(vars, D)
        elif method == 'local':
            local_search = LocalDirectedGraphSearch(k_max=10)
            G = local_search.fit(vars, D)

        if not has_cycle(G):
            score = bayesian_score(vars, G, D)
            if score > best_score:
                best_score, best_G = score, G
            print(f"iter {i} score {score} best_score {best_score}")

    print(best_score)
    idx2names = {i: vars[i].name for i in range(len(vars))}
    write_gph(best_G, idx2names, outfile)

def has_cycle(G):
    # nx.find_cycle will throw exception if cycle is found
    try:
        nx.find_cycle(G)
        return True
    except nx.exception.NetworkXNoCycle as ex:
        return False

def bayesian_score_component(M, alpha):
    p = np.sum(loggamma(alpha + M))
    p -= np.sum(loggamma(alpha))
    p += np.sum(loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p

def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return np.sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))

def statistics(vars, G, D):
    n = len(vars)
    r = [var.r_i for var in vars]
    q = [np.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    M = [np.zeros((int(q[i]), int(r[i]))) for i in range(n)]
    for o in range(len(D.index)):
        for i in range(n):
            # NOTE: minus 1 because python is zero-indexed
            k = D.loc[o, vars[i].name] - 1
            parents = [p for p in G.predecessors(i)]
            j = 0
            if len(parents) != 0:
                r_p = [r[p] for p in parents]
                o_p = [D.loc[o, vars[p].name] - 1 for p in parents]
                j = np.ravel_multi_index(o_p, r_p)
            M[i][j, k] += 1
    return M

def prior(vars, G):
    n = len(vars)
    r = [var.r_i for var in vars]
    q = [np.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    return [np.ones((int(q[i]), int(r[i]))) for i in range(n)]

def main():
    if len(sys.argv) != 5:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph method n_iter")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    method = sys.argv[3]
    n_iter = int(sys.argv[4])
    compute(inputfilename, outputfilename, method, n_iter)


if __name__ == '__main__':
    main()

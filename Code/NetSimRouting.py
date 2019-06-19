#!/usr/bin/python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import filecmp
import os
import sys
import random
import argparse
import time
import matplotlib.pyplot as plt


# Creates the SP lengths matrix
def calcSP(G):
    netSize = G.number_of_nodes()
    SP = np.zeros((netSize, netSize))
    for i in range(netSize):
        for j in range(netSize):
            try:
                SP[i][j] = nx.shortest_path_length(G, source=i, target=j)
            except nx.exception.NetworkXNoPath:
                SP[i][j] = -1
    return SP


def isValid(G, R, mapP):
    netSize = G.number_of_nodes()
    for src in range(netSize):
        for dst in range(netSize):
            k = SPlenMatrix[src][dst] + args.min_length + 1
            aux = src
            path = []
            while aux != dst:
                path.append(aux)
                nextPort = int(R[aux][dst])
                nextNode = int(mapP[aux][nextPort])
                if not G.has_edge(aux, nextNode):
                    k = 0
                aux = nextNode
                k -= 1
                if (k < 1):
                    print(aux, dst, nextPort)
                    print("Expected path", paths[str(src) + '-' + str(dst)])
                    print("k", k)
                    print("path", path)
                    #print("Next node in port edge", mapP[aux][nextPort])
                    return False
    return True


def createMapping(G):
    mapP = dict()
    edges_data = G.edges.data()
    for edge in edges_data:
        src = edge[0]
        dst = edge[1]
        port = edge[2]['port']
        if mapP.get(src) is None:
            mapP[src] = dict()
        mapP[src][port] = dst
    return mapP


def generateAPPaths(G):
    netSize = G.number_of_nodes()
    paths = []
    for src in range(netSize):
        for dst in range(netSize):
            if src == dst:
                continue
            limit = SPlenMatrix[src][dst] + args.max_length
            path = nx.all_simple_paths(G, src, dst, limit)
            pathHash[str(src) + '-' + str(dst)] = path
            paths.append(path)
    return paths


def calculateWeights(paths):
    paths_weights = []
    # Calculate the weights of the paths
    for path in paths:
        path_weight = 0
        for link in path:
            path_weight += weights[link]
        paths_weights.append((path_weight, path))

    return paths_weights


def chosePath(paths, diff):

    if diff == "normal":
        return random.choice(paths)

    paths_weigths = calculateWeights(paths)
    if diff == "easy":
        min_path = min(paths_weigths, key=lambda t: t[0])
        p = [x[1] for x in paths_weigths if x[0] <= (min_path[0] * 1.1)]
    elif diff == "hard":
        max_path = max(paths_weigths, key=lambda t: t[0])
        p = [x[1] for x in paths_weigths if x[0] >= (max_path[0] * 0.90)]
    return random.choice(p)


def generateAPRouting(G):
    netSize = G.number_of_nodes()
    R = np.zeros((netSize, netSize)) - 1
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='k')
    w = []
    path_weigth = []
    for src in range(netSize):
        for dst in range(netSize):
            if src == dst:
                continue
            try:
                min_limit = SPlenMatrix[src][dst] + args.min_length
                for _ in range(10):
                    rp = next(pathHash[str(src) + '-' + str(dst)])
                    if len(rp) > int(min_limit):
                        paths.setdefault(str(src) + '-' + str(dst), []).append(rp)
            except StopIteration:
                pass

            paths_pool = paths[str(src) + '-' + str(dst)]

            real_path = chosePath(paths_pool, args.difficulty)
            for link in real_path:
                weights[link] += 1
                w.append(weights[link])
            # Set the port for the node to the next node
            # This is going to be changed later on to support source aware routing
            R[src][dst] = G[src][real_path[1]][0]['port']

    #  Uncomment the part below to plot the routings
    #         path_edges = list(zip(real_path, real_path[1:]))
    #         if src > 45:
    #             nx.draw_networkx_nodes(G, pos, nodelist=real_path, node_color='r')
    #             for link in real_path:
    #                 path_weigth.append(weights[link] * 0.03)
    #             nx.draw_networkx_edges(
    #                 G, pos, edgelist=path_edges, edge_color='r', width=path_weigth)
    # print(np.average(path_weigth))
    # print(len(np.nonzero(path_weigth)))
    # plt.axis('equal')
    # plt.show()
    # print(real_path)
    return R


def saveRouting(R, name, output_dir):
    newFile = "%s/%s.txt" % (output_dir, name)

    np.savetxt(newFile, R, fmt='%1i', delimiter=',', newline=',\n')
    directory = os.fsencode(output_dir)
    for file in os.listdir(directory):
        filename = "%s/%s" % (output_dir, os.fsdecode(file))
        if (filename == newFile):
            continue
        if filecmp.cmp(filename, newFile):
            if filename != newFile:
                print('Repeated routing: ' + filename + ' ' + newFile)
                os.remove(newFile)
                return False

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Topology name")
    parser.add_argument("-n", help="Number of routings ",
                        type=int, default=0)
    parser.add_argument("--min_length", help="Minimum length for the paths added to the shortest path",
                        type=int, default=0)
    parser.add_argument("--difficulty", help="Sets the theoretical difficulty",
                        type=str, default="normal", choices=["easy", "normal", "hard"])
    args = parser.parse_args()

    base_dir = "./simulator/params/" + args.name
    output_dir = "%s/routing" % base_dir
    if (not os.path.isdir(base_dir)):
        print("Directory not exists: " + base_dir)
        sys.exit(0)

    if (os.path.isdir(output_dir)):
        print("Ouput directory already exists. Remove it before start")
        # sys.exit(0)
    # os.mkdir(output_dir)

    if args.max_length == 0 and args.min_length > 0:
        args.max_length = args.min_length + 1

    print("Routing Generator")
    print("Creating {} routings".format(args.difficulty))

    G0 = nx.read_gml(base_dir + "/graph_attr.txt", destringizer=int)
    SPlenMatrix = calcSP(G0)
    netSize = G0.number_of_nodes()

    # Shortest Paths
    repe = 0
    it = 0
    max_retries = 10000

    t = 0
    pathHash = dict()  # Stores generator paths for a pair source - destination
    paths = dict()  # Stores all paths for a spoir source - destination
    weights = [0] * netSize  # Stores the weights of each link

    P = generateAPPaths(G0)
    mapP = createMapping(G0)
    print(mapP)
    while it < args.n and max_retries != 0:
        G = G0
        netSize = G.number_of_nodes()
        R = np.zeros((netSize, netSize)) - 1

        start = time.time()
        R = generateAPRouting(G)
        print(R)
        valid = isValid(G, R, mapP)
        print("Is routing Valid?", valid)
        if not valid:
            print("Routing has loops")
            print("Iteration", it)
            max_retries -= 1
            continue
        end = time.time()
        print("Total time", end - start)
        isNew = saveRouting(R, 'AP_k_' + str(it), output_dir)
        if not isNew:
            max_retries -= 1
        else:
            it += 1
        if it % 100 == 99:
            print('Iteration ' + str(it + 1))

    if (max_retries == 0):
        print("Not managed to create %d different routings" % (args.n))
    print()
    print("Metrics")
    print("*" * 20)
    print()
    print("Total time", t)
    print("Average time", t / args.n)
    print("Repeated routings", 500 - max_retries)

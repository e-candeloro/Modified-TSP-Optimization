# imports
import random
import time
from itertools import combinations
import matplotlib.pyplot as plt
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


random.seed(10000)

# SPECIAL CONSTANTS
QUIET = 0
FILE_PATH_POINTS = "test_graph_points.csv"
FILE_PATH_COSTS = "test_graph_costs.csv"
FILE_PATH_COMPARISON_DATASET = "dataset_1.csv"
DATA_COMPARISON = 0  # if 1, switch to load the comparison dataset

N = 10  # number of vertices if we use a random graph
RANDOM = 0  # if random is 0 we use a csv dataset, otherwise we create a random directed graph


# UTILITY FUNCTIONS

# defines random points in the 2D cartesian plane: purely for illustration purposes
def random_points(n):
    points = {}
    for i in range(n):
        points[i] = (random.randint(0, 100), random.randint(0, 100))
    return(points)

# saves the directed graph in two csv files


def save_graph_csv(points, v_profits, a_costs, graphname="graph"):
    # saving points and profits
    df_points = pd.DataFrame.from_dict(
        points, orient='index', columns=["x_cor", "y_cor"])
    df_profits = pd.DataFrame.from_dict(
        v_profits, orient='index', columns=['vertex_profits'])
    df_vertexs = pd.concat([df_points, df_profits], axis=1)
    df_vertexs.to_csv(f"{graphname}_points_profits.csv", index=True)

    # saving arch costs
    df_costs = pd.DataFrame.from_dict(
        a_costs, orient='index', columns=["arch_costs"])
    df_costs.to_csv(f"{graphname}_arch_costs.csv", index=True)

    print("Saved the following data:\n")
    print(df_vertexs)
    print(df_costs)

# loads a directed graph from two csv files


def load_graph_csv(points_file="", arch_file=""):
    # read the csv files
    if not(len(points_file) > 0 and len(arch_file) > 0):
        print("need to specify filepaths")
        return None, None, None
    df_file_points = pd.read_csv(
        points_file, index_col=0)
    df_file_costs = pd.read_csv(arch_file, index_col=0)

    # create the vertex and vertex profit dictionaries
    loaded_points = df_file_points.to_dict(orient='index')
    points = {}
    profits = {}
    for i, _ in enumerate(loaded_points):
        points[i] = (loaded_points[i]["x_cor"], loaded_points[i]["y_cor"])
        profits[i] = (loaded_points[i]["vertex_profits"])

    loaded_costs = df_file_costs.to_dict()
    costs = {}
    for key, value in enumerate(loaded_costs["arch_costs"]):
        costs[tuple(eval(value))] = loaded_costs["arch_costs"][value]

    print("loaded the following data:\n")
    print(f"Points: {points}\nProfits: {profits}\nCosts: {costs}")

    return points, profits, costs

# load dataset from a csv to compare result with other students


def load_graph_dataset(filename=''):
    assert len(filename) > 0
    df_data = pd.read_csv(filename, index_col=0)
    df_data = df_data.fillna(0)
    profits = df_data['Profits'].to_dict()

    arch_cost = {}
    for i in range(len(df_data)):
        for j in range(len(df_data.iloc[:, 0:-1])):
            data = df_data.iloc[i, j]
            if data != 0:
                arch_cost[i, j] = df_data.iloc[i, j]

    points = random_points(len(df_data))

    return points, profits, arch_cost

# plot the directed graph with some optional additional info


def plot_dgraph(points, profits, archs, title="Graph", figsize=(12, 12), save_fig=None, show_vars=None, color='b'):
    """
    Plot a directed graph

    :param points: list of points.
    :param profits: list of point profits
    :param archs: list of selected archs
    :param title: title of the figure.
    :param figsize: width and height of the figure
    :param save_fig: if provided, path to file in which the figure will be save.
    :param show_vars: if provided, shows the cost and profits in the graph.
    :param color: color of graph, default is b = blue
    :return: None
    """

    plt.figure(figsize=figsize)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title, fontsize=15)
    plt.grid()
    x = [points[i][0] for i in points.keys()]
    y = [points[i][1] for i in points.keys()]
    profit = [profits[i]*10 for i in profits.keys()]
    if len(x) != len(profit):
        plt.scatter(x, y)
    else:
        plt.scatter(x, y, s=profit)

    n = len(points)

    # Add label to points and profits
    for i, label in enumerate(points):
        plt.annotate('{}'.format(i), (x[i]+0.2, y[i]+0.2), size=23)
        if show_vars:
            plt.annotate(f"p:{profits[i]}",
                         (x[i], y[i]+1), size=20, color='red')

    for (i, j) in archs.keys():
        plt.arrow(x[i], y[i], x[j]-x[i], y[j]-y[i], fc=color,
                  ec=color, head_width=1, head_length=1, length_includes_head=True)
        if show_vars:
            if i > j:
                plt.annotate(f"c:{archs[i,j]}", ((points[i][0]+points[j][0])/2 + 1,
                             (points[i][1]+points[j][1])/2 + 2), size=15, color='black')
            else:
                plt.annotate(f"c:{archs[i,j]}", ((points[i][0]+points[j][0])/2 - 1,
                                                 (points[i][1]+points[j][1])/2 - 2), size=15, color='black')

    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()

# Plots the tour solution with arrows to show the arches direction


def plot_tour(points, tour, title="Tour", figsize=(12, 12), save_fig=None):
    """
    Plot a tour.

    :param points: list of points.
    :param tour: list of indexes that describes in which order the points are
                 visited.
    :param title: title of the figure.
    :param figsize: width and height of the figure
    :param save_fig: if provided, path to file in which the figure will be save.
    :return: None
    """
    x = [points[i][0] for i in points.keys()]
    y = [points[i][1] for i in points.keys()]

    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=60)
    n = len(points)

    # Add label to points
    for i, label in enumerate(points):
        plt.annotate('{}'.format(i), (x[i]+0.1, y[i]+0.1), size=25)

   # Add the arcs
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([points[i][0], points[j][0]], [
                         points[i][1], points[j][1]], 'b', alpha=.01)
    for (i, j) in zip(tour[:], tour[1:] + tour[:1]):
        plt.annotate("", xy=points[j], xytext=points[i],
                     arrowprops=dict(facecolor='black', width=1))
        plt.plot([points[i][0], points[j][0]], [
                 points[i][1], points[j][1]], 'r', alpha=1.)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title, fontsize=15)
    plt.grid()

    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()

# Given the model, executes the subtour elimination of possible subtours of the given step solution


def subtourelim(model, where):

    if where == GRB.Callback.MIPSOL:
        # make a list of arches and vertices selected in the solution
        vals = model.cbGetSolution(model._vars)
        vertices = model.cbGetSolution(model._vars_vertices)

        num_selected_vert = len([vertices[i]
                                for i in vertices if vertices[i] > 0.5])

        sel_verts = gp.tuplelist(i for i in vertices.keys()
                                 if vertices[i] > 0.5)
        # print(
        # f"Selected Vertices to search for subtours: {num_selected_vert}\nList: {sel_verts}")

        selected_archs = gp.tuplelist((i, j) for i, j in vals.keys()
                                      if vals[i, j] > 0.5)
        #print(f"Selected Arches to search for subtours:{selected_archs} ")

        # find the shortest cycle in the selected arches list, given the selected vertices (lenght of path)
        tour = subtour(selected_archs, sel_verts, num_selected_vert)

        if len(tour) < num_selected_vert:
            # if there is a subtour, add the elimination constr. for every pair of cities in tour
            if not QUIET:
                print('\n>>> Subtour eliminated  %s\n' % str(tour))
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour)-1)

# Given a tuplelist of arches, find the shortest subtour


def subtour(arches, vertices, n):
    unvisited = vertices  # list of unvisited vertices
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in arches.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


def main():

    n = N

    # creates a fully connected directed graph, with arches costs and vertex profits
    if RANDOM > 0:
        random.seed(12345)
        points = random_points(n)
        arch_costs = {(i, j): random.randint(1, 10)
                      for i in range(n) for j in range(n) if i != j}
        vertex_profits = {i:
                          random.randint(1, 10)
                          for i in range(n)}
    # else, loads a graph from two csv files with the specified FILE_PATH names
    else:
        if DATA_COMPARISON:
            points, vertex_profits, arch_costs = load_graph_dataset(
                filename=FILE_PATH_COMPARISON_DATASET)
        else:
            points, vertex_profits, arch_costs = load_graph_csv(
                points_file=FILE_PATH_POINTS, arch_file=FILE_PATH_COSTS)

    # shows the graph data and plots it
    print(
        f"Initialized the following directed graph:\nPoints: {points}\nPoints Profits: {vertex_profits}\n Arch Costs: {arch_costs}\n")
    plot_dgraph(points, vertex_profits, arch_costs, show_vars=True)

    if QUIET:
        gp.setParam('OutputFlag', 0)

    # CREATING THE MODEL
    m = gp.Model("Mod_TSP")
    m.reset()

    # CREATING THE MODEL VARIABLES

    # variables for vertex selection
    Y_vertices = m.addVars(points.keys(),
                           vtype=GRB.BINARY, name='v')
    print(f"Total vertices: {len(Y_vertices)}")

    # variables for arches selection
    X_archs = m.addVars(arch_costs.keys(),
                        vtype=GRB.BINARY, name='a')
    print(f"Total arcs: {len(X_archs)}")

    # CREATING THE MODEL CONSTRAINTS
    # add vertex 0 (DEPOT VERTEX) constraint: the vertex 0 will always be selected
    m.addConstr(Y_vertices[0] == 1, name="depot")

    # Add degree-2 constraint only on selected vertices (we might not need an hamiltonian tour!)
    m.addConstrs(X_archs.sum(i, '*') == Y_vertices[i]
                 for i in Y_vertices.keys())
    m.addConstrs(X_archs.sum('*', j) == Y_vertices[j]
                 for j in Y_vertices.keys())

    # CREATING THE MODEL OBJECTIVE FUNCTION
    """
    We want to select a closed path so that the total profit, 
    given by the total gain at the selected vertices minus the total cost 
    of the selected archs, is as high as possible
    """
    tot_gains = gp.quicksum(Y_vertices[i]*vertex_profits[i]
                            for i in Y_vertices.keys())
    tot_cost = gp.quicksum(X_archs[i, j]*arch_costs[i, j]
                           for i, j in X_archs.keys())

    m.setObjective(tot_gains - tot_cost, GRB.MAXIMIZE)

    m.update()

    m.Params.lazyConstraints = 1  # activating lazy constraints in the model
    m._vars = X_archs
    m._vars_vertices = Y_vertices
    start = time.process_time()  # start time for computation time
    m.optimize(subtourelim)  # optimize using subtours elimination
    end = time.process_time()  # end time

    var_x = m.getAttr('x', X_archs)
    var_y = m.getAttr('x', Y_vertices)

    # tuple-list of the selected vertices
    sel_verts = gp.tuplelist(i for i in var_y.keys()
                             if var_y[i] > 0.5)

    # tuple list of the selected arches
    selected_archs = gp.tuplelist((i, j)
                                  for i, j in var_x.keys() if var_x[i, j] > 0.5)

    # find the tour of len(sel_verts) for the solution after the sub-tours elimination
    tour = subtour(selected_archs, sel_verts, len(sel_verts))

    # print some info of the selected solution
    print('')
    print("Points: ", points)
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g\n Time = %g' % (m.objVal, end-start))
    print('')

    sol_archs = {(i, j): arch_costs[i, j] for i, j in selected_archs}
    sol_tour = {i: vertex_profits[i] for i in tour}
    sol_points = {i: points[i] for i in tour}
    print(
        f"Solution archs dict: {sol_archs}\nSolution tour dict: {sol_tour}\nSolution points dict: {sol_points}")

    # plot the directed graph tour solution
    plot_tour(points, tour)


if __name__ == '__main__':
    main()

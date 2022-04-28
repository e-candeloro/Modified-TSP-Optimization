# Modified-TSP-Optimization
This is the first homework done for the course of Automated Decision Making (2021-2022) at the University of Modena and Reggio Emilia.
In the Python script (and also notebook) a modified TSP optimization problem is solved using Gurobi and other common scientific libraries such as Pandas.

### Setup
Consider a directed acyclic graph **G = (V, A)** with vertex set **V = {0, 1, . . . , n − 1}**.
- Vertex 0 is a depot
- Let **c : A → Z** denote the cost of the arcs
- Let **p : V → Z** denote the profit of the vertices
### Objective of the homework
**The problem asks to find a tour that starts and terminates in
the depot, visist a (sub)set S ⊆ V of the vertices and
maximizes the difference between the prizes of the vertices
visited and the costs of the arcs used.**
## ALGORITHM DESCRIPTION
**NOTE**: the main script has comments explaining thealgorithm, also in this document all the
support functions used in the script for loading data, visualizing it, etc... are not described.
### 1)Data Loading and Visualization
Inside the main() function of the script, the data for a directed graph is generated in three
possible ways: loading the data from two CSV files, loading the data from one CSV file
(comparison data) or generating a random directed graph.
After the data generation, the graph is visualized using Matplotlib (the script will continue
closing the graph).
The three main data are: the points dict, the points profits dict and the arches dict with the
cost associated.

### 2)Model Creation
The Gurobi model is created and then the model variable, constraint and objective function
are initialized.
Variables:
- Y_vertices = vector for selecting some or all of  the given n vertices
- X_archs =  matrix for selecting the arches for a vertex i to a vertex j
### 3)Constraints:
- Depot constraint = always selects the vertex 0
- Degree-2 constraint = every selected vertex (with the Y_vertices vector) needs to have
one arch entering it and one arch leaving it, so a degree of two
### 4)Objective function:
The problem requires maximizing the total net profit: that is the difference between the total
gains minus the total costs, where:
- total gains = sum of all the selected  vertex profits
- total costs = sum of all the selected arches costs
### 5)Model Optimization using lazy constraints
Using the lazy constraints functionalities of Gurobi, the model is optimized applying a
sub-tour elimination for each step. The sub-tour elimination is considered only on the
selected vertices of that step, and it is necessary to avoid sub-tours that don’t consider the
vertex 0 (because we can find solutions where the vertex 0 is selected, but it is inside
another cycle!). The toy dataset shown in the default script, shows that the sub-tour
elimination discard multiple independent cycles as a solution.
After the optimization, the results are shown printing the optimal path (as a list of selected
vertices), the total maximum net profit obtained, the computation time and finally the graph
tour is shown.

## RESULTS
### Test Graph

![Toy graph](https://user-images.githubusercontent.com/67196406/165771780-7266bbf8-f654-4f2b-b9be-c17862bb1c90.png)

### Test Graph Solution

![Toy graph Solution](https://user-images.githubusercontent.com/67196406/165771951-75858949-1ea1-483d-aa36-15ae257fd60d.png)

**The solution of this graph is the selection of nodes 0,1,3 with a total gain of 50 + 50 + 1 = 101 minus the arch cost 1 + 1 +1 = 3
The total profit is 101 - 3 = 98**

### SCRIPT SOLUTION:
Optimal tour: **(0, 1, 3)**

Optimal cost: **98**

## ADDITIONAL INFO
All additional info can be found in the repositoty pdf files, as well as the notebook ipynb and Python script plus the csv datasets.

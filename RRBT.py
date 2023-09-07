import numpy as np
import cvxpy as cp
from scipy.stats import mvn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from core.commons import confidence_ellipse, cm2inch
from core.masterClasses import settings

# Set plot font sizes, types, etc.
mpl.rcParams['figure.dpi'] = 300
setup = settings()
np.random.seed(3)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Times"
})


def get_model(factor=1):
    # State transition matrix
    A = np.array([[1, 0.95, 0, 0],
                  [0, 0.90, 0, 0],
                  [0, 0, 1, 0.93],
                  [0, 0., 0, 0.96]])

    B = np.array([[0.48, 0],
                  [0.94, 0],
                  [0, 0.43],
                  [0, 0.92]])

    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

    w_cov = factor * np.diag([0.10, 0.02, 0.10, 0.02])
    v_cov = factor * np.diag([0.1, 0.1])

    u_min = [-4, -4]
    u_max = [4, 4]

    return A, B, C, w_cov, v_cov, u_min, u_max


class Model:
    def __init__(self, A, B, C, w_cov, v_cov, u_min, u_max):
        self.A = A
        self.B = B
        self.C = C
        self.w_cov = w_cov
        self.v_cov = v_cov

        self.dim_x = A.shape[0]
        self.dim_u = B.shape[1]
        self.dim_y = C.shape[0]

        self.U = {'min': u_min, 'max': u_max}

    def move_towards(self, x, point, state_space):
        # Compute control `u` to move `x` closest towards a given `point`

        x_next = cp.Variable(self.dim_x)
        u = cp.Variable(self.dim_u)

        # Define constraints
        constraints = [x_next == A @ x + B @ u,
                       u >= self.U['min'],
                       u <= self.U['max'],
                       x_next >= state_space[:, 0],
                       x_next <= state_space[:, 1]]

        # Objective is to minimize distance to the given point
        obj = cp.quad_form(x_next - point, np.eye(self.dim_x))

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()

        return prob.status, u.value, x_next.value

    def propagate_belief_covariance(self, belief_node, K):
        # Propagate a belief node based on a given control input

        # Kalman filter covariance prediction
        cov_predict = self.A @ belief_node.cov @ self.A.T + self.w_cov

        # Kalman filter covariance correction
        S = self.C @ cov_predict @ self.C.T + self.v_cov
        L = cov_predict @ self.C.T @ np.linalg.inv(S)
        cov = cov_predict - L @ self.C @ cov_predict  # Eq. (21)

        # Update belief node distribution
        Z = (self.A - self.B @ K)
        distr = Z @ belief_node.distr @ Z.T + L @ self.C @ cov_predict  # Eq. (33)

        cost = belief_node.cost + 1  # Eq. (11)

        # Create the belief node
        new_belief_node = Belief_node(cov, distr, cost, parent_belief=belief_node)

        return new_belief_node


class Vertex:
    def __init__(self, state, belief_nodes):
        self.state = state
        self.belief_nodes = belief_nodes
        return

    def add_belief_node(self, belief_node):
        self.belief_nodes.add(belief_node)


class Belief_node:
    def __init__(self, cov, distr, cost, parent_belief):
        self.cov = cov
        self.distr = distr
        self.cost = cost
        self.parent_belief = parent_belief
        self.parent_vertex = None
        self.parent_edge = None
        return

    def set_parent_vertex(self, parent_vertex):
        self.parent_vertex = parent_vertex
        return

    def set_parent_edge(self, parent_edge):
        self.parent_edge = parent_edge


class Edge:
    def __init__(self, parent, u, K, child):
        self.parent = parent
        self.u = u
        self.K = K
        self.child = child
        return


class Graph:
    def __init__(self):
        self.vertices = set()
        self.edges = set()

    # Define root node
    # def define_root(state, belief_nodes):
    #     root_vertex = Vertex(state, belief_nodes)
    #     self.vertices.add(root_vertex)
    #     return root_vertex

    def add_vertex(self, state, belief_nodes):
        new_vertex = Vertex(state, belief_nodes)
        self.vertices.add(new_vertex)

        return new_vertex

    def add_edge(self, edge):
        self.edges.add(edge)

        return

    def find_nearest_vertex(self, point):
        # Return the nearest vertex and all vertices within threshold radius
        n = len(self.vertices)
        d = len(point)
        threshold = 10 * (np.log10(n) / n) ** (1 / d)

        min_distance = np.infty
        nearest_vertex = None

        near_vertices = set()

        for vertex in self.vertices:
            distance = np.linalg.norm(point - vertex.state)
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = vertex

            if distance < threshold:
                near_vertices.add(vertex)

        return nearest_vertex, near_vertices


class Problem:

    def __init__(self, state_space):
        self.state_space = state_space
        self.dim = len(state_space)
        self.goal = set({})
        self.obstacles = set({})
        return

    def add_obstacle(self, box):
        self.obstacles.add(Object(box))

    def add_goal(self, box):
        self.goal.add(Object(box))

    def sample(self):
        done = False
        while not done:
            # Sample random point in state space
            rnd = np.random.rand(self.dim)
            point = self.state_space[:, 0] * (1 - rnd) + self.state_space[:, 1] * rnd

            # Check if it collides with an obstacle
            collision = False
            for obstacle in self.obstacles:
                if obstacle.contains(point):
                    collision = True

            if not collision:
                done = True

        return point


def probability_goal_contains(mean, belief_node, problem):
    # Check the total probability to be in a set of objects

    prob = 0

    for object in problem.goal:
        prob += mvn.mvnun(lower=object.box[:, 0], upper=object.box[:, 1],
                          means=mean, covar=belief_node.cov + belief_node.distr)[0]

    return prob


def probability_obstacle_contains(mean, belief_node, problem):
    # Check the total probability to be in a set of objects

    prob = 0

    for object in problem.obstacles:
        prob += mvn.mvnun(lower=object.box[:, 0], upper=object.box[:, 1],
                          means=mean, covar=belief_node.cov + belief_node.distr)[0]

    prob_in_state_space = mvn.mvnun(lower=problem.state_space[:, 0], upper=problem.state_space[:, 1],
                                    means=mean, covar=belief_node.cov + belief_node.distr)[0]

    prob += 1 - prob_in_state_space

    return prob


class Object:

    def __init__(self, box):
        self.box = box
        self.dim = len(box)

    def contains(self, point):

        # Check if the given point collides with obstacle
        if all(point <= self.box[:, 1]) and all(point >= self.box[:, 0]):
            return True
        else:
            return False


def plot_layout(ax, problem):
    # Draw goal states
    for goal in problem.goal:
        goal_lower = goal.box[[0, 2], 0]
        goal_width = goal.box[[0, 2], 1] - goal.box[[0, 2], 0]
        goalState = Rectangle(goal_lower, width=goal_width[0], height=goal_width[1],
                              color="green", alpha=0.3, linewidth=None)
        ax.add_patch(goalState)

    # Draw critical states
    for obstacle in problem.obstacles:
        obstacle_lower = obstacle.box[[0, 2], 0]
        obstacle_width = obstacle.box[[0, 2], 1] - obstacle.box[[0, 2], 0]
        criticalState = Rectangle(obstacle_lower, width=obstacle_width[0], height=obstacle_width[1],
                                  color="red", alpha=0.3, linewidth=None)
        ax.add_patch(criticalState)


def plot_graph(G, problem):
    # Plot graph (belief means and connect them via edges)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for vertex in G.vertices:
        plt.scatter(vertex.state[0], vertex.state[2], marker='o', c='r')

    for edge in G.edges:
        parent = edge.parent.parent_vertex
        child = edge.child.parent_vertex
        xvals = np.array([parent.state[0], child.state[0]])
        yvals = np.array([parent.state[2], child.state[2]])

        plt.plot(xvals, yvals, 'bo', linestyle="--")

    plot_layout(ax, problem)

    ax.set_xlim(problem.state_space[0, :])  # + [-1, 1])
    ax.set_ylim(problem.state_space[2, :])  # + [-1, 1])

    # Set tight layout
    fig.tight_layout()

    plt.show()

    return


def lqr(A, B, Q, R, verbose=False):
    import control as ct

    K, S, E = ct.dlqr(A, B, Q, R)

    if verbose:
        print('Eigenvalues of closed-loop system:', E)
        print('Control gain matrix:', K)

    return K


def extract_plan(G, vertex, belief_node):
    # Extract the plan from a graph

    # Iterate backwards
    plan = []
    done = False
    while not done:
        # If belief node has no parent, then we are already at the root
        if belief_node.parent_edge is None:
            break

        # Extract current step
        plan += [{
            'mean': vertex.state,
            'u': belief_node.parent_edge.u,
            'K': belief_node.parent_edge.K,
            'cov': belief_node.cov,
            'distr': belief_node.distr,
            'cost': belief_node.cost
        }]

        # Go one step back
        belief_node = belief_node.parent_edge.parent
        vertex = belief_node.parent_vertex

    return plan[::-1]


class KalmanFilter:

    def __init__(self, model, mean, covariance):
        self.model = model
        self.mean = mean
        self.covariance = covariance

        return

    def update(self, u, y, set):
        # Update Kalman filter belief
        mean_predict = self.model.A @ self.mean + self.model.B @ u
        covariance_predict = self.model.A @ self.covariance @ self.model.A.T + self.model.w_cov

        S = self.model.C @ covariance_predict @ self.model.C.T + self.model.v_cov
        L = covariance_predict @ self.model.C.T @ np.linalg.inv(S)

        mean = mean_predict + L @ (y - self.model.C @ mean_predict)
        covariance = covariance_predict - L @ self.model.C @ covariance_predict

        if set:
            self.mean = mean
            self.covariance = covariance

        return mean


class Simulator:

    def __init__(self, model, problem):
        self.plan = plan
        self.model = model
        self.problem = problem
        self.simulations = []
        return

    def simulate(self, plan, constrain_input=True):
        # Generate a new simulation under the given plan

        # Sample initial condition
        x = np.zeros((len(plan) + 1, self.model.dim_x))
        x[0] = np.random.multivariate_normal(self.model.x0, self.model.cov0)
        y = np.zeros((len(plan) + 1, self.model.dim_y))
        y[0] = np.random.multivariate_normal(C @ x[0], self.model.v_cov)

        u_nom = np.zeros((len(plan), self.model.dim_u))
        u_stabilize_unconstrained = np.zeros((len(plan), self.model.dim_u))
        u_stabilize = np.zeros((len(plan), self.model.dim_u))

        # Generate noise samples
        w = np.random.multivariate_normal(np.zeros(self.model.dim_x), self.model.w_cov, len(plan))
        v = np.random.multivariate_normal(np.zeros(self.model.dim_y), self.model.v_cov, len(plan))

        filter = KalmanFilter(self.model, self.model.x0, self.model.cov0)

        for k in range(len(plan)):

            nominal_state = plan[k]['mean']
            u_nom[k] = plan[k]['u']
            K = plan[k]['K']

            # Update state by applying nominal control `u` and adding process noise `w[k]`
            x[k + 1] = self.model.A @ x[k] + self.model.B @ u_nom[k] + w[k]
            y[k + 1] = self.model.C @ x[k + 1] + v[k]

            # Now update filter
            mean = filter.update(u_nom[k], y[k + 1], set=False)

            # Stabilize with `K` proportional to difference between filter man and the nominal state
            u_stabilize_unconstrained[k] = - K @ (mean - nominal_state)

            if constrain_input:
                u_stabilize[k] = np.maximum(np.minimum(u_stabilize_unconstrained[k],
                                                       model.U['max'] - u_nom[k]),
                                            model.U['min'] - u_nom[k])

            else:
                u_stabilize[k] = u_stabilize_unconstrained[k]

            x[k + 1] = self.model.A @ x[k] + self.model.B @ (u_nom[k] + u_stabilize[k]) + w[k]
            y[k + 1] = self.model.C @ x[k + 1] + v[k]

            _ = filter.update(u_nom[k], y[k + 1], set=True)

        u_error = u_stabilize - u_stabilize_unconstrained

        return x, u_nom, y, u_stabilize, u_error


def plot_plan(model, plan, sims, problem, file_suffix='NA'):
    # Plot graph (belief means and connect them via edges)

    # Number of std deviations to plot ellipses for
    n_std = 2

    fig, ax = plt.subplots(figsize=cm2inch(6.1, 5))

    plt.xlabel('x', labelpad=0)
    plt.ylabel('y', labelpad=0)

    # ax.set_aspect('equal')
    ax.set_xticks([-10, -2, 6])
    ax.set_yticks([-10, -2, 6])

    plot_layout(ax, problem)

    # Plot simulations
    for (x, u, y, u_stabilize, u_error) in sims:
        plt.plot(x[:, 0], x[:, 2], 'ro', markersize=0.6, linestyle="dotted", linewidth=0.05, alpha=0.7)

    xfrom = model.x0[0]
    yfrom = model.x0[2]

    confidence_ellipse(model.x0[[0, 2]], model.cov0[[0, 2], :][:, [0, 2]], ax, n_std=n_std, edgecolor='black',
                       facecolor='none', linewidth=0.6, zorder=10)

    for k in range(len(plan)):
        mean = plan[k]['mean'][[0, 2]]
        cov = plan[k]['cov'][[0, 2], :][:, [0, 2]] + plan[k]['distr'][[0, 2], :][:, [0, 2]]

        xto = mean[0]
        yto = mean[1]

        confidence_ellipse(mean, cov, ax, n_std=n_std, edgecolor='black',
                           facecolor='none', linewidth=0.6, zorder=10)

        xvals = np.array([xfrom, xto])
        yvals = np.array([yfrom, yto])
        plt.plot(xvals, yvals, 'bo', linestyle="--", linewidth=1, markersize=1)

        xfrom = xto
        yfrom = yto

    ax.set_xlim(problem.state_space[0, :])  # + [-1, 1])
    ax.set_ylim(problem.state_space[2, :])  # + [-1, 1])

    # Set tight layout
    fig.tight_layout()

    # Save figure
    plt.savefig('RRBT_scatter_{}.pdf'.format(str(file_suffix)), format='pdf', bbox_inches='tight')

    plt.show()

    return


def simulate_plan_and_plot(model, problem, plan, MC, MC_iters_plot, CONSTRAIN_INPUT):
    SIM = Simulator(model, problem)
    sims = [None] * MC
    for i in range(MC):
        sims[i] = SIM.simulate(plan, constrain_input=CONSTRAIN_INPUT)

    if CONSTRAIN_INPUT:
        file_suffix = 'withInputConstraints'
    else:
        file_suffix = 'noInputConstraints'

    plot_plan(model, plan, sims[:MC_iters_plot], problem, file_suffix=file_suffix)

    # Check empirical safety probability at time k=4
    nr_safe = 0
    k = 12
    for (x, u, y, u_stabilize, u_error) in sims:

        collision = False
        for obstacle in problem.obstacles:
            if obstacle.contains(x[k]):
                collision = True

        if not collision:
            nr_safe += 1

    print('Fraction of paths safe at k=4 is: {:.2f}'.format(nr_safe / MC))


if __name__ == "__main__":

    # Noise multiplication factor
    f = 1

    # RRBT iterations
    M = 1000

    # Monte Carlo iterations (simulations of trajectories)
    MC = 100
    MC_iters_plot = 25

    verbose = False

    # Get dynamical model
    A, B, C, w_cov, v_cov, u_min, u_max = get_model(f)
    model = Model(A, B, C, w_cov, v_cov, u_min, u_max)
    model.x0 = np.array([-8, 0, -8, 0])
    model.cov0 = np.diag([2, .01, 2, .01])

    # Define state space
    state_space = np.array([
        [-11, 11],
        [-7.5, 7.5],
        [-11, 11],
        [-7.5, 7.5]
    ])

    # Define obstacles
    obstacles = [
        np.array([[-9.5, -3.5], [-7.5, 7.5], [-2, 3], [-7.5, 7.5]]),
        np.array([[0, 4.5], [-7.5, 7.5], [-10, -6], [-7.5, 7.5]]),
        np.array([[-0.7, 5], [-7.5, 7.5], [-1.5, 3], [-7.5, 7.5]]),
        np.array([[1.0, 4], [-7.5, 7.5], [8, 11], [-7.5, 7.5]]),
    ]

    problem = Problem(state_space)
    problem.add_goal(box=np.array([[-11, -5], [-7.5, 7.5], [5, 11], [-7.5, 7.5]]))
    for obstacle in obstacles:
        problem.add_obstacle(box=obstacle)

    # Initial state conditions
    distr0 = np.zeros((model.dim_x, model.dim_x))

    # Define empty graph with root vertex plus belief node
    G = Graph()
    root_node = G.add_vertex(model.x0, set())
    root_belief_node = Belief_node(model.cov0, distr0, 0, None)
    root_belief_node.set_parent_vertex(root_node)
    root_node.add_belief_node(root_belief_node)

    delta = 0.01
    eta = 0.99
    i = 0
    success = False
    goal_vertex = None
    goal_belief_node = None

    while i < M:
        if i % 100 == 0:
            print('\nIteration {}'.format(i))

        # Sample random `point` free of any obstacle
        point = problem.sample()

        # Find vertex closest to `point`
        v_nearest, v_near = G.find_nearest_vertex(point)

        # Create edge toward `v_nearest` (as much as possible in one control step)
        status, u, x_next = model.move_towards(v_nearest.state, point, problem.state_space)

        belief_node_queue = set()

        # Search for a belief node of `v_nearest` that leads to a safe transition
        if status != 'optimal':
            if verbose:
                print('>> WARNING: LP TO PROPAGATE DYNAMICS COULD NOT BE SOLVED')
            # Skip to next iteration
            continue

        # Line 7: CONNECT `v_nearest.x` with `x_next`
        Qhat = np.diag([1, 1, 1, 1])  # np.eye(model.dim_x)
        Rhat = 0.01 * np.eye(model.dim_u)
        K = lqr(model.A, model.B, Qhat, Rhat)

        # Line 8: PROPAGATE `e_nearest` for every belief node `n` of `v_nearest`
        # Check if there is a belief node in `v_nearest` that leads to a safe transition
        for belief_node in v_nearest.belief_nodes:

            # Propagate belief covariance
            belief_node_plus = model.propagate_belief_covariance(belief_node, K)

            # Check the probability that the successor belief node is unsafe
            prob_unsafe = probability_obstacle_contains(x_next, belief_node_plus, problem)

            if prob_unsafe < delta:
                # Safe, so add this belief node to list
                belief_node_queue.add((belief_node, belief_node_plus))

        # If there exist safe successor belief nodes
        if len(belief_node_queue) > 0:
            if verbose:
                print('-- Add vertex at belief mean {}'.format(x_next))

            # Line 9: Add the new node
            new_vertex = G.add_vertex(x_next, set())

            # Line 10: Add its belief nodes
            for (belief_node_from, belief_node_to) in belief_node_queue:
                # Add belief node for the newly added vertex
                belief_node_to.set_parent_vertex(new_vertex)
                new_vertex.add_belief_node(belief_node_to)

                # Add edge in the graph
                new_edge = Edge(belief_node_from, u, K, belief_node_to)
                G.add_edge(new_edge)
                belief_node_to.set_parent_edge(new_edge)

            # Line 14: Also connect other vertices that are "nearby"
            for near_vertex in v_near:

                # Check if we can indeed move towards the new vertex from this near vertex
                status, u, x_next = model.move_towards(near_vertex.state, new_vertex.state, problem.state_space)

                # If this transition is indeed possible, also add this corresponding edge
                if status == 'optimal':
                    if all(np.isclose(new_vertex.state, x_next)):
                        if verbose:
                            print('-- Make edge from existing belief node')

                        # Check if there is a belief node in `v_nearest` that leads to a safe transition
                        for near_belief_node in near_vertex.belief_nodes:

                            # Propagate belief covariance
                            belief_node_to = model.propagate_belief_covariance(near_belief_node, K)

                            # Check the probability that the successor belief node is unsafe
                            prob_unsafe = probability_obstacle_contains(x_next, belief_node_to, problem)

                            if prob_unsafe < delta:
                                # Check if this belief node already exists
                                exists = False

                                # Iterate over belief nodes of new vertex
                                for n in new_vertex.belief_nodes:
                                    if np.all(np.isclose(belief_node_to.cov, n.cov)) and \
                                            np.all(np.isclose(belief_node_to.distr, n.distr)):

                                        exists = True

                                        # Belief node already exists, so check if we can decrease cost
                                        if belief_node_to.cost < n.cost:
                                            # Update edge wiring
                                            n.parent_edge.parent = near_belief_node
                                            n.parent_edge.u = u
                                            n.parent_edge.K = K
                                            if verbose:
                                                print(
                                                    '--- Successor belief already exists; decrease cost from {} to {}'.format(
                                                        n.cost, belief_node_to.cost))

                                # If this belief node does not yet exist, create a new one
                                if not exists:
                                    # Add belief node for the newly added vertex
                                    belief_node_to.set_parent_vertex(new_vertex)
                                    new_vertex.add_belief_node(belief_node_to)

                                    # Add edge in the graph
                                    new_edge = Edge(near_belief_node, u, K, belief_node_to)
                                    G.add_edge(new_edge)
                                    belief_node_to.set_parent_edge(new_edge)

            # Check if the goal has been reached already
            for belief_node in new_vertex.belief_nodes:
                goal_probability = probability_goal_contains(new_vertex.state, belief_node, problem)

                # print('Probability to reach the goal is: {:.2f}'.format(goal_probability))
                if goal_probability > eta:

                    success = True
                    print('> GOAL REACHED WITH SUFFICIENT PROBABILITY')

                    if goal_vertex is not None:
                        # If cost can be decreased, replace
                        if belief_node.cost < goal_belief_node.cost:
                            print('> DECREASE BEST COST FROM {} TO {}'.format(goal_belief_node.cost,
                                                                              belief_node.cost))
                            goal_vertex = new_vertex
                            goal_belief_node = belief_node

                        else:
                            print('> COST COULD NOT BE DECREASED (BEST: {}, CURRENT: {})'.format(goal_belief_node.cost,
                                                                                                 belief_node.cost))

                    else:
                        goal_vertex = new_vertex
                        goal_belief_node = belief_node

        i += 1

    # Plot the RRBT
    plot_graph(G, problem)

if success == True:
    # Extract the best plan from the RRBT
    plan = extract_plan(G, goal_vertex, goal_belief_node)

    # Simulate and plot once without input constraints, and once with input constraints
    simulate_plan_and_plot(model, problem, plan, MC, MC_iters_plot, CONSTRAIN_INPUT=False)
    simulate_plan_and_plot(model, problem, plan, MC, MC_iters_plot, CONSTRAIN_INPUT=True)

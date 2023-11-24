# 2D Quadrotor MPC
import numpy as np
import matplotlib.pyplot as plt
import cvxpy


xlim = 15
ulim = 5


def mpc_one_step(current_state: np.ndarray):
    # Define system dynamics and cost
    A = np.array([[1,1], [0,1]])
    B = np.array([[0], [1]])
    Q = np.diag([1.0, 1.0])
    R = np.array([[1.0]])

    N = 5

    # Initial state
    x0 = current_state

    # Reference state
    xg = np.array([[0], [0]])

    # Define optimization variables
    x = cvxpy.Variable((2, N+1), name="x")
    u = cvxpy.Variable((1, N), name="u")

    # Define parameters for the initial state and reference state
    x_init = cvxpy.Parameter((2, 1), name="x_init")
    x_init.value = x0
    x_ref = cvxpy.Parameter((2, 1), name="x_ref")
    x_ref.value = xg

    objective = 0.0
    constraints = []

    # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
    for i in range(N):
        objective += cvxpy.quad_form(u[:, i], R)

    # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q
    for i in range(N+1):
        objective += cvxpy.quad_form(x[:, i:i+1] - x_ref, Q)

    # Add dynamics constraints to the optimization problem
    for i in range(N):
        constraints += [x[:, i+1] == A @ x[:, i] + B @ u[:, i]]

    # Add L-inf norm of the control inputs and states to the constraints
    for i in range(N):
        constraints += [
            x[:, i] <=  xlim,
            x[:, i] >= -xlim,
            u[:, i] <=  ulim,
            u[:, i] >= -ulim,
        ]


    # Add constraints on the initial state
    constraints += [x[:, 0] == cvxpy.vec(x_init)]

    problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    # Solve the optimization problem for a single step
    problem.solve(warm_start=True, solver=cvxpy.OSQP, polish=True)
    if problem.status != cvxpy.OPTIMAL:
        return None

    return u[:, 0].value[0]


def main():
    samples = 500
    X1 = np.linspace(-xlim, xlim, num=samples)
    X2 = np.linspace(-xlim, xlim, num=samples)

    valid_data = []
    invalid_data = []

    # TODO: parallelize using joblib.parallel
    for i, x1 in enumerate(X1):
        print(f'Progress: {round(i / len(X1) * 100, 2)}%', end='\r')

        for x2 in X2:
            u = mpc_one_step(np.array([[x1], [x2]]))
            if u is not None:
                valid_data.append([x1, x2, u])
            else:
                invalid_data.append([x1, x2])
    print('Progress: 100.00%')

    valid_data = np.array(valid_data)
    invalid_data = np.array(invalid_data)

    np.savetxt(f'valid_data_{samples}x{samples}.csv', valid_data, fmt='%f', delimiter=',')
    np.savetxt(f'invalid_data_{samples}x{samples}.csv', invalid_data, fmt='%f', delimiter=',')

    # plot
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], c = valid_data[:, 2])

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('u')

    plt.subplot(1, 2, 2)
    plt.scatter(invalid_data[:, 0], invalid_data[:, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    # Add box constraint on x plot
    plt.plot([-xlim, -xlim], [-xlim, xlim], color="red")
    plt.plot([xlim, xlim], [-xlim, xlim], color="red")
    plt.plot([-xlim, xlim], [-xlim, -xlim], color="red")
    plt.plot([-xlim, xlim], [xlim, xlim], color="red", label="Constraints")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse
import json
import ast
import math

def parse_arguments():
    try:
        __IPYTHON__
        _in_ipython_session = True
    except NameError:
        _in_ipython_session = False

    if _in_ipython_session:
        args = argparse.Namespace(split='adapt_test', 
                                  savepath="data_2D/", 
                                  seed=2024, 
                                  verbose=1, 
                                  dimension=2,
                                  nb_steps=100)
        return args

    else:
        parser = argparse.ArgumentParser(description='Generate ODE data for multiple dynamical systems')
        parser.add_argument('--split', type=str, choices=['train', 'test', 'adapt', 'adapt_test'], default='train', help='Data split to generate')
        parser.add_argument('--savepath', type=str, default='tmp/', help='Path to save generated data')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
        parser.add_argument('--dimension', type=int, default=2, help='Dimension of ODEs')
        parser.add_argument('--nb_steps', type=int, default=100, help='Number of time steps to simulate')
        return parser.parse_args()

def parse_lambda(lambda_str):
    """Parse a lambda function string and return a callable function."""
    lambda_ast = ast.parse(lambda_str).body[0].value
    return eval(compile(ast.Expression(lambda_ast), '<string>', 'eval'))

def load_ode_definitions(dimension):
    # Load ODE definitions: Symbolic Regression of Dynamical Systems with Transformers
    with open(f'ode_definitions_{dimension}D.json', 'r') as f:
        ode_defs = json.load(f)

    # Parse the lambda functions
    for ode in ode_defs.values():
        ode['function'] = parse_lambda(ode['function'])
    
    return ode_defs

def generate_environments(reference_params, n_envs, adaptation=False):
    if adaptation:
        # Generate 2 environments in training domain and 2 outside
        envs = []
        for i in range(4):
            env = {}
            for param, value in reference_params.items():
                if i < 2:
                    # In training domain
                    env[param] = value * np.random.uniform(0.9, 1.1)
                else:
                    # Outside training domain
                    env[param] = value * np.random.uniform(0.8, 1.2)
            envs.append(env)
    else:
        # Generate training environments
        envs = []
        for _ in range(n_envs):
            env = {}
            for param, value in reference_params.items():
                env[param] = value * np.random.uniform(0.8, 1.2)
            envs.append(env)
    return envs

def generate_initial_conditions(reference_ic, n_ic):
    ic1, ic2 = reference_ic

    # return [np.array(reference_ic) * np.random.uniform(0.8, 1.2, size=len(reference_ic)) for _ in range(n_ic)]
    # return [np.array(reference_ic)]

    ## Sample a paramter to interpolate between the two initial conditions
    # alpha = np.random.uniform(0, 1, size=1)
    # return [alpha * np.array(ic1) + (1 - alpha) * np.array(ic2)]

    ## Sample a paramter to interpolate. Higer probability to sample either of the two initial conditions themselves. Have one sample paramter for each dimension
    # initial_conditions = []
    # for _ in range(n_ic):
    #     alpha = np.random.uniform(0, 1, size=len(reference_ic))
    #     # alpha = np.where(alpha < 0.1, 0, alpha)
    #     # alpha = np.where(alpha > 0.9, 1, alpha)
    #     initial_conditions.append(alpha * np.array(ic1) + (1 - alpha) * np.array(ic2))

    # return initial_conditions

    ## Rndomly pick one of the two initial conditions
    return [np.array(ic1) if np.random.uniform(0, 1) < 0.5 else np.array(ic2) for _ in range(n_ic)]


def simulate_ode(ode_func, t_span, initial_state, args, dt):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    # print("t_eval", t_eval.shape)
    solution = solve_ivp(ode_func, t_span, initial_state, args=args, t_eval=t_eval, method='RK45')
    # print("solution", solution.y.shape, t_eval.shape, t_eval[-1], solution.t[-1])
    return solution.t, solution.y.T

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    ode_definitions = load_ode_definitions(args.dimension)
    
    if args.split == 'train':
        n_envs, n_ic = 9, 4
    elif args.split == 'test':
        n_envs, n_ic = 9, 32
    elif args.split == 'adapt':
        n_envs, n_ic = 4, 1
    elif args.split == 'adapt_test':
        n_envs, n_ic = 4, 32

    all_data = []
    all_t_eval = []
    all_environments = {}

    for ode_id, ode_info in ode_definitions.items():
        if args.verbose:
            print(f"Processing ODE {ode_id}: {ode_info['name']}")

        # Generate environments and initial conditions
        environments = generate_environments(ode_info['parameters'], n_envs, args.split in ['adapt', 'adapt_test'])
        initial_conditions = generate_initial_conditions(ode_info['initial_values'], n_ic)

        # Adjust time parameters
        T = ode_info.get("time_horizon", 1)
        dt = T / args.nb_steps  # Approximately 20 time steps per ODE

        ode_data = np.zeros((n_envs, n_ic, int(T/dt), len(ode_info['initial_values'])))

        for i, env in enumerate(environments):
            for j, ic in enumerate(initial_conditions):
                t, trajectory = simulate_ode(ode_info['function'], (0, T), ic, tuple(env.values()), dt)
                ode_data[i, j] = trajectory

        all_data.append(ode_data)
        all_t_eval.append(t)
        all_environments[ode_id] = environments

    # Save data
    filename = f"{args.savepath}/{args.split}_data.npz"
    np.savez(filename, t=np.stack(all_t_eval), X=np.stack(all_data))
    # Save environments
    with open(f"{args.savepath}/{args.split}_envs.json", 'w') as f:
        json.dump(all_environments, f, indent=4)

    if args.verbose:
        print(f"Data saved to {filename}")



if __name__ == "__main__":
    main()






#%%

if __name__ == "__main__":

    ## PLot all trajectories on the same plot for one single ODE
    ode_id = 0

    args = parse_arguments()
    ode_defs = load_ode_definitions(args.dimension)

    for ode_id, ode_def in enumerate(ode_defs.keys()):
        ## Load the data
        filename = f"{args.savepath}/{args.split}_data.npz"
        data = np.load(filename)

        t = data['t'][ode_id]
        X = data['X'][ode_id]

        # print("t", t.shape)
        # print("X", X.shape)

        plt.figure(figsize=(10, 4))

        ## Plot all environments from the same initial condition
        for i in range(X.shape[0]):
            plt.plot(t, X[:, i, :, 0].T, color="royalblue", alpha=(i+1)/(X.shape[0]+1))
            plt.plot(t, X[:, i, :, 1].T, color="crimson", alpha=(i+1)/(X.shape[0]+1))

        plt.title(f"ODE {ode_def}")

        plt.xlabel("Time")
        # plt.legend()
        plt.show()

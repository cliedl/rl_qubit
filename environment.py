import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QubitEnv(gym.Env):
    """
    A custom environment for simulating a qubit using OpenAI's gym interface.
    The state of the qubit is represented by a vector [E, Re(S), Im(S)],
    where 
    E: energy (excited state probability), 
    S: Coherence (related to dipole moment)
    """

    def __init__(self,
                 initial_state=np.array([0, 0, 0], dtype=np.float32),
                 target_state=np.array([1, 0, 0], dtype=np.float32),
                 dt=0.001):
        """
        Initialize the environment.

        Parameters:
        - initial_state: The starting state of the qubit.
        - target_state: The target state we want the qubit to reach.
        - dt: Time step for the simulation (in units of the natural lifetime of the qubit)
        """
        super(QubitEnv, self).__init__()

        self.initial_state = np.array(initial_state, dtype=np.float32)
        self.state = self.initial_state.copy()
        self.target_state = np.array(target_state, dtype=np.float32)
        self.dt = dt

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=np.array([-100., -100., 0]),
                                       high=np.array([100., 100., 1]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0, -0.5, -0.5]),
                                            high=np.array([1., 0.5, 0.5]),
                                            dtype=np.float32)

    def step(self, action):
        """
        Apply an action to the environment and step it forward in time.

        Parameters:
        - action: The control parameters (Omega, Delta) and a flag indicating if the episode is done.

        Returns:
        - state: The new state of the qubit.
        - reward: The reward from the current action.
        - done: Whether the episode is finished.
        - info: Additional information.
        """
        Omega, Delta, done = action

        E, S_real, S_imag = self.state
        S = S_real + 1j * S_imag

        # optical Bloch equations
        # TODO: Think about state representation, maybe Bloch sphere angles instead?
        E_new = E + (-E + 0.5 * 1j * Omega * (S - np.conj(S))) * self.dt
        S_new = S + (-1j * Delta * S - 0.5 * S - 0.5 *
                     1j * Omega * (1 - 2 * E)) * self.dt

        self.state = [np.real(E_new), np.real(S_new), np.imag(S_new)]

        # Compute the reward
        reward = self.compute_reward(done)

        # TODO: Define the conditions for episode termination.
        # For now, we terminate the episode if the done action is selected by the agent.
        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
        - state: The reset state of the qubit.
        """
        self.state = self.initial_state.copy()
        return self.state

    def compute_reward(self, done):
        """
        Compute the reward based on the current state and the target state.

        Parameters:
        - done: Whether the action indicated the end of the episode.

        Returns:
        - reward: The computed reward.
        """

        # For now: Reward is simply eucl. distance between current state and target state when episode finishes, otherwise 0
        if done:
            distance = np.linalg.norm(self.state - self.target_state)
            reward = - distance
        else:
            reward = 0
        return reward

    def render(self):
        # Extract the state components
        E, S_real, S_imag = self.state

        # Construct Bloch vector from state components
        x, y, z = 2*S_real, 2*S_imag, 1-2*E

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw the Bloch sphere
        phi, theta = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        PHI, THETA = np.meshgrid(phi, theta)
        R = 1  # Radius of the sphere
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        # Plot the surface
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        color='black', alpha=0.3, linewidth=0)

        # Plot the qubit state
        ax.scatter([x], [y], [z], color="blue", s=50)

        # Set plot display parameters
        ax.set_xlim([-R*1.01, R*1.01])
        ax.set_ylim([-R*1.01, R*1.01])
        ax.set_zlim([-R*1.01, R*1.01])
        ax.set_xlabel('Re(d)')
        ax.set_ylabel('Im(d)')
        ax.set_zlabel('Inversion')

        return fig

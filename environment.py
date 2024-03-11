import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class QubitEnv(gym.Env):
    """
    A custom environment for simulating a qubit using OpenAI's gym interface.
    The state of the qubit is represented by a vector [E, Re(S), Im(S)],
    where 
    E: energy (excited state probability), 
    S: Coherence (related to dipole moment)
    """

    def __init__(self,
                 initial_state=np.array([0., 0., 0.]),
                 target_state=np.array([1., 0., 0.]),
                 dt=0.001):
        """
        Initialize the environment.

        Parameters:
        - initial_state: The starting state of the qubit.
        - target_state: The target state we want the qubit to reach.
        - dt: Time step for the simulation (in units of the natural lifetime of the qubit)
        """
        super().__init__()

        self.initial_state = initial_state
        self.state = self.initial_state.copy()
        self.target_state = target_state
        self.dt = dt

        # Define the action and observation spaces
        self.action_space = Box(low=np.array([0., -20., 0.]),
                                high=np.array([20., 20., 1.]))

        # TODO: This should eventually be limited to measurement outcomes, not the entire state
        self.observation_space = Box(low=np.array([0, -1., -1.]),
                                     high=np.array([1., 1., 1.]))

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
        Omega, Delta, terminated = action
        terminated = terminated > 0.5

        E, S_real, S_imag = self.state
        S = S_real + 1j * S_imag

        # optical Bloch equations
        # TODO: Think about state representation, maybe Bloch sphere angles instead?
        E_new = E + (-E + 0.5 * 1j * Omega * (S - np.conj(S))) * self.dt
        S_new = S + (-1j * Delta * S - 0.5 * S - 0.5 *
                     1j * Omega * (1 - 2 * E)) * self.dt

        self.state = [np.real(E_new), np.real(S_new), np.imag(S_new)]

        # Compute the reward
        reward = self.compute_reward(terminated)

        # TODO: Define the conditions for episode termination.
        truncated = False
        # For now, we terminate the episode if the done action is selected by the agent.
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        """
        Reset the environment to the initial state.

        Returns:
        - state: The reset state of the qubit.
        """
        self.state = self.initial_state.copy()
        return self.state, {}

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

import numpy as np


class GridWorld(object):
    def __init__(self, shape, reward, num_of_w, num_of_p, num_of_r, num_of_steps):
        """
        to randomly initiate the map and starting point
        :param shape: shape of the world
        :param reward: a tuple or list of length 3, refers to reward of normal, penalty and reward cells respectively
        :param num_of_w: number of walls
        :param num_of_p: number of penalty cells
        :param num_of_r: number of reward cells
        :param num_of_steps: number of walking steps in each training sample
        """
        self.shape = shape
        self.state_space = shape[0]*shape[1]
        self.action_space = 5
        self.num_of_steps = num_of_steps

        # randomly select wall, penalty and reward cells
        rand = np.random.choice(range(self.state_space), size=num_of_w+num_of_p+num_of_r, replace=False)
        self.w = rand[0:num_of_w]
        self.p = rand[num_of_w:num_of_w+num_of_p]
        self.r = rand[num_of_w+num_of_p:]

        # build reward matrix
        self.R = np.ones(self.state_space) * reward[0]
        self.R[self.w] = None
        self.R[self.p] = reward[1]
        self.R[self.r] = reward[2]

        # build action matrix, if in state i, the agent can do the action j, then A(i,j) == 1, otherwise 0
        self.A = np.array(tuple(map(self._if_action_valid, range(self.state_space))))

        # To find the exact best solution in each state.
        # If in state i, action j is the best solution, then A(i,j) == 1, otherwise 0
        self.Optimal = self._find_optimal()

        # find out the space where the agent is safe to start(not in W, P or R) and set the starting point
        self.start_state = tuple(set(range(self.state_space)).difference(set(rand)))
        self.state = None
        self.steps = 0
        self.reset()

    def reset(self):
        """
        To reset to a random starting point
        """
        self.state = np.random.choice(self.start_state)
        self.steps = 0

    def step(self, action):
        """
        Given an action, the agent make a move
        :param action: an int in range(5)
        :return: state after action, instant reward, if the sample is done
        """
        self.state = self._act(self.state, action)
        self.steps += 1
        if self.steps == self.num_of_steps:
            done = True
        else:
            done = False
        return self.state, self.R[self.state], done

    def render(self):
        """
        Render a picture of the map
        """
        for i in range(self.state_space):
            if i in self.w:
                print('W ', end='')
            elif i in self.p:
                print('P ', end='')
            elif i in self.r:
                print('R ', end='')
            elif i == self.state:
                print('X ', end='')
            else:
                print('O ', end='')
            if (i + 1) % self.shape[1] == 0:
                print('\n')

    def _if_action_valid(self, state):
        if state in self.w:
            return np.full(self.action_space, np.nan)

        allow_list = np.ones(self.action_space)
        i, j = self._coordinate_transform(state)

        if i == 0:
            allow_list[1] = np.nan
        elif i == self.shape[0]-1:
            allow_list[2] = np.nan

        if j == 0:
            allow_list[3] = np.nan
        elif j == self.shape[1]-1:
            allow_list[4] = np.nan

        for wall in self.w:
            w_i, w_j = self._coordinate_transform(wall)
            if w_i != i and w_j != j:
                continue
            elif w_i == i and np.abs(w_j - j) == 1:
                if w_j - j == -1:
                    allow_list[3] = np.nan
                else:
                    allow_list[4] = np.nan
            elif w_j == j and np.abs(w_i - i) == 1:
                if w_i - i == -1:
                    allow_list[1] = np.nan
                else:
                    allow_list[2] = np.nan

        return allow_list

    def _act(self, state, action):
        if np.isnan(self.A[state][action]):
            return -1
        else:
            i, j = self._coordinate_transform(state)
            if action == 1:
                i = i - 1
            elif action == 2:
                i = i + 1
            elif action == 3:
                j = j - 1
            elif action == 4:
                j = j + 1
            return self._reversed_transform(i, j)

    @staticmethod
    def _get_reward(state, r_matrix):
        if state == -1:
            return np.nan
        else:
            return r_matrix[state]

    def _coordinate_transform(self, index):
        return index // self.shape[1], index % self.shape[1]

    def _reversed_transform(self, i, j):
        return i * self.shape[1] + j

    def _all_next_state(self, state):
        return self._act(state, 0), self._act(state, 1), self._act(state, 2), self._act(state, 3), self._act(state, 4)

    def _find_optimal(self):
        s_next = np.array(tuple(map(self._all_next_state, range(self.state_space))))
        r_matrix = np.zeros(shape=(self.state_space, self.action_space))
        for i in range(self.num_of_steps):
            r_matrix = np.array([[self._get_reward(x, self.R) + np.nanmax(self._get_reward(x, r_matrix)) for x in row] for row in s_next])
        return np.array([row == np.nanmax(row) for row in r_matrix])

    def one_hot(self, state):
        oh = np.zeros(self.state_space - len(self.w))
        idx = state
        for i in range(state):
            if i in self.w:
                idx -= 1
        oh[idx] = 1
        return oh


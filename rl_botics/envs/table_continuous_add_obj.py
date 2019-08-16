import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from collections import deque


class StochasticContinuousTable:
    """
        Table Environment with clean and dirty dishes during Human Robot Collaboration
        Humans can now add and remove objects from the scene.
    """
    def __init__(self,
                 sizes=None,
                 max_clean=4,
                 max_dirty=4,
                 max_human=1,
                 obj_width=0.1,
                 rand_obj=True,
                 cam_loc=None,
                 partial=True,
                 noise=True,
                 cam_view=False,
                 time_limit=None,
                 hist_len=8
                 ):
        """
        :param sizes:       Array of sizes of the grid such that: low <= x < high, low < y < high
        :param max_clean:   (int) Number of clean objects
        :param max_dirty:   (int) Number of dirty objects
        :param max_human:   (int) Number of humans
        :param obj_width:   (Float) Width of each object. Currently, all objects have same width
        :param rand_obj:    (Boolean) If true, number of object on table varies
        :param cam_loc:     (Numpy array of shape (1,2) ) Camera location (fixed)
        :param partial:     (Boolean) True if POMDP, else, MDP
        :param noise:       (Boolean) True if noisy observation
        :param cam_view:    (Boolean) True to plot camera view
        :param static:      (Boolean) If true, same grid at every episode otherwise random postions at every episode
        """
        # low <= x < high, low < y < high
        if sizes:
            self.low = sizes[0]
            self.high = sizes[1]
        else:
            self.low = -1
            self.high = 1

        # Partial Observability and Noisy Sensor Reading
        self.partial = partial
        self.noise = noise

        # Camera location
        if cam_loc:
            self.cam_loc = cam_loc
        else:
            self.cam_loc = np.zeros((1, 2))
            self.cam_loc[0, 0] = 0.0
            self.cam_loc[0, 1] = -1.0
        self.cam_view = cam_view

        # Max objects on the table
        self.max_clean = max_clean
        self.max_dirty = max_dirty
        self.max_human = max_human
        self.max_tot_obj = self.max_clean + self.max_dirty + self.max_human

        # Max object types
        self.max_obj_types = ["c"] * self.max_clean + \
                             ["d"] * self.max_dirty + \
                             ["h"] * self.max_human
        self.max_types = np.array([1.0] * self.max_clean +
                                  [2.0] * self.max_dirty +
                                  [3.0] * self.max_human)

        # Generate objects
        self.rand_obj = rand_obj
        self._gen_obj()

        # Set rewards
        self._set_rewards()

        # Object width
        self.obj_width = obj_width

        # Generate Grid
        self._gen_grid()

        # Move and Remove locations
        self.move_loc = np.asarray([self.low-1, 0])
        self.remove_loc = np.asarray([self.high+1, 0])

        # Move and Remove action lists
        self.move_actions = []
        self.remove_actions = []

        # Observation and Action dimensions
        self.hist_len = hist_len

        self.action_space = Discrete(self.max_tot_obj * 2 + 1)
        if self.partial:
            self.observation_space = Discrete(4 * self.max_tot_obj * self.hist_len)
        else:
            self.observation_space = Discrete(4 * self.max_tot_obj)
        # print("History length:", self.observation_space.n)

        # Redundant action per episdoe counter
        self.redundant_actions = 0

        # Time limit
        self.t = 0
        if time_limit:
            self.t_lim = time_limit
        elif self.partial:
            self.t_lim = self.max_tot_obj * 4
        else:
            self.t_lim = self.max_tot_obj * 3

        # History
        self.history = deque(maxlen=self.hist_len)

        # Create plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def _set_rewards(self):
        """
        Set rewards
        """
        self.collision_penalty = -156
        self.time_limit_penalty = -40
        self.success_rew = 150

    def _gen_obj(self):
        """
        Generate objects on the table such that number1of objects less than max values
        """
        if self.rand_obj:
            c_lb = self.max_clean
            d_lb = self.max_dirty
            h_lb = self.max_human
            self.num_clean_dish = np.random.randint(1, self.max_clean+1)
            self.num_dirty_dish = np.random.randint(1, self.max_dirty+1)
            self.num_human = np.random.randint(1, self.max_human+1)
        else:
            self.num_clean_dish = self.max_clean
            self.num_dirty_dish = self.max_dirty
            self.num_human = self.max_human

        self.tot_obj = self.num_clean_dish + self.num_dirty_dish + self.num_human
        self.obj_types = ["c"] * self.num_clean_dish + \
                         ["d"] * self.num_dirty_dish + \
                         ["h"] * self.num_human
        self.types = np.array([1.0] * self.num_clean_dish +
                              [2.0] * self.num_dirty_dish +
                              [3.0] * self.num_human)

        # Set valid actions
        self.move_actions = np.arange(1, self.tot_obj+1)
        self.remove_actions = np.arange(self.max_tot_obj+1, self.max_tot_obj + self.tot_obj+1)
        # print(self.tot_obj, self.move_actions, self.remove_actions)


        assert self.move_actions.shape[0] == self.tot_obj
        assert self.remove_actions.shape[0] == self.tot_obj

    def _gen_pos(self):
        """
        :return: (float) x, y
        """
        mu = 0
        sigma = 0.3
        pos = np.random.normal(mu, sigma, size=2)
        while np.any(pos < np.asarray([self.low, self.low])) \
                or np.any(pos > np.asarray([self.high, self.high]))\
                or np.any(pos == self.grid[:, 2]):
            pos = np.random.normal(mu, sigma, size=2)
        return pos[0], pos[1]

    def _gen_grid(self):
        """
            Generate initial grid
            [x1, y1, in_view, type]
            [x2, y2, in_view, type]
                    :
            [xN, yN, in_view, type]

            in_view: 1 if in view else -1
            type:   1 clean, 2 dirty, 3 human, -2 unkown
        """
        self.grid = np.zeros((self.tot_obj, 4))
        self.oc_grid = np.zeros((self.tot_obj, 4))
        for obj_id in range(self.tot_obj):
            x, y = self._gen_pos()
            in_view = 1.0               # Fully visible grid. Mask it later during observation gneration
            if self.obj_types[obj_id] == "c":
                obj_type = 1.0
            elif self.obj_types[obj_id] == "d":
                obj_type = 2.0
            elif self.obj_types[obj_id] == "h":
                obj_type = 3.0
            else:
                obj_type = -2.0
            self.grid[obj_id] = np.asarray([x, y, in_view, obj_type])

    def _gen_occlusions(self):
        """
            Add occlusions to objects in the grid.
        """
        # self.oc_grid = self.grid.copy()
        self.oc_grid[:, 2] = 1.0
        # self.oc_grid[:, 3] = self.types
        c = np.squeeze(self.cam_loc)
        pos = self.oc_grid[:, :2]

        # Compute distances from Camera to each object. Obtain sorted indices, closest object to furthest
        dist = np.linalg.norm(self.cam_loc-pos, axis=1)
        sorted_indices = np.argsort(dist)

        # Lists of occluded objects, tangency points of each object
        occluded = []
        self.a_list = []
        self.b_list = []
        # print(sorted_indices)

        # For each object, check if the objects behind are occluded or not
        for count, i in enumerate(sorted_indices):
            # a, b, are (approximate) tangency points originating from camera till object periphery
            a = pos[i] + np.asarray([-self.obj_width, 0.0])
            b = pos[i] + np.asarray([ self.obj_width, 0.0])
            self.a_list.append(a)
            self.b_list.append(b)

            # check if each point is on the RIGHT of line (CA) and LEFT of line (CB)
            for _, j in enumerate(sorted_indices[count+1:]):
                p = pos[j]

                # Check if right of line (CA) and left of line (CB) i.e. camera -- point A
                # https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
                da = (p[0] - c[0]) * (a[1] - c[1]) - (p[1] - c[1]) * (a[0] - c[0])
                db = (p[0] - c[0]) * (b[1] - c[1]) - (p[1] - c[1]) * (b[0] - c[0])

                if da > 0:
                    # Point P is on the RIGHT of line (CA)
                    if db < 0:
                        # Point P is on the LEFT of the line (CB)
                        occluded.append(j)
                        # Set in_view to -1 and obj_type to unknown
                        # self.oc_grid[j, 2] = -1.0
                        # self.oc_grid[j, 3] = -2.0
                        self.oc_grid[j, :] = -10.0
                        # print("Changed oc grid:\n", self.oc_grid == self.grid)

        # self.grid = oc_grid

    def _gen_obs(self):
        """
        :return: Flattened observation as row-vector - 1x(4*total_objects)
        """
        self.oc_grid = self.grid.copy()
        if self.partial:
            self._gen_occlusions()

        # Add noise
        if self.noise:
            for _ in range(int(self.tot_obj/3)):
                id = np.random.randint(0, self.tot_obj)
                self.oc_grid[id, :] = -10

        # Add observation to history
        obs = self.oc_grid.reshape(1, -1)
        zeros_padding = -10 * np.ones((1, self.max_tot_obj * 4 - obs.shape[1]))
        obs = np.concatenate([obs, zeros_padding], axis=1)
        # plt.imshow(obs.reshape(-1, 4))
        # plt.show()

        # self.history.append(obs)
        self.history.appendleft(obs)

        # Return history if partial observability
        if self.partial:
            obs_hist = np.concatenate(np.asarray(self.history)).reshape(1, -1)
            zeros_padding = -10 * np.ones((1, self.observation_space.n - obs_hist.shape[1]))
            obs_hist = np.concatenate([obs_hist, zeros_padding], axis=1)
            assert obs_hist.shape[1] == self.observation_space.n, "History not same shape as observation_space"

            return obs_hist
        else:
            return obs

    def _get_obj_type(self, obj_id):
        """
        :param obj_id: Index of object
        :return: (string) Object type i.e. "c", "d", "h"
        """
        return self.obj_types[obj_id]

    def update_human_pos(self):
        """
            Add dynamic motion for human
        """
        # TODO: Ensure human stays inside the table
        human_pos = self.grid[-self.num_human:, :2]
        max_dx = 0.1
        dpos = np.random.uniform(-max_dx, max_dx, size=(self.num_human, 2))
        human_pos = human_pos + dpos
        self.grid[-self.num_human:, :2] = human_pos

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        """
            Plot grid.
        """
        plt.cla()
        # Create variables for plotting
        visible = self.grid[self.oc_grid[:, 2] == 1]

        # Generate clean, dirty and humans which are visible
        clean = visible[visible[:, -1] == 1][:, :2]
        dirty = visible[visible[:, -1] == 2][:, :2]
        human = visible[visible[:, -1] == 3][:, :2]

        # Generate occlusions
        occluded = self.grid[self.oc_grid[:, 2] == -10][:, :2]
        # Arrays for plotting
        data = [clean, dirty, human, occluded, self.cam_loc]
        colors = ["blue", "green", "red", "black", "black"]
        groups = ["Clean", "Dirty", "Human", "Occlusion", "Camera"]
        marker = ["o", "o", "v", "p", "s"]

        for d, color, group, m in zip(data, colors, groups, marker):
            x = d[:, 0]
            y = d[:, 1]
            self.ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=750, label=group, marker=m)

        # Plot lines showing view-angles
        if self.cam_view:
            cx = np.squeeze(self.cam_loc)[0]
            cy = np.squeeze(self.cam_loc)[1]
            for i in range(len(self.a_list)):
                plt.plot([cx, self.a_list[i][0]], [cy, self.a_list[i][1]], "b-.")
                plt.plot([cx, self.b_list[i][0]], [cy, self.b_list[i][1]], "b-.")

        plt.title("Continuous Table Environment")
        plt.ylim(bottom=self.low, top=self.high)
        plt.xlim(left=self.low-1.2, right=self.high+1.2)
        plt.axvline(x=self.high)
        plt.axvline(x=self.low)
        plt.tight_layout()
        plt.legend(loc=1, markerscale=0.5)
        plt.grid(False)
        plt.pause(0.5)
        plt.show(block=False)

    def reset(self):
        """
            Reset environment
        :return: Observation (see self._gen_obs() )
        """
        # if self.t:
        #     print("Uselss action percentage: %.1f " % (self.redundant_actions/self.t*100))
        # Reset
        self.redundant_actions = 0
        self.t = 0
        self._gen_obj()
        self._gen_grid()
        return self._gen_obs()

    def step(self, action):
        """
            Actions:
            0:              Done
            1:              Do nothing
            2 to N+1:       Move object (self.move_loc)
            N+2 to 2N+1:    Remove object (self.remove_loc)

        :param action: (int) Action to take on the grid
        :return: Observation, Reward, Done, Info
        """

        self.t += 1
        # print(self.t)
        info = None
        done = False
        rew = 0.0
        if action == 0:     # Done action
            # Get indexes of each object
            removed_obj_idx = np.nonzero(self.grid[:, 0] == self.remove_loc[0])
            removed_obj_types = self.types[removed_obj_idx]

            # Compute number of removed clean, dirty and humans
            num_c = np.sum(removed_obj_types == 1)
            num_d = np.sum(removed_obj_types == 2)
            num_h = np.sum(removed_obj_types == 3)
            if num_h:
                rew = self.collision_penalty
                info = 3
                done = True
            elif num_d == self.num_dirty_dish and num_c == 0 and num_h == 0:
                rew = self.success_rew
                info = 1
                done = True
            else:
                rew = -30

        # elif action == 1:    # Do nothing action
        #     self.do_nothing_count += 1
        #     rew = -1

        # elif action <= self.tot_obj:     # Move object
        elif np.isin(action, self.move_actions):
            obj_id = action - 1
            obj_type = self._get_obj_type(obj_id)

            # Has object already been removed?
            if (self.grid[obj_id, 0] == self.remove_loc[0]).all():
                rew = -7
            elif obj_type == "h":     # Collision!
                rew = self.collision_penalty
                info = 3
                done = True
            else:       # Moving dish
                rew = -1
                if (self.grid[obj_id, 0:2] == self.move_loc).all(): # Move from move_loc to table
                    x, y = self._gen_pos()
                    self.grid[obj_id, 0:2] = (x, y)
                else:
                    self.grid[obj_id, 0:2] = self.move_loc          # Move from table to move_loc

        # elif action <= 2 * self.tot_obj:     # Remove object from table
        elif np.isin(action, self.remove_actions):
            obj_id = action - self.max_tot_obj - 1
            obj_type = self._get_obj_type(obj_id)
            # print("Removing object with ID: ", obj_id)

            if (self.grid[obj_id, 0:2] == self.remove_loc).all():  # Has object already been removed?
                rew = -7
            elif obj_type == "h":     # Collision!
                rew = self.collision_penalty
                info = 3
                done = True
            elif obj_type == "d":   # Remove dirty object
                rew = 17
                self.grid[obj_id, 0:2] = self.remove_loc
            else:                   # Remove clean object
                rew = -11
                self.grid[obj_id, 0:2] = self.remove_loc
        else:
            self.redundant_actions += 1
            rew = -50
            # print("Useless action: ", action, self.tot_obj)

        # Update human position
        self.update_human_pos()

        # Randomly add an object to the scene
        if np.random.rand() > 0.7:

        # Randomly remove an object from the scene

        # Test if time has exceeded limit
        if self.t > self.t_lim:
            rew = self.time_limit_penalty
            info = 2
            done = True

        obs = self._gen_obs()
        return obs, rew, done, info


if __name__ == "__main__":
    env = ContinuousTable()
    env.reset()
    env.render()
    act_dim = env.action_space.n
    obs_dim = env.observation_space.n
    ep_rew = 0.0
    ep_count = 0
    for _ in range(100):
        env.render()
        a = np.random.randint(act_dim)
        obs, rew, done, info = env.step(a)
        # print("Reward: ", rew)
        ep_rew += rew
        if done:
            ep_count += 1
            print("Episode reward: ", ep_rew)
            ep_rew = 0
            env.reset()
            continue

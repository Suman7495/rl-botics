import numpy as np
import random
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from collections import deque

class ContinuousTable:
    """
        Table Environment with clean and dirty dishes during Human Robot Collaboration
    """
    def __init__(self,
                 sizes=None,
                 n_clean=3,
                 n_dirty=2,
                 n_human=1,
                 obj_width=0.1,
                 cam_loc=None,
                 partial=True,
                 noise=False,
                 cam_view=False,
                 static=False,
                 time_limit=None,
                 hist_len=5
                 ):
        """
        :param sizes:       Array of sizes of the grid such that: low <= x < high, low < y < high
        :param n_clean:     (int) Number of clean objects
        :param n_dirty:     (int) Number of dirty objects
        :param n_human:     (int) Number of humans
        :param cam_loc:     (Numpy array of shape (1,2) ) Camera location (fixed)
        :param partial:     (Boolean) True if POMDP, else, MDP
        :param noise:       (Boolean) True if noisy observation
        :param cam_view:    (Boolean) True to plot camera view
        :param obj_width:   (Float) Width of each object. Currently, all objects have same width
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

        # Number of dishes and humans
        self.num_clean_dish = n_clean
        self.num_dirty_dish = n_dirty
        self.num_human = n_human
        self.tot_obj = self.num_clean_dish + self.num_dirty_dish + self.num_human
        self.obj_types = ["c"] * self.num_clean_dish + \
                        ["d"] * self.num_dirty_dish + \
                        ["h"] * self.num_human
        self.obj_width = obj_width

        # Generate Grid
        self.grid = np.zeros((self.tot_obj, 4))
        self._gen_grid()
        self.static = static
        if self.static:
            self.init_grid = self.grid

        # Move and Remove locations
        self.move_loc = np.asarray([self.low-1, 0])
        self.remove_loc = np.asarray([self.high+1, 0])

        # Observation and Action dimensions
        self.hist_len = hist_len
        self.action_space = Discrete(self.tot_obj * 2 + 2)
        if self.partial:
            self.observation_space = Discrete(4 * self.tot_obj * self.hist_len)
        else:
            self.observation_space = Discrete(4 * self.tot_obj)

        # Time limit
        self.t = 0
        if time_limit:
            self.t_lim = time_limit
        elif self.partial:
            self.t_lim = self.tot_obj * 4
        else:
            self.t_lim = self.tot_obj * 3

        # History
        self.history = deque(maxlen=self.hist_len)

        # Create plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def _gen_pos(self):
        """
        :return: (float) x, y
        """
        #TODO: Ensure position is not taken
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
        """
        for obj_id in range(self.tot_obj):
            x, y = self._gen_pos()
            in_view = 1.0               # Fully visible grid. Mask it later during observation gneration
            if self.obj_types[obj_id] == "c":
                type = 1.0
            elif self.obj_types[obj_id] == "d":
                type = -1.0
            else:
                type = 0.0
            self.grid[obj_id] = np.asarray([x, y, in_view, type])

    def _gen_occlusions(self):
        """
            Add occlusions to objects in the grid.
        """
        oc_grid = self.grid
        oc_grid[:, 2] = 1.0
        c = np.squeeze(self.cam_loc)
        pos = oc_grid[:, :2]

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
                        oc_grid[j, 2] = -1.0

        self.grid = oc_grid

    def _gen_obs(self):
        """
        :return: Flattened observation as row-vector - 1x(4*total_objects)
        """
        if self.partial:
            self._gen_occlusions()
            obs = self.grid.reshape(1, -1)
        else:
            # Return full grid as a row vector 1x(Rows*Cols)
            obs = self.grid.reshape(1, -1)

        if self.noise:
            # Mask observation with noise
            mask = np.random.randint(0, 2, size=obs.shape).astype(np.bool)
            noise = np.random.randint(0, self.tot_obj, size=obs.shape)
            obs[mask] = noise[mask]

        # Add observation to history
        self.history.append(obs)
        if self.partial:
            obs_hist = np.concatenate(np.asarray(self.history)).reshape(1, -1)
            zeros_padding = np.zeros((1, self.observation_space.n - obs_hist.shape[1]))
            obs_hist = np.concatenate([obs_hist, zeros_padding], axis=1)
            assert obs_hist.shape[1]==self.observation_space.n
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
        # print("Grid:\n ", self.grid, "\n")
        plt.cla()
        # Create variables for plotting
        visible = self.grid[self.grid[:, 2] == 1]
        # print("Visible:\n ", visible, "\n")

        # Generate clean, dirty and humans which are visible
        clean = visible[visible[:, -1] ==  1][:, :2]
        dirty = visible[visible[:, -1] == -1][:, :2]
        human = visible[visible[:, -1] ==  0][:, :2]

        # Generate occlusions
        occluded = self.grid[self.grid[:, 2] == -1][:, :2]
        # print("Occludeed:\n ", occluded)

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
        plt.pause(0.06)
        plt.show(block=False)

    def reset(self):
        """
            Reset environment
        :return: Observation (see self._gen_obs() )
        """
        self.t = 0
        if self.static:
            self.grid = self.init_grid
        else:
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
        info = None
        done = False
        rew = 0.0
        if action == 0:     # Done action
            # Get indexes of each object
            removed_obj_idx = np.nonzero(self.grid[:, 0] == self.remove_loc[0])
            removed_obj_types = self.grid[removed_obj_idx, -1]
            num_c = np.sum(removed_obj_types == 1)
            num_d = np.sum(removed_obj_types == -1)
            num_h = np.sum(removed_obj_types == 0)
            if num_h:
                info = 3
                rew = -56

            # Compute reward wrt to the remaining objects
            cost1 = -32 * num_c                              # Removed clean dishes (-)
            cost2 =  18 * num_d                              # Removed dirty dishes (+)
            cost3 =  3  * (self.num_clean_dish - num_c)      # Remaining clean dishes (+)
            cost4 = -25 * (self.num_dirty_dish - num_d)      # Remaining dirty dishes (-)
            rew = cost1 + cost2 + cost3 + cost4
            if num_d == self.num_dirty_dish and num_c == 0 and num_h == 0:
                rew = 100
                info = 1
                done = True
            else:
                rew = -30
                # print("Number of clean/ dirty dishes removed: ", num_c, num_h)

        elif action == 1:    # Do nothing action
            rew = -1

        elif action <= self.tot_obj + 1:     # Move object
            obj_id = action - 2
            obj_type = self._get_obj_type(obj_id)

            # Has object already been removed?
            if (self.grid[obj_id, 0] == self.remove_loc[0]).all():
                rew = -7
            elif obj_type == "h":     # Collision!
                rew = -56
                done = True
                info = 3

            else:       # Moving clean dish
                rew = -1
                if (self.grid[obj_id, 0:2] == self.move_loc).all():
                    x, y = self._gen_pos()
                    self.grid[obj_id, 0:2] = (x, y)
                else:
                    self.grid[obj_id, 0:2] = self.move_loc

        elif action <= 2 * self.tot_obj + 1:     # Remove object from table
            obj_id = action - self.tot_obj - 2
            obj_type = self._get_obj_type(obj_id)

            if (self.grid[obj_id, 0:2] == self.remove_loc).all():  # Has object already been removed?
                rew = -7
            elif obj_type == "h":     # Collision!
                rew = -56
                done = True
                info = 3
            elif obj_type == "d":   # Remove dirty object
                rew = 17
                self.grid[obj_id, 0:2] = self.remove_loc
            else:                   # Remove clean object
                rew = -11
                self.grid[obj_id, 0:2] = self.remove_loc
        else:
            print("Unrecognized action: ", action)

        # Update human position
        self.update_human_pos()

        # Test if time has exceeded limit
        if self.t > self.t_lim:
            done = True
            rew = -15
            info = 2

        obs = self._gen_obs()
        return obs, rew, done, info


if __name__ == "__main__":
    env = ContinuousTable()
    env.reset()
    env.render()
    act_dim = env.action_space.n
    obs_dim = env.observation_space.n
    ep_rew = 0.0
    for _ in range(100):
        env.render()
        a = np.random.randint(act_dim)
        obs, rew, done, info = env.step(a)
        # print("Reward: ", rew)
        ep_rew += rew
        if done:
           print("Episode reward: ", ep_rew)
           ep_rew = 0
           env.reset()
           continue

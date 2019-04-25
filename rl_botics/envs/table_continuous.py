import numpy as np
import random
import matplotlib.pyplot as plt

class ContinuousTable():
    """
        Table Environment with clean and dirty dishes during Human Robot Collaboration
    """
    def __init__(self, sizes = None,
                 n_clean=3,
                 n_dirty=2,
                 n_human=1,
                 cam_loc=None,
                 partial=True,
                 noise=False,
                 cam_view=True,
                 obj_width=0.2
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
        self.obj_list = [""] + ["c"] * self.num_clean_dish + ["d"] * self.num_dirty_dish + ["h"] * self.num_human
        self.obj_width = obj_width

        # Observation and Action dimensions
        # TODO: add observation and action dimensions (if partial 9, else row*col)

        # Generate Grid
        self.grid = np.zeros((self.tot_obj, 4))
        self._gen_grid()

    def _gen_pos(self):
        """
        :return: (float) x, y
        """
        pos = np.random.normal(size=2)
        while np.any(pos < np.asarray([self.low, self.low])) or np.any(pos > np.asarray([self.high, self.high])):
            pos = np.random.normal(size=2)
        return pos[0], pos[1]

    def _gen_grid(self):
        """
            Generate initial grid
        """
        for i in range(self.tot_obj):
            x, y = self._gen_pos()
            in_view = 1.0               # Fully visible grid. Mask it later during observation gneration
            if self.obj_list[i+1] == "c":
                type = 1.0
            elif self.obj_list[i+1] == "d":
                type = -1.0
            else:
                type = 0.0
            self.grid[i] = np.asarray([x, y, in_view, type])

        self.init_grid = self.grid
        self._gen_occlusions()
        self.print()


    def _gen_occlusions(self):
        """
            Add occlusions to objects in the grid.
        """
        oc_grid = self.grid
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
            # Obtain a 3x3 square camera view around camera position, reshape to row vector 1x9
            self._gen_occlusions()
            obs = self.grid.reshape(1, -1)
        else:
            # Return full grid as a row vector 1x(Rows*Cols)
            obs = self.grid.reshape(1, -1)

        if self.noise:
            # Mask observation with noise
            mask = np.random.randint(0,2,size=obs.shape).astype(np.bool)
            noise = np.random.randint(0, self.tot_obj, size=obs.shape)
            obs[mask] = noise[mask]
        return obs

    def _get_obj_type(self, obj_id):
        """
        :param obj_id: Index of object
        :return: (string) Object type e.g. "c", "d", "h"
        """
        return self.obj_list[obj_id]

    def print(self):
        """
            Plot grid.
        """
        print("Grid:\n ", self.grid, "\n")

        # Create variables for plotting
        visible = self.grid[self.grid[:, 2] == 1]
        print("Visible:\n ", visible, "\n")

        # Generate clean, dirty and humans which are visible
        clean = visible[visible[:, -1] ==  1][:, :2]
        dirty = visible[visible[:, -1] == -1][:, :2]
        human = visible[visible[:, -1] ==  0][:, :2]

        # Generate occlusions
        occluded = self.grid[self.grid[:, 2] == -1][:, :2]
        print("Occludeed:\n ", occluded)

        data = [clean, dirty, human, occluded, self.cam_loc]
        colors = ["blue", "green", "red", "black", "black"]
        groups = ["Clean", "Dirty", "Human", "Occlusion", "Camera"]
        marker = ["o", "o", "v", "o","s"]

        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for d, color, group, m in zip(data, colors, groups, marker):
            x = d[:, 0]
            y = d[:, 1]
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=150, label=group, marker=m)

        # Plot lines showing view-angles
        if self.cam_view:
            cx = np.squeeze(self.cam_loc)[0]
            cy = np.squeeze(self.cam_loc)[1]
            for i in range(len(self.a_list)):
                plt.plot([cx, self.a_list[i][0]], [cy, self.a_list[i][1]], "b-.")
                plt.plot([cx, self.b_list[i][0]], [cy, self.b_list[i][1]], "b-.")

        plt.title("Continous Table Environment")
        plt.ylim(bottom=self.low, top=self.high)
        plt.xlim(left=self.low, right=self.high)
        plt.legend()
        # plt.grid()
        plt.show(block=False)
        print("It works!")

    def reset(self):
        """
            Reset environment
        :return: Observation (see self._gen_obs() )
        """
        self.grid = self.init_grid
        return self._gen_obs()

    def step(self, action):
        """
        :param action: (int) Action to take on the grid
        :return: Observation, Reward, Done, Info - like OpenAI Gym
        """
        # TODO: Complete
        done = False
        rew = 0.0
        if action == 0:
            done = True

            # Get indexes of each object
            range_c = np.asarray(range(1,self.num_clean_dish+1))
            range_d = np.asarray(range(self.num_clean_dish+1, self.num_clean_dish+self.num_dirty_dish+1))
            range_h = np.asarray(range(self.tot_obj-self.num_human+1, self.tot_obj+1))

            # Number of remaining objects
            num_c = len(np.intersect1d(self.grid, range_c))
            num_d = len(np.intersect1d(self.grid, range_d))
            num_h = len(np.intersect1d(self.grid, range_h))

            # Compute reward wrt to the remaining objects
            cost1 =  20 * num_c                              # Remaining clean dishes (+)
            cost2 = -10 * num_d                              # Remaining dirty dishes (-)
            cost3 = - 5 * (self.num_clean_dish - num_c)      # Clean dishes in the wash (-)
            cost4 =  25 * (self.num_dirty_dish - num_d)      # Dirty dishes in the wash (+)
            rew = cost1 + cost2 + cost3 + cost4

        elif action < self.tot_obj + 1:
            # Remove object from table
            junk = 0

        elif action < 2 * self.tot_obj + 1:
            # Move object to the side of the table
            junk = 0
        else:
            print("Unrecognized action")

        obs = self._gen_obs()
        info = None
        return obs, rew, done, info

    def render(self):
        self.print()


env = ContinuousTable()
plt.show()

#
# env.reset()
# env.render()
# ep_rew = 0.0
# for _ in range(10):
#     # env.render()
#     a = np.random.randint(5)
#     obs, rew, done, info = env.step(a)
#     print("Reward: ", rew)
#     ep_rew += rew
#     if done:
#        # print("Episode reward: ". ep_rew)
#        ep_rew = 0
#        continue
# env.render()
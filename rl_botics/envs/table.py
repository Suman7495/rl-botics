import numpy as np
import random


class Table():
    """
        Table Environment with clean and dirty dishes during Human Robot Collaboration
    """
    def __init__(self, sizes, n_clean=3, n_dirty=2, n_human=1, cam_pos=(2, 2), partial=False, noise=False):
        self.rows = sizes[0]
        self.cols = sizes[1]

        # Partial Observability and Noisy Sensor Reading
        self.partial = partial
        self.noise = noise

        # Positions of Camera
        self.init_cam_pos = np.asarray(cam_pos)
        self.cam_pos = self.init_cam_pos

        # Restrict Camera Motion
        self.cam_row_bound = self.rows - 1
        self.cam_col_bound = self.cols - 1

        # Number of dishes and humans
        self.num_clean_dish = n_clean
        self.num_dirty_dish = n_dirty
        self.num_human = n_human
        self.tot_obj = self.num_clean_dish + self.num_dirty_dish + self.num_human
        self.obj_list = [""] + ["c"] * self.num_clean_dish + ["d"] * self.num_dirty_dish + ["h"] * self.num_human

        # Observation and Action dimensions
        # TODO: add observation and action dimensions (if partial 9, else row*col)

        # Generate Grid
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self._gen_grid()

    def _gen_pos(self):
        r = random.randint(0, self.rows - 1)
        c = random.randint(0, self.cols - 1)
        return r, c

    def _gen_grid(self):
        for i in range(self.tot_obj):
            r, c = self._gen_pos()
            while self.grid[r][c]:
                r, c = self._gen_pos()
            self.grid[r][c] = i + 1

        assert np.count_nonzero(self.grid) == self.tot_obj, "All objects failed to be placed on table."

        self.init_grid = self.grid
        self._gen_obs()

    def _gen_obs(self):
        if self.partial:
            # Obtain a 3x3 square camera view around camera position, reshape to row vector 1x9
            r = self.cam_pos[0]
            c = self.cam_pos[1]
            obs = np.squeeze(self.grid[r-1:r+2, c-1:c+2].reshape(1,-1))
        else:
            # Return full grid as a row vector 1x(Rows*Cols)
            obs = self.grid.reshape(1,-1)

        if self.noise:
            # Mask observation with noise
            mask = np.random.randint(0,2,size=obs.shape).astype(np.bool)
            noise = np.random.randint(0, self.tot_obj, size=obs.shape)
            obs[mask] = noise[mask]
        return obs

    def _get_obj_type(self, obj_id):
        return self.obj_list[obj_id]

    def print(self):
        print(self.grid)

    def reset(self):
        self.grid = self.init_grid
        self.cam_pos = self.init_cam_pos
        return self._gen_obs()

    def step(self, action):
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

        elif action == 1:
            # Move camera up (row-1)
            if self.cam_pos[0] > 1:
                self.cam_pos[0] -= 1
                rew = -1.0
            else:
                rew = 5.0  # punish to attempt to move camera outside boundary

        elif action == 2:
            # Move camera down (row+1)
            if self.cam_pos[0] < self.cam_row_bound:
                self.cam_pos[0] += 1
                rew = -1.0
            else:
                rew = -5.0  # punish to attempt to move camera outside boundary

        elif action == 3:
            # Move camera left (col-1)
            if self.cam_pos[1] > 1:
                self.cam_pos[1] -= 1
                rew = -1.0
            else:
                rew = -5.0  # punish to attempt to move camera outside boundary

        elif action == 4:
            # Move camera right (col+1)
            if self.cam_pos[1] < self.cam_col_bound:
                self.cam_pos[1] += 1
                rew = -1.0
            else:
                rew -= 5.0  # punish to attempt to move camera outside boundary
        elif action < self.tot_obj + 5:
            # Remove object from table
            obj_id = action - 4
            obj_type = self._get_obj_type(obj_id)
            idx = np.where(self.grid==obj_id)
            r = np.squeeze(idx[0])
            c = np.squeeze(idx[1])
            self.grid[r][c] = 0

            # Generate reward depending on removed object type
            if obj_type == 'c':
                rew = -10
            elif obj_type == 'd':
                rew = 10
            elif obj_type == 'h':
                rew = -50
                done = True
            else:
                print("Unrecognized object type. Cannot remove object.")
        else:
            print("Unrecognized action")

        obs = self._gen_obs()
        info = None
        return obs, rew, done, info

    def render(self):
        self.print()


env = Table([5, 5], partial=True, noise=False)
env.reset()
env.render()
ep_rew = 0.0
for _ in range(10):
    # env.render()
    a = np.random.randint(5)
    obs, rew, done, info = env.step(a)
    print("Reward: ", rew)
    ep_rew += rew
    if done:
       # print("Episode reward: ". ep_rew)
       ep_rew = 0
       continue
env.render()
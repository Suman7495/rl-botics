import numpy as np
import random


class Table():
    """
        Table Environment with clean and dirty dishes during Human Robot Collaboration
    """
    def __init__(self, sizes, n_clean=3, n_dirty=2, n_human=1, cam_pos=(2, 2), partial=False):
        self.rows = sizes[0]
        self.cols = sizes[1]
        self.partial = partial

        # Number of dishes and humans
        self.num_clean_dish = n_clean
        self.num_dirty_dish = n_dirty
        self.num_human = n_human
        self.tot_obj = self.num_clean_dish + self.num_dirty_dish + self.num_human

        # Positions of Camera
        self.init_cam_pos = np.asarray(cam_pos)
        self.cam_pos = self.init_cam_pos

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
        r = self.cam_pos[0]
        c = self.cam_pos[1]
        obs = np.squeeze(self.grid[r-1:r+2, c-1:c+2].reshape(1,-1))
        print("Noiseless observation: ", obs)
        if self.partial:
            mask = np.random.randint(0,2,size=obs.shape).astype(np.bool)
            noise = np.random.randint(0, self.tot_obj, size=obs.shape)
            obs[mask] = noise[mask]
        print("Noisy observation : ", obs)
        return obs

    def _gen_rew(self):
        return NotImplemented

    def print(self):
        print(self.grid)

    def reset(self):
        self.grid = self.init_grid
        self.cam_pos = self.init_cam_pos
        return self._gen_obs()

    def step(self, action):
        # TODO: Complete
        return NotImplemented

    def render(self):
        self.print()


table = Table([5, 5], partial=True)
table.print()
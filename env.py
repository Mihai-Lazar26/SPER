"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (0, 50)
        self.y_range = (0, 30)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        # [x, y, w, h]
        obs_boundary = [
            [0, 0, 0.25, 30],
            [0, 30, 50, 0.25],
            [0.25, 0, 50, 0.25],
            [50, 0.25, 0.25, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # [x, y, w, h]
        obs_rectangle = [
            [14, 12, 2, 8],
            [18, 22, 8, 3],
            [22, 4, 5, 12],
            [32, 14, 2, 10],
            [40, 17, 7, 1]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        # [x, y, r]
        obs_cir = [
            [7, 12, 4.5],
            [15, 5, 2],
            [37, 7, 3],
            [40, 26, 3]
        ]

        return obs_cir

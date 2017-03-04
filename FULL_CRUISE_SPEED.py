import numpy as np

MINSPEED = 20
MAXSPEED = 80
SPEEDSIZE = 10




class CRUISE_SPEED(object):


    def function(self, timestep_length):
        self.min_speed = MINSPEED
        self.max_speed = MAXSPEED
        self.timestep_length = timestep_length
        self.speed_size = SPEEDSIZE
        self.speed_seq = np.array(range(self.min_speed, self.max_speed, self.speed_size))# [20,30,40,50,60,70]
        self.random_index = np.random.random_integers(0, np.size(self.speed_seq)-1)
        return self.speed_seq[self.random_index]/100.0

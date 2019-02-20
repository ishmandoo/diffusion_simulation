import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler
import random
from enum import Enum

class BC(Enum):
    REFLECT = 1
    WRAP = 2
    ELECTRODE = 3


class Simulator:
    def __init__(self, 
        n=10000, 
        name="unnamed", 
        dt = 1., 
        alpha=0, 
        w=1000, 
        bc=BC.WRAP, 
        getDiffusivity=lambda x: 1.+x, 
        count_flux=False,
        electrode_absorb_prob=1.):

        self.name = name
        self.alpha = alpha
        self.w = w
        self.w2 = self.w/2

        
        self.bc = bc
        self.electrode_absorb_prob = electrode_absorb_prob

        self.getDiffusivity = getDiffusivity
        self.diffusivities = getDiffusivity(np.linspace(0,1))

        self.n = n
        self.n_steps = 0
        self.dt = dt
        self.sqrt_dt = np.sqrt(self.dt)

        self.n_neg_exit = 0
        self.n_pos_exit = 0

        self.n_neg_reflect = 0
        self.n_pos_reflect = 0


        self.count_flux = count_flux
        self.n_pos_center_cross = 0
        self.n_neg_center_cross = 0

        self.xs = None
        self.pos_exits = np.repeat(0,self.n)
        self.neg_exits = np.repeat(0,self.n)

        self.pos_reflections = np.repeat(0,self.n)
        self.neg_reflections = np.repeat(0,self.n)

        self.absorb_rands = np.ones(self.n, dtype=bool)
        self.not_absorb_rands = np.zeros(self.n, dtype=bool)

    def sanitize(self):
        self.getDiffusivity = None

    def resetFluxes(self):
        self.n_neg_exit = 0
        self.n_pos_exit = 0
        self.n_pos_center_cross = 0
        self.n_neg_center_cross = 0


    def setStartingXsDelta(self, x0):
        self.xs = np.repeat(x0*self.w,self.n)

    def setStartingXsFlat(self):
        self.xs =  np.linspace(0.,self.w,self.n)

    def setStartingXsNormal(self, x0, sigma):
        self.xs =  np.random.normal(x0*self.w, scale=sigma*self.w, size=self.n)


    def stepNum(self):
        self.n_steps += 1
        # calculate random direction choices
        rands = np.random.choice([1.,-1.], size=self.n)

        # calculate differentials for gradients near each particle
        dx = 0.01
        diffusivities = self.getDiffusivity(self.xs/self.w)
        diffusivity_pos = self.getDiffusivity(self.xs/self.w + dx)
        diffusivity_neg = self.getDiffusivity(self.xs/self.w - dx)

        # calculate gradients for each particle
        diffusivity_gradients = (diffusivity_pos - diffusivity_neg) / (2 *dx * self.w)

        # calculate new position for each particle
        new_xs = self.xs + (self.alpha * diffusivity_gradients * self.dt) + (rands * np.sqrt(2*diffusivities) * self.sqrt_dt)
 
        # if counting flux, add up the number that crossed the mid point
        if self.count_flux:
            self.n_pos_center_cross += np.sum((self.xs < self.w2)&(new_xs > self.w2))
            self.n_neg_center_cross += np.sum((self.xs > self.w2)&(new_xs < self.w2))

        # find the indices of particles that exited the domain
        exit_pos_indices = (new_xs > self.w)
        exit_neg_indices = (new_xs < 0.)

        # unless perfect absorption, calculate random choices for whether the particle was absorbed by the boundary
        # absorb_rands default to all true
        if self.electrode_absorb_prob < 1.:
            self.absorb_rands = np.random.rand(self.n) < self.electrode_absorb_prob
            self.not_absorb_rands = np.logical_not(self.absorb_rands)
        
        # apply these random choices to find the indices of the particles that are absorbed by the domain...
        exit_pos_and_absorb_indices = exit_pos_indices & self.absorb_rands        
        exit_neg_and_absorb_indices = exit_neg_indices & self.absorb_rands
 
        # ...and the particles that are reflected
        exit_pos_and_reflect_indices = exit_pos_indices & self.not_absorb_rands
        exit_neg_and_reflect_indices = exit_neg_indices & self.not_absorb_rands

        # use the indices to calculate the number that were absorbed
        self.n_pos_exit += np.sum(exit_pos_and_absorb_indices)
        self.n_neg_exit += np.sum(exit_neg_and_absorb_indices)

        # use the indices to calculate the number that were reflected
        self.n_pos_reflect += np.sum(exit_pos_and_reflect_indices)
        self.n_neg_reflect += np.sum(exit_neg_and_reflect_indices)
        
        # use the indices to keep track of how many times each particle has been absorbed
        self.pos_exits[exit_pos_and_absorb_indices] += 1
        self.neg_exits[exit_neg_and_absorb_indices] += 1

        self.pos_reflections[exit_pos_and_reflect_indices] += 1
        self.neg_reflections[exit_neg_and_reflect_indices] += 1

        # for electrode bc, transport particles
        if self.bc is BC.ELECTRODE:
            # to the opposite boundary if aborbed
            new_xs[exit_pos_and_absorb_indices] = 0.
            new_xs[exit_neg_and_absorb_indices] = self.w
            
            if self.electrode_absorb_prob < 1.: # no need to do this for perfect absorption
                # to the same boundary if reflected
                new_xs[exit_pos_and_reflect_indices] = self.w
                new_xs[exit_neg_and_reflect_indices] = 0.
        
        # not checked for correctness
        if self.bc is BC.WRAP:
            new_xs[exit_pos_and_absorb_indices] = new_xs[exit_pos_and_absorb_indices] - self.w
            new_xs[exit_neg_and_absorb_indices] = self.w + new_xs[exit_neg_and_absorb_indices]

            if self.electrode_absorb_prob < 1.:
                new_xs[exit_pos_and_reflect_indices] = self.w - (new_xs[exit_pos_and_reflect_indices] - self.w)
                new_xs[exit_neg_and_reflect_indices] = - new_xs[exit_neg_and_reflect_indices]
        

        if self.bc is BC.REFLECT:
            new_xs[exit_pos_indices] = self.w - (new_xs[exit_pos_indices] - self.w)
            new_xs[exit_neg_indices] = -new_xs[exit_neg_indices]     

        self.xs = new_xs

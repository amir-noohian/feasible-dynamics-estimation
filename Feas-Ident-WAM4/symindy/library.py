import numpy as np
from deap import gp


class Library:
    def __init__(self, nc, dimensions, library_name="generalized", h = 100):
        self.nc = nc
        self.dimensions = dimensions
        self.library_name = library_name
        self.h = h

    def create_pset(self):
        size_input = self.dimensions + self.nc
        # TODO let the dimensionality be a function of an input file
        intypes = [float for i in range(size_input)]
        # 1)name, 2)type of each input, 3)type of the output
        pset = gp.PrimitiveSetTyped("MAIN", intypes, float)
        self.pset = pset

    def polynomial_library(self):
        self.pset.addPrimitive(np.multiply, [float, float], float, name="mul")
        self.pset.addPrimitive(np.add, [float, float], float, name="add")

    def fourier_library(self):
        self.pset.addPrimitive(np.sin, [float], float, name="sin")
        self.pset.addPrimitive(np.cos, [float], float, name="cos")

    def fourier_f1(self, q):
        return (4/np.pi) * np.sin(1 * self.h * q) / 1

    def fourier_f2(self, q):
        return (4/np.pi) * np.sin(3 * self.h * q) / 3

    def fourier_f3(self, q):
        return (4/np.pi) * np.sin(5 * self.h * q) / 5

    def fourier_f4(self, q):
        return (4/np.pi) * np.sin(7 * self.h * q) / 7

    def fourier_f5(self, q):
        return (4/np.pi) * np.sin(9 * self.h * q) / 9
    
    def sin_k(x, k):
        return np.sin(k * x)
    
    def arctan_k(x,k):
        return np.arctan(x*k)
    

    def advanced_wam_model(self):
        ''' 
        Advanced WAM Model from 
        'A Friction-model-based Framework for Reinforcement Learning 
        of Robotic Tasks in Non-rigid Environments'
        Adria Colome, Antoni Planells and Carme Torras'''
        self.polynomial_library()

        self.pset.addPrimitive(np.sign, [float], float, name='sign')
        self.pset.addPrimitive(np.abs, [float], float, name='abs')
        self.pset.addPrimitive(self.arctan_k, [float, float], float, name='arctan_with_scale')

        self.pset.addPrimitive(self.fourier_f1, [float], float, name='fourier_f1')
        self.pset.addPrimitive(self.fourier_f2, [float], float, name='fourier_f2')
        self.pset.addPrimitive(self.fourier_f3, [float], float, name='fourier_f3')
        self.pset.addPrimitive(self.fourier_f4, [float], float, name='fourier_f4')
        self.pset.addPrimitive(self.fourier_f5, [float], float, name='fourier_f5')


    def generalized_library(self):
        self.polynomial_library()
        self.fourier_library()
        # call all the libraries

    def __call__(self):
        self.create_pset()
        if self.library_name == "polynomial":
            self.polynomial_library()
        elif self.library_name == "fourier":
            self.fourier_library()
        elif self.library_name == "generalized":
            self.generalized_library()
        elif self.library_name == 'advanced_wam_model':
            self.advanced_wam_model()
        return self.pset

import os, sys
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import re

class Cuboid():
    def __init__(self, *properties, **kwargs):
        self.properties = self.convert_data_to_mpl(properties)
        self.kwargs = kwargs


    def convert_data_to_mpl(self, properties):
        """to matplotlib.patches.Rectangle(xy, width, height, angle=0.0, **kwargs)[source]
           from <lower corner x y z> <upper corner x y z>"""
        xy = (properties[0], properties[1])
        width = properties[2]-properties[0]
        height = properties[3] - properties[1]
        return [xy,width,height]

    def get_patch(self):
        return Rectangle(*self.properties,**self.kwargs)

class Sphere():
    def __init__(self, *properties, **kwargs):
        self.properties = self.convert_data_to_mpl(properties)
        self.kwargs = kwargs


    def convert_data_to_mpl(self, properties):
        """to matplotlib.patches.Circle(xy, radius=5, **kwargs)[source]
           from sphere <center       x y z>				  <radius>"""

        return ((properties[0],properties[1]),properties[2])

    def get_patch(self):
        return Circle(*self.properties,**self.kwargs)



class Geometry():
    title_pattern = re.compile("([a-z]+).*")
    properties_pattern = re.compile("[\d\.]+")
    fig_types = {'cuboid': Cuboid, 'sphere':Sphere}
    """Class of pre-processing geometry.in file"""
    def __init__(self, fname="geometry.in", output="geometry.png", tfsf=None, xlim=(0,1.0), zlim=(0,1.0)):
        self.objects = []
        self.index_color_dict = {0: "white", 1: "blue", 2: "red", 3: "black", 4: "yellow", 5: "gray"}
        self.xlim=xlim
        self.zlim=zlim
        self.output = output
        for string in open(fname, 'r'):
            if Geometry.title_pattern.match(string):
                type = Geometry.title_pattern.findall(string)[0]
                properties = Geometry.properties_pattern.findall(string)
                properties = [float(string) for string in properties]
                material_index = int(properties[0])
                properties = properties[1:]
                self.objects.append(self.fig_types[type](*properties, facecolor=self.index_color_dict[material_index]))
        if tfsf:
            self.tfsf_properties=[]
            for string in open(tfsf,'r'):
                self.tfsf_properties.append(eval(string.rstrip()))
            self.tfsf_properties=(self.tfsf_properties[0],self.tfsf_properties[2],self.tfsf_properties[1],self.tfsf_properties[3])
            self.objects.append(Cuboid(*self.tfsf_properties,fill=False,edgecolor='green',linestyle='--'))

    def draw(self, show=False):
        figures = []
        fig, ax = plt.subplots(1)
        for object in self.objects:
            ax.add_patch(object.get_patch())

        plt.xlim(self.xlim)
        plt.ylim(self.zlim)
        if show:
            plt.show()
        else:
            plt.savefig(self.output)
            plt.close()




"""    def compile_patterns(self):
        self.figures = {}
        self.figures["cuboid"] = "asdf"


    def is_shape_pattern(self):
        ellipse_pattern =

    def get_shape(self):


    def create_geometry(self):"""

if __name__=="__main__":
    Geometry(tfsf='tfsf.txt', xlim=(0.0, 0.364883)).draw()

#!/usr/bin/env python
'''
This script plots environemnt monitoring data obtained from 
'/tmp/gazebo_gym_experiments/', which has been created when 
calling a gazebo environment monitoring function in a test.

Args:

  default    Plots an averaged graph.
  arg1='i'   Prints an interpolated graph.

  arg1=int   Plots an averaged graph using 'arg1' as average 
             size delimiter.
  arg2='b'   Prints both graphs, the averaged one using 'arg1' 
             and the full data plot. 'arg2' must be 'b'.
  
Examples:

  python display_image.py
  python display_image.py i
  python display_image.py 20
  python display_image.py 20 b

'''
import os
import gym
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys
import argparse
import numpy as np
from scipy.interpolate import pchip

class LivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        #data_key can be set to 'episode_lengths'
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("episodes")
        plt.ylabel("cumulated episode rewards")
        fig = plt.gcf().canvas.set_window_title('averaged_simulation_graph')
        matplotlib.rcParams.update({'font.size': 15})

    def plot(self, mod1, mod2):
        results = gym.monitoring.monitor.load_results(self.outdir)
        data =  results[self.data_key]
        avg_data = []

        if mod1 == None or mod1.isdigit():
            if mod1 == None:
                mod1 = len(data)/50
                if mod1 == 0:
                    mod1 = 1
            else:
                mod1=int(mod1)
            for i, val in enumerate(data):
                if i%mod1==0:
                    if (i+mod1) < len(data):
                        avg =  sum(data[i:i+mod1])/mod1
                        avg_data.append(avg)
            new_data = expand(avg_data,mod1)
            if mod2 == 'b': #both avg and full
                plt.plot(data, color='blue')
                plt.plot(new_data, color='red', linewidth=2.5)
            else:
                plt.plot(new_data, color=self.line_color)

        elif mod1 == 'i': #interpolate data
            avg_data = []
            avg_data_points = []
            mod1 = len(data)/50
            if mod1 == 0:
                mod1 = 1
            data_fix = 0
            for i, val in enumerate(data):
                if i%mod1==0:
                    if (i+mod1) < len(data):
                        avg =  sum(data[i:i+mod1])/mod1
                        avg_data.append(avg)
                        avg_data_points.append(i)
                if (i+mod1) == len(data):
                    data_fix = mod1

            
            x = np.arange(len(avg_data))
            y = np.array(avg_data)
            #print x
            #print y
            #print str(len(avg_data)*mod1)
            #print data_fix
            interp = pchip(avg_data_points, avg_data)
            xx = np.linspace(0, len(data)-data_fix, 1000)
            plt.plot(xx, interp(xx), color='green', linewidth=2.5)

        

        

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)

def expand(lst, n):
    lst = [[i]*n for i in lst]
    lst = list(itertools.chain.from_iterable(lst))
    return lst

def pause():
    programPause = raw_input("Press the <ENTER> key to finish...")

if __name__ == '__main__':

    outdir = '/tmp/gazebo_gym_experiments'
    plotter = LivePlot(outdir)
    if len(sys.argv)==1:
        plotter.plot(None, None)
    elif len(sys.argv)==2:
        plotter.plot(sys.argv[1], None)
    else:
        plotter.plot(sys.argv[1], sys.argv[2])
    pause()

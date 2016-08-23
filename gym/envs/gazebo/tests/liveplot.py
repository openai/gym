#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt

class LivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    def plot(self):
        results = gym.monitoring.monitor.load_results(self.outdir)
        data =  results[self.data_key]

        #only update plot if data is different (plot calls are expensive)
        if data !=  self._last_data:
            self._last_data = data
            plt.plot(data, color=self.line_color)

            # pause so matplotlib will display
            # may want to figure out matplotlib animation or use a different library in the future
            plt.pause(0.000001)

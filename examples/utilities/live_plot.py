import gym

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
        fig = plt.gcf().canvas.set_window_title('')

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

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True, seed=0)

    # You may optionally include a LivePlot so that you can see
    # how your agent is performing.  Use plotter.plot() to update
    # the graph.
    plotter = LivePlot(outdir)

    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            ob, reward, done, _ = env.step(env.action_space.sample())
            if done:
                break

            plotter.plot()
            env.render()

    # Dump result info to disk
    env.monitor.close()

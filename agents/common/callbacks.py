from stable_baselines3.common.callbacks import BaseCallback

class PolarizationMetricsCallback(BaseCallback):
    """
    Custom callback to log polarization metrics at the end of each episode.
    """
    def __init__(self, verbose=0):
        super(PolarizationMetricsCallback, self).__init__(verbose)
        self.iteration = 0

    def _on_step(self) -> bool:
        # 1. Always grab the info from the current step
        infos = self.locals.get('infos', [{}])
        if infos:
            info = infos[-1]
            
            # Record current metrics (shows the evolution within the 100 steps)
            if 'consensus_degree' in info:
                self.logger.record('metrics/step_consensus', info['consensus_degree'])
            
            # 2. Check if the episode just ended
            # 'dones' is a numpy array in VecEnv; we check the first one
            dones = self.locals.get('dones')
            if dones is not None and dones[0]:
                # This is the "Final Result" of the episode
                if 'consensus_degree' in info:
                    self.logger.record('metrics/episode_final_consensus', info['consensus_degree'])
                
                # Force a write to the log file so you don't have to wait to see it
                self.logger.dump(step=self.num_timesteps)

        return True
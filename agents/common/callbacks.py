from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class PolarizationMetricsCallback(BaseCallback):
    """
    Custom callback to log mean polarization metrics across all parallel environments.
    """
    def __init__(self, verbose=0):
        super(PolarizationMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Retrieve all info dictionaries and done flags for all environments
        infos = self.locals.get('infos')
        dones = self.locals.get('dones')

        if infos is None:
            return True

        # 1. Log Step-wise Mean Metrics (aggregating across all n_envs)
        # We only record if the key exists in at least one info dict
        metrics_to_log = ['consensus_degree', 'change_effort', 'topo_deviation', 'reward_value']
        
        for metric in metrics_to_log:
            values = [info.get(metric) for info in infos if info.get(metric) is not None]
            if values:
                self.logger.record(f'metrics/step_mean_{metric}', np.mean(values))

        # 2. Log Episode Final Metrics
        # Check if ANY environment terminated
        if dones is not None and np.any(dones):
            # Only aggregate the metrics from environments that actually terminated
            terminated_indices = np.where(dones)[0]
            
            for metric in metrics_to_log:
                terminal_values = [infos[i].get(metric) for i in terminated_indices if infos[i].get(metric) is not None]
                if terminal_values:
                    self.logger.record(f'metrics/episode_final_{metric}', np.mean(terminal_values))

        return True
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

class EpisodeVisualizer:
    def __init__(self, env_config):
        self.max_range = env_config.get('max_range', 500)
        self.num_arrays = env_config.get('num_arrays', 4)

        self.history = {
            'uav_pos': [],
            'user_pos': [],
            'user_needs': [],
            'actions': [],
            'pinn_scores': [], # To track if they picked highly interfering users
            'step_reward': []
        }

    def record_step(self, env, action, reward, pinn_matrix):
        """Records the exact state of the environment at a single step."""
        self.history['uav_pos'].append(env.uav_position.copy())
        self.history['user_pos'].append(env.locations.copy())

        # Track remaining needs (normalized for color mapping)
        needs_remaining = np.maximum(env.needs - env.progress, 0.0)
        self.history['user_needs'].append(needs_remaining.copy())

        # Track action (which users are served)
        if isinstance(action, np.ndarray) and action.ndim > 0:
            act = action.copy()
        else:
            act = np.full(self.num_arrays, int(action))
        self.history['actions'].append(act)

        # Track how much PINN penalty they ate this step
        step_pinn_score = 1.0
        for i in range(len(act)):
            for j in range(i+1, len(act)):
                if act[i] != act[j]:
                    score = pinn_matrix[act[i], act[j]]
                    step_pinn_score = min(step_pinn_score, score)
        self.history['pinn_scores'].append(step_pinn_score)
        self.history['step_reward'].append(reward)

    def generate_gif(self, filename="diagnostics/episode_trace.gif", fps=5):
        """Renders the recorded history into an animated GIF and a static policy plot."""
        if not self.history['uav_pos']:
            return

        print(f"Generating GIF: {filename}...")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Set up the figure and axis
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        ax_map = fig.add_subplot(gs[0])
        ax_stats = fig.add_subplot(gs[1])

        # Static setup for map
        ax_map.set_xlim(-self.max_range, self.max_range)
        ax_map.set_ylim(-self.max_range, self.max_range)
        ax_map.set_title("UAV & User Beamforming Map")
        ax_map.grid(True, linestyle='--', alpha=0.5)

        # Pre-calculate plotting stats
        n_steps = len(self.history['actions'])
        switch_rates = []
        for t in range(1, n_steps):
            changes = np.sum(self.history['actions'][t] != self.history['actions'][t-1])
            switch_rates.append(changes / self.num_arrays)
        switch_rates = [0] + switch_rates # pad step 0

        def update(frame):
            ax_map.clear()
            ax_stats.clear()

            # --- MAP Rendering ---
            ax_map.set_xlim(-self.max_range, self.max_range)
            ax_map.set_ylim(-self.max_range, self.max_range)
            ax_map.set_title(f"Step: {frame}/{n_steps}")
            ax_map.grid(True, linestyle='--', alpha=0.5)

            uav = self.history['uav_pos'][frame]
            users = self.history['user_pos'][frame]
            needs = self.history['user_needs'][frame]
            actions = self.history['actions'][frame]

            # Draw Users (Color = Need, Red=High, Green=Done)
            scatter = ax_map.scatter(users[:, 0], users[:, 1], c=needs, cmap='RdYlGn_r', vmin=0, vmax=10, s=100, edgecolors='black')

            # Annotate Users
            for i, (x, y) in enumerate(users):
                ax_map.annotate(str(i), (x+10, y+10), fontsize=9)

            # Draw UAV Trail
            if frame > 0:
                trail = np.array(self.history['uav_pos'][:frame+1])
                ax_map.plot(trail[:, 0], trail[:, 1], color='gray', linestyle=':', alpha=0.5)

            # Draw UAV
            ax_map.scatter(uav[0], uav[1], marker='^', color='black', s=200, label='UAV', zorder=5)

            # Draw Beams
            for arr_idx, u_idx in enumerate(actions):
                # Offset slightly so multiple beams to same user don't perfectly overlap
                offset = (np.random.rand(2) - 0.5) * 5
                target = users[u_idx] + offset
                ax_map.plot([uav[0], target[0]], [uav[1], target[1]], color='blue', alpha=0.6, linewidth=2)

            # --- STATS Rendering ---
            ax_stats.set_title("Policy Interpretation Dashboard")
            ax_stats.set_xlim(0, n_steps)

            # Plot 1: PINN Safety Score
            ax_stats.plot(range(frame+1), self.history['pinn_scores'][:frame+1], color='orange', label='Min Pairwise PINN Score')
            ax_stats.set_ylim(-0.1, 1.1)
            ax_stats.axhline(0.2, color='red', linestyle='--', alpha=0.3, label='Interference Danger Zone')

            # Plot 2: Switching Rate (Secondary Y axis)
            ax2 = ax_stats.twinx()
            ax2.plot(range(frame+1), switch_rates[:frame+1], color='purple', alpha=0.5, label='Beam Switch Rate')
            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel("Switch Rate", color='purple')

            ax_stats.legend(loc='upper left', fontsize=8)
            ax_stats.set_xlabel("Steps")
            ax_stats.set_ylabel("PINN Isolation Score")

        anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
        writer = PillowWriter(fps=5)
        anim.save(filename, writer=writer)
        plt.close(fig)
        print(f"GIF saved successfully: {filename}")

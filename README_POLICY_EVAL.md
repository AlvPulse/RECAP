# Policy Evaluation and Action Masking Analysis

## Why does our RL policy outperform S-FCFS and M-FCFS?

1. **Strategic Interference Management via Action Masking:**
   Traditional Multi-user algorithms (like M-FCFS) fail catastrophically because they blindly assign users without regarding the angular separation. Our `Action Masking` actively prevents the agent from selecting spatially correlated users that would lead to un-resolvable interference or "DoF Cardinality" exhaustion.

2. **Sum-Throughput Reward Focusing:**
   We strictly rewarded the agent for actual bit transfers. The agent learned that, thanks to the action masking isolating users, it can safely exploit the 4 parallel arrays to serve multiple separated users concurrently. A single array (S-FCFS) has a hard upper bound of 1 transmission per step. The 4-array RL policy pushes past this bound by concurrently transmitting to isolated users without destructive self-interference.

3. **Max-Min Fairness + Completion Bonus:**
   We combined a max-min fairness state penalty with a massive completion bonus (`(n_served_now / 8) * 10.0`). This trains the agent to not just greedily maximize sum throughput, but strategically rotate through users to terminate the episode as fast as possible, beating RR (Round Robin) and PF (Proportional Fairness) algorithms in JFI score.

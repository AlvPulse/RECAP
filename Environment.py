import gymnasium as gym
from gymnasium import spaces
import copy
import numpy as np

f= 28 #GHz
c= 0.3
landa= c/f

# Constants
BANDWIDTH = 0.35  # 350 MHz
TIME_INTERVAL = 0.25  # 100 ms per time step
SINR_THRESHOLD_DB = 3  # Threshold SINR in dB
SINR_THRESHOLD_LINEAR = 10 ** (SINR_THRESHOLD_DB / 10)  # Convert dB to linear scale
UAV_HEIGHT = 50  # UAV height in meters
MAX_RANGE = 500  # Maximum range of users around the UAV in meters
MAX_EPISODE_TIME = 30  # Maximum episode time in seconds (1 minute)
MIN_PROGRESS = 1e-4
UAV_SPEED= 30
MAX_NEED=10
GRID_SIZE= 128

class UAVEnv(gym.Env):
    def __init__(self, num_users=8, num_arrays=4, num_elements_per_array=8, max_queue_size=100):
        super(UAVEnv, self).__init__()

        self.num_users = num_users
        self.num_arrays = num_arrays
        self.num_elements_per_array = num_elements_per_array
        self.max_queue_size = max_queue_size
        self.current_time = 0.0  # Track the current episode time

        self.bts_gain = 10 ** (50 / 10) * 10 ** (10 / 10)  # 50 dB TX, 10 dB RX for BTS-UAV path
        self.uav_user_gain = 10 ** (24 / 10) * 10 ** (0 / 10)  # 24 dB TX, 0 dB RX for UAV-User path


        # Array configuration (D, PhaseTable) for each array
        self.array_configs = [array_locs(self.num_elements_per_array) for _ in range(self.num_arrays)]

        # Define the action and observation space
        #self.action_space = spaces.MultiDiscrete()
        #self.action_space =spaces.MultiDiscrete(np.ones((self.num_arrays,))*self.num_users)
        self.action_space =spaces.Discrete(self.num_users)

        # Observation space: user needs, progress toward needs, delay, SINR, location, and UAV position
        self.observation_space = spaces.Dict({
            'needs': spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            #'progress': spaces.Box(low=0, high=MAX_NEED, shape=(self.num_users,), dtype=np.float64),
            #'delay': spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            #'sinr': spaces.Box(low=0, high=1000, shape=(self.num_users,), dtype=np.float64),
            #'locations': spaces.Box(low=-0.5, high=0.5, shape=(self.num_users, 2), dtype=np.float64),  # (x, y) coordinates for users
            'directions': spaces.Box(low=-0.5, high=0.5, shape=(self.num_users,), dtype=np.float64),  # (x, y) 
            #'uav_position': spaces.Box(low=-MAX_RANGE, high=MAX_RANGE, shape=(2,), dtype=np.float64)  # (x, y) coordinates for UAV (ignoring height as it's constant)
            #'active_distance':spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'distance':spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'user_satisfied': spaces.MultiBinary(self.num_users)
        })

        # Initialize environment state
        self.meanR = 0
        self.varR  = 1
        self.countR = 1e-4
        self.reset()
        self.last_action = None
        

    def reset(self,seed=None,options=None):
        if(seed):
            np.random.seed(seed)
        # Reset the environment state
        self.locations = np.random.uniform(-MAX_RANGE, MAX_RANGE, (self.num_users, 2))  # Random (x, y) locations within 100m range
        self.uav_position = np.random.uniform(-MAX_RANGE/2, MAX_RANGE/2, (2,))   # UAV starts at the origin
        self.needs = np.zeros(self.num_users)  # Needs start at 0, but some users will declare needs randomly
        self.progress = np.zeros(self.num_users)  # How much of the need has been fulfilled
        self.sinr = np.zeros(self.num_users)  # SINR for each user will be calculated dynamically
        self.delay = np.zeros(self.num_users)  # Delay experienced by each user
        self.current_time = 0.0  # Reset episode time

        # Queue of users who have declared their needs
        self.queue = np.zeros(self.num_users)
        #self.current_step=0

        # Randomly initialize at least one user's need at the start of the episode
        #initial_users = np.random.choice(self.num_users, size=np.random.randint(1, self.num_arrays+2), replace=False)
        #for user_idx in initial_users:
        for user_idx in range(self.num_users):
            self._declare_need(user_idx)

        return self._get_observation(), {}

    def copy(self):
        return copy.deepcopy(self)

    def step(self, action):
        switch_cost= 0.25
        # Increment time by the time interval
        self.current_time += TIME_INTERVAL
        #print(type(action))
        if (type(action)==np.int64) or( type(action)==int):
            selected_users= action* np.ones(self.num_arrays) 
            selected_users = selected_users.astype(int)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            selected_users= action.item()* np.ones(self.num_arrays)
            selected_users = selected_users.astype(int)
        else:
            # Apply actions: each array selects a user from the queue
            selected_users = action
        #print("selected_users",selected_users)
        # Update SINR based on the selected users and their locations
        self._calculate_sinr(selected_users)

        # Increment the delay for all users
        active_users = (self.needs > 0) & (self.needs > self.progress)

        for i in range(self.num_users):
          if active_users[i]:
            self.delay[i] += 1  # Increase delay for unserved users

        # the delay would become zero for served users
        # Update delay, progress, and queue based on the selected users

        if isinstance(selected_users, (int, np.integer)):
            selected_users = [selected_users]
        
        Throughput= 0
        for array_idx, user_idx in enumerate(selected_users):
          if active_users[user_idx]:
            Throughput+= self._serve_user(user_idx)
        # Update UAV location based on the optimal location for the selected users
        mean_distance= self._update_uav_location(selected_users)
        #print("diff_user_ratio_selcted",diff_user_ratio_avg)
        # Calculate the reward
        reward, delay_fairness, Bandwidth,Thr_fairness = self._calculate_reward(Throughput, mean_distance)##
        


        
        inactive_user_indices = [index for index, active in enumerate(active_users) if not active]
        
        matches = np.isin(selected_users, inactive_user_indices)

        # Sum the boolean array to count the number of matches
        Wrong_count = np.sum(matches)
        if(Wrong_count>0):
            print("active users",active_users)
            print("selected_user",selected_users)
        reward =reward-Wrong_count*10
        # Randomly declare new needs for users throughout the episode
        if np.random.rand() < 0.2:  # 20% chance each step that a user declares a need
            # Step 2: Randomly select one inactive user index
          if inactive_user_indices:  # Check if there are any inactive users
            selected_index = np.random.choice(inactive_user_indices)
            if self.needs[selected_index] == 0:  # Only declare need if the user hasn't already declared one
                self._declare_need(selected_index)
                #print("New User:", selected_index)
        

        # Check if done (e.g., all needs met or max time steps reached)
        done = np.all(self.progress >= self.needs)
        time_bonus_coeff= 0.0002
        if done:
            bonus = time_bonus_coeff * (MAX_EPISODE_TIME - self.current_time)**2
            reward += bonus
        # print(done.type)
        Truncated= self.current_time >= MAX_EPISODE_TIME

        # Apply penalty if time limit is reached and not all needs are met
        # if self.current_time >= MAX_EPISODE_TIME and not np.all(self.progress >= self.needs):
        #     reward -= 100000  # Apply a high penalty

        # Create the next observation
        observation = self._get_observation()
        # if(Truncated):
        #   print (self.needs-self.progress)
        # if(done):
        #   print(self.current_time)
        info ={"JFI":delay_fairness,"Bandwidth":Bandwidth,"Thr_fairness":Thr_fairness}
        # if(reward>0):
        #     print("positive reward")
        #     print(observation)
        

        if self.last_action is not None and action != self.last_action:
            reward -= switch_cost
            #print("step_num",int(self.current_time/TIME_INTERVAL),"action",action, "P action",self.last_action,"reward",reward)
            #print(observation['needs'])
        self.last_action = action
        return observation, reward, bool(done),Truncated, info

    def _declare_need(self, user_idx):
        # Declare a need for a specific user
        #self.needs[user_idx] = np.random.uniform(1, 10)  # Random need between 10 and 100 units
        self.needs[user_idx] = 10  #Unifying the need of the users

    def _serve_user(self, user_idx):
        throughput=0
        # Update the state of the user being served based on SINR
        if self.sinr[user_idx] > SINR_THRESHOLD_LINEAR:  # Only serve if SINR is above threshold
            # Calculate throughput using Shannon formula
            throughput = BANDWIDTH * np.log2(1 + self.sinr[user_idx]) * TIME_INTERVAL  # Throughput in bits
            self.delay[user_idx] = 0  # Reset delay for served user
            if(self.progress[user_idx]+throughput >self.needs[user_idx]):
                throughput= self.needs[user_idx]- self.progress[user_idx]
            self.progress[user_idx] += throughput  # Add throughput to the progress
        return throughput
                


    def _calculate_sinr(self, selected_users):
        # Initialize arrays for signal and interference levels for all users
        Noise_level = calculate_noise_level_db(BANDWIDTH)
        signals = np.ones(self.num_users)*Noise_level
        interferences = np.ones(self.num_users)*Noise_level
        if isinstance(selected_users, (int, np.integer)):
            selected_users = selected_users* np.ones(self.num_arrays)
            selected_users=selected_users.astype(int)
        #print(type(selected_users))
        #print(selected_users.size)
        #print(selected_users)

        # Calculate signal and interference for each array and user
        for array_idx, user_idx in enumerate(selected_users):
            # Calculate the desired user's direction (theta, phi) and distance
            #print("user_idx",user_idx)
            #print("array_idx",array_idx)
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)

            # Calculate the interference directions and distances to all other users
            other_user_indices = [i for i in selected_users if i != user_idx]
            RuserUAV= np.linalg.norm(self.uav_position - user_location)
            D, PhaseTable = self.array_configs[array_idx]
            Noise_level = calculate_noise_level_db(BANDWIDTH)
            if(other_user_indices):
              interference_directions, interference_distances = self._calculate_interference_directions(other_user_indices)

              # Get array configuration for this array

              # Use pert2d_null_multi to get Signal for the desired user and Interference for all other users
              Signal, InterferencedB = pert2d_null_multi(D, PhaseTable, direction[0], direction[1], RuserUAV,
                                                      interference_directions[:, 0], interference_directions[:, 1], interference_distances,Noise_level)
              Interference= 10**(InterferencedB/10)
              # Store signal for the desired user
              signals[user_idx] = Signal

              # Accumulate interference from this array to all other users
              interferences[other_user_indices] += Interference
            else:
              phBest= phase_code_finder(D, PhaseTable, direction[0], direction[1])
              Signal = find_gain_of_tphi(direction[0], direction[1],phBest, D) -total_path_loss(RuserUAV)
              signals[user_idx] = Signal

        #print("Signal",signals)
        interferences[interferences<Noise_level]=Noise_level
        # Calculate SINR for each user

        SINR = signals - (interferences)  # Avoid division by zero
        SINR[SINR<0]=0

        self.sinr= SINR

    def _calculate_direction(self, user_location):
        # Calculate direction (theta, phi) from the UAV to the user
        diff_x, diff_y = user_location - self.uav_position
        distance = np.sqrt(diff_x ** 2 + diff_y ** 2 + UAV_HEIGHT ** 2)  # 3D distance considering UAV height
        theta = np.degrees( np.arccos(UAV_HEIGHT / distance) )  # Angle with respect to the z-axis (elevation angle)
        phi = np.degrees( np.arctan2(diff_y, diff_x))  # Azimuth angle (horizontal angle)
        return theta, phi

    def _calculate_interference_directions(self, user_indices):
        directions = []
        distances = []
        for user_idx in user_indices:
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)
            distance = np.linalg.norm(np.append(user_location - self.uav_position, UAV_HEIGHT))  # 3D distance
            directions.append(direction)
            distances.append(distance)
        return np.array(directions), np.array(distances)

    def _update_uav_location(self, selected_users):
        # Update UAV location towards the average optimal location for the selected users
        optimal_locations = []
        for user_idx in selected_users:
            optimal_location = self._calculate_optimal_location(user_idx)
            optimal_locations.append(optimal_location)
#######  
        
        if optimal_locations:
            avg_optimal_location = np.mean(optimal_locations, axis=0)
            # Move UAV towards the average optimal location by one unit
            direction = avg_optimal_location - self.uav_position
            direction[direction<max(direction)]=0
            direction = direction / np.linalg.norm(direction)
            # print(direction)
            # print(self.uav_position)
            self.uav_position += direction*UAV_SPEED*TIME_INTERVAL
        user_distance = np.linalg.norm(self.uav_position - self.locations,axis=1)
        return np.mean(user_distance[selected_users])

    def _calculate_optimal_location(self, user_idx):
        # Calculate the optimal location of the UAV for the given user based on the Friis equation
        #print(self.uav_position)
        # BTS gain and UAV-user gain (already in linear scale)
        gain_ratio = self.bts_gain / self.uav_user_gain

        # Get user location
        user_location = self.locations[user_idx]

        # Calculate the optimal distance ratio
        optimal_distance_ratio = np.sqrt(gain_ratio)

        # Current distance from UAV to user
        user_distance = np.linalg.norm(self.uav_position - user_location)

        # Calculate optimal distance from UAV to user based on the Friis equation
        optimal_user_distance = user_distance / optimal_distance_ratio

        # Determine the direction from UAV to the user
        direction = user_location - self.uav_position
        if np.linalg.norm(direction) > 1e-6:  # Avoid division by zero
            direction = direction / np.linalg.norm(direction)

        # Calculate the optimal location based on the desired distance
        optimal_location = user_location - direction * optimal_user_distance
        return optimal_location

    def _calculate_reward(self, throuput,mean_distance):
        
        closeness_reward=10/(mean_distance+10)
        #print("closeness_reward",closeness_reward)
        user_weight= 1
        half_life_weight= 0.5
        throughput_weight =0.2
        
        # Only consider users who have declared a need
        active_users = (self.needs > 0) & (self.needs > self.progress)
        half_life_users = (self.needs/2>  self.progress)
        n_half_life= np.sum(half_life_users)
        n_active= (np.sum(active_users))
        
        
        episode_reward= episode_reward = (
            user_weight/(n_active+0.5)
          + half_life_weight/(n_half_life+1)
        )
        # SINR reward for all selected users
        sinr_reward = throuput**3 *throughput_weight
        #print("throuput",throuput)

        # Delay fairness using Jain's Fairness Index (JFI) for active users only
        sum_delay = np.sum(self.delay[active_users])
        sum_delay_squared = np.sum(self.delay[active_users] ** 2)
        jfi_delay = (sum_delay ** 2) / (np.sum(active_users) * sum_delay_squared + 1e-6)  # Avoid division by zero
        #print("jfi_delay",jfi_delay)
        # Throughput fairness using proportional fairness, applying a minimum progress threshold
        proportional_fairness_reward = np.sum(np.log(np.maximum(self.progress[active_users]/MAX_NEED, MIN_PROGRESS)))  # Avoid log(0) and extreme penalties
        #print("proportional_fairness_reward",proportional_fairness_reward)
        # Combined reward
        alpha = 2  # Weight for delay fairness
        beta = 0.1   # Weight for throughput fairness

        raw_reward = sinr_reward - alpha * jfi_delay + beta * proportional_fairness_reward - episode_reward
        #raw_reward = sinr_reward + beta * proportional_fairness_reward - episode_reward
        #reward = sinr_reward - alpha * jfi_delay - episode_reward
        #raw_reward = sinr_reward + episode_reward + closeness_reward -3
        #raw_reward = sinr_reward + episode_reward -3


        #print("sinr_reward",sinr_reward)
        #print("alpha * jfi_delay",alpha * jfi_delay)
        ##print("episode_reward",episode_reward)
        #print("Location",self.uav_position)

        self.countR += 1
        delta = raw_reward - self.meanR
        self.meanR += delta / self.countR
        delta2 = raw_reward - self.meanR
        self.varR = ( (self.countR-1)*self.varR + delta*delta2 ) / self.countR
        # Normalize & clip
        norm_rew = (raw_reward - self.meanR) / (np.sqrt(self.varR) + 1e-8)
        norm_rew = np.clip(norm_rew, -10, +10)
        #if(n_active< 2):
        #    print(f"raw: {raw_reward:.2f},norm: {norm_rew:.2f}, sinr: {sinr_reward:.2f}, eps: {episode_reward:.2f}, close: {closeness_reward:.2f}")
        
        return raw_reward/10, jfi_delay, sinr_reward, proportional_fairness_reward

    # Define `action_masks` in your env
    def get_action_mask(self):
        # e.g., False for satisfied users
        return [self.needs>self.progress]

    def get_active_distances(self):
        needs_progress= self.needs-self.progress
        user_satisfied =needs_progress<=0
        needs_progress[user_satisfied]=0
        distance= np.linalg.norm(self.locations-self.uav_position,axis=1)
        distance[user_satisfied]= 0
        return distance


    def _get_observation(self):
        needs_progress= self.needs-self.progress
        user_satisfied =needs_progress<=0
        needs_progress[user_satisfied]=0
        distance= np.linalg.norm(self.locations-self.uav_position,axis=1)
        #distance[user_satisfied]= 0
        
        user_directions=np.zeros((self.num_users,))
        for user_idx in range(self.num_users):
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)
            user_directions[user_idx]=direction[1]
        user_directions[user_satisfied]=0
        
        return {
            'needs': needs_progress/MAX_NEED,
            #'progress': self.progress,
            #'delay': self.delay/MAX_EPISODE_TIME,
            #'sinr': self.sinr,
            #'locations': (self.locations-self.uav_position)/MAX_RANGE/2,
            'directions': user_directions/360,
            #'uav_position': self.uav_position
            'distance': distance/MAX_RANGE/2,
            'user_satisfied': user_satisfied
        }


def make_env():
    return UAVEnv()

class VisualUAVEnv(UAVEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(GRID_SIZE, GRID_SIZE, 3),
            dtype=np.float32
        )

    def _get_observation(self):
        return self.visualize_environment()

    def visualize_environment(self):
        image = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        active_users = (self.needs > 0) & (self.needs > self.progress)

        uav_x, uav_y = self.uav_position + MAX_RANGE
        uav_x, uav_y = int((uav_x / (2 * MAX_RANGE)) * GRID_SIZE), int((uav_y / (2 * MAX_RANGE)) * GRID_SIZE)
        image[uav_x, uav_y, :] = 1.0  # White pixel for UAV

        for i, (x, y) in enumerate(self.locations):
            if active_users[i]:
                user_x, user_y = x + MAX_RANGE, y + MAX_RANGE
                user_x, user_y = int((user_x / (2 * MAX_RANGE)) * GRID_SIZE), int((user_y / (2 * MAX_RANGE)) * GRID_SIZE)

                image[user_x, user_y, 1] = (self.needs[i] - self.progress[i]) / self.needs[i]  # Green
                image[user_x, user_y, 0] = self.delay[i] / MAX_EPISODE_TIME                   # Red
                image[user_x, user_y, 2] = self.progress[i] / MAX_NEED                        # Blue

        return image

def array_locs(N):
    """
    Initializes the array locations and computes the phase table for a given number of elements N.

    Parameters:
    N (int): Number of elements in the array.

    Returns:
    tuple: (D, PhaseTable2)
        D (numpy.ndarray): Array of element locations.
        PhaseTable2 (numpy.ndarray): Phase table for the array.
    """
    if N < 16 or N in [20, 24, 36, 60]:
        Ny = 1
    elif N < 64:
        Ny = 2
    elif N == 256:
        Ny = 8
    else:
        Ny = 4

    Nx = N // (4 * Ny)
    xAr = np.arange(-Nx, Nx + 1)
    yAr = np.arange(-Ny, Ny + 1)
    x = (xAr[:-1] + 0.5) / 2
    y = (yAr[:-1] + 0.5) / 2

    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    D = np.column_stack((X, Y))
    PhaseTable1 = np.tile([0, 180], (N, 1))

    added_phase_flag = np.mod(np.sum(D, axis=1) * 2, 2)
    added_phase = np.tile(added_phase_flag * 90, (2, 1)).T
    PhaseTable2 = PhaseTable1 + added_phase

    return D, PhaseTable2


def los_probability_uav(d, h_uav, h_user=1.5, a=12.08, b=0.11):
    """
    Height-aware LoS probability model for UAVs in dense urban mmWave.

    Parameters
    ----------
    d : float or np.array
        Horizontal distance (in meters)
    h_uav : float
        UAV height (in meters)
    h_user : float
        Ground user height (default = 1.5 m)
    a, b : float
        Environment-specific parameters (dense urban by default)

    Returns
    -------
    P_LoS : float or np.array
        LoS probability

            Al-Hourani, K., Kandeepan, S., & Lardner, S.
    “Optimal LAP Altitude for Maximum Coverage”, IEEE WC Letters, 2014.
    
    Also cited by:
    
    3GPP TR 36.777 (UAV-enhanced LTE)
    
    Multiple IEEE/Elsevier UAV mmWave papers
    """
    #print("hello")
    h_diff = h_uav - h_user
    theta_deg = np.degrees(np.arctan2(h_diff, d))
    P_LoS = 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))
    return np.clip(P_LoS, 0.0, 1.0)

def los_probability(d_n_m, h_n, h_m, beta_a, beta_b, epsilon_i):
    """
    Calculate the Line-of-Sight (LoS) probability based on the blockage model.

    Parameters:
    d_n_m (float): Horizontal distance between UAV and user.
    h_n (float): Height of the UAV.
    h_m (float): Height of the user (usually close to 0).
    beta_a (float): Ratio of obstacle area to total area.
    beta_b (float): Obstacle density parameter.
    epsilon_i (float): Scale parameter for obstacle height distribution (Rayleigh distribution).

    Returns:
    float: LoS probability.
    """
    # Vertical distance between UAV and user
    Dh_n_m = np.abs(h_n - h_m)

    # Blockage model parameter gamma
    gamma = np.floor(np.sqrt(beta_a * beta_b)*d_n_m)
    #print("gamma",gamma)
    #print("exponent",(gamma * h_n-(0.5)* Dh_n_m )**2)
    P_LoS=1
    if gamma==0:
      return 1
    else:
      # Calculate LoS probability using the blockage model
      for n in range(gamma.astype(int)+1):
        P_LoS_i = 1-np.exp(-(gamma * h_n-(0.5+n)* Dh_n_m )**2/ (2*(epsilon_i*gamma)**2))
        P_LoS=P_LoS*P_LoS_i
        #print("P_LoS_i",P_LoS_i)
    # Ensure the LoS probability is between 0 and 1
    P_LoS = np.clip(P_LoS, 0, 1)
    #print("P_LoS",P_LoS)
    return P_LoS

def path_loss_los(d_n_m, f_c, zeta_0, zeta_1, zeta_2):
    """
    Calculate the path loss for Line-of-Sight (LoS) link.

    Parameters:
    d_n_m (float): Distance between UAV and user.
    f_c (float): Carrier frequency in GHz.
    zeta_0, zeta_1, zeta_2 (float): LoS path loss parameters.

    Returns:
    float: Path loss in dB.
    """
    return zeta_0 + zeta_1 * np.log10(d_n_m) + zeta_2 * np.log10(f_c)

def path_loss_nlos(d_n_m, f_c, eta_0, eta_1, eta_2):
    """
    Calculate the path loss for Non-Line-of-Sight (NLoS) link.

    Parameters:
    d_n_m (float): Distance between UAV and user.
    f_c (float): Carrier frequency in GHz.
    eta_0, eta_1, eta_2 (float): NLoS path loss parameters.

    Returns:
    float: Path loss in dB.
    """
    return eta_0 + eta_1 * np.log10(d_n_m) + eta_2 * np.log10(f_c)

def rayleigh_fading():
    """
    Apply Rayleigh fading to the signal.

    Returns:
    float: Fading gain in linear scale (not dB).
    """
    fading_gain = np.random.rayleigh(scale=1.0)
    return fading_gain


zeta_params = (38.77, 16.7, 18.2)  # LoS path loss parameters
eta_params = (36.85, 30, 18.9)  # NLoS path loss parameters
beta_a = 0.4  # Ratio of obstacle area to total area
beta_b = 0.4  # Obstacle density parameter
epsilon_i = 1  # Scale parameter for obstacle height (Rayleigh)

def total_path_loss(d_n_m, h_n=100, h_m=1.5, f_c=28):
    """
    Calculate the total path loss considering both LoS and NLoS probabilities and fading.

    Parameters:
    d_n_m (float): Horizantal Distance between UAV and user.
    h_n (float): Height of the UAV.
    h_m (float): Height of the user.
    f_c (float): Carrier frequency in GHz.
    zeta_params (tuple): Parameters for LoS path loss (zeta_0, zeta_1, zeta_2).
    eta_params (tuple): Parameters for NLoS path loss (eta_0, eta_1, eta_2).
    beta_a, beta_b, epsilon_i (float): Parameters for the LoS probability model.

    Returns:
    float: Total path loss in dB.
    """
    # Calculate LoS probability
    #P_LoS = los_probability(d_n_m, h_n, h_m, beta_a, beta_b, epsilon_i)
    P_LoS = los_probability_uav(d_n_m, h_n)
    #print(P_LoS)
    # Calculate path loss for LoS and NLoS links
    PL_LoS = path_loss_los(d_n_m, f_c, *zeta_params)
    PL_NLoS = path_loss_nlos(d_n_m, f_c, *eta_params)

    # Total path loss with fading
    total_PL = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

    # Apply Rayleigh fading
    fading_gain = rayleigh_fading()

    # Convert fading gain to dB and apply to total path loss
    total_PL_with_fading = total_PL + 1.5 * np.log10(fading_gain)
    #print("fading_gain",10 * np.log10(fading_gain))

    return total_PL_with_fading

def phase_code_finder(D, PhaseTable, theta, phi):
    """
    Finds the phase code for the array based on the input parameters.

    Parameters:
    D (numpy.ndarray): Array element locations.
    PhaseTable (numpy.ndarray): Phase table for the array.
    theta (float): Signal direction angle in degrees.
    phi (float): Signal azimuth angle in degrees.

    Returns:
    numpy.ndarray: Calculated phase code.
    """
    PhaseCode1 = PhaseTable[:, 0]

    # Assuming error_calculator is implemented as described earlier
    ErrorArray = error_calculator(D, theta, phi, PhaseCode1)

    PhaseCode = PhaseCode1.copy()
    mask = (ErrorArray > 90) & (ErrorArray < 270)
    PhaseCode[mask] += 180

    return PhaseCode

def fom_calc_null_multi(D, PhaseCode, theta0, phi0, thetaN, phiN, RN, Noise_level):
    """
    Calculates the Figure of Merit (FoM) based on signal gain and interference.

    Parameters:
    D (numpy.ndarray): Array element locations.
    PhaseCode (numpy.ndarray): Phase table for the array.
    theta0, phi0 (float): Signal direction angles.
    thetaN, phiN (numpy.ndarray): Null direction angles.
    RN (numpy.ndarray): Distances to null sources.
    Noise_level (float): Noise level in dB.

    Returns:
    tuple: (FoM, Gain0)
        FoM (float): Figure of Merit.
        Gain0 (float): Gain in the main signal direction.
    """
    Gain0 = find_gain_of_tphi_n(theta0, phi0, PhaseCode, D)
    GainArr = np.zeros_like(thetaN)

    for i in range(len(RN)):
        GainNull = find_gain_of_tphi(thetaN[i], phiN[i], PhaseCode, D) - db(RN[i])
        #print("GainNull",GainNull)
        #print("RN",RN[i])
        if GainNull < Noise_level:
            GainNull = Noise_level
        GainArr[i] = GainNull

    FoM = Gain0 + np.max(GainArr)
    #print("Gain0:",Gain0)
    #print("GainArr",GainArr)
    #print("FOM",FoM)
    return FoM, Gain0

def find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCode, D):
    """
    Calculates the interference caused by nulls.

    Parameters:
    thetaN, phiN (numpy.ndarray): Null direction angles.
    RN (numpy.ndarray): Distances to null sources.
    PhaseCode (numpy.ndarray): Phase code for the array.
    D (numpy.ndarray): Array element locations.

    Returns:
    numpy.ndarray: Interference values.
    """
    Interference = np.zeros_like(thetaN).astype(float)

    for i in range(len(thetaN)):
        ErrorArray = error_calculator(D, thetaN[i], phiN[i], PhaseCode)
        Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))/2
        Interference[i] = Gain - db(4*RN[i]*np.pi/landa)

    return Interference

def find_gain_of_tphi_n(theta, phi, PhaseCode, D):
    """
    Calculates the gain in a given direction.

    Parameters:
    theta, phi (float): Direction angles.
    PhaseCode (numpy.ndarray): Phase code for the array.
    D (numpy.ndarray): Array element locations.

    Returns:
    float: Gain value.
    """
    N=len(PhaseCode)
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))/2
    return Gain-db(N)/2

def find_gain_of_tphi(theta, phi, PhaseCode, D):
    """
    Calculates the gain in a given direction.

    Parameters:
    theta, phi (float): Direction angles.
    PhaseCode (numpy.ndarray): Phase code for the array.
    D (numpy.ndarray): Array element locations.

    Returns:
    float: Gain value.
    """
    ErrorArray = error_calculator(D, theta, phi, PhaseCode)
    Gain = db(np.abs(np.sum(np.exp(1j * np.deg2rad(ErrorArray)))))
    return Gain

def find_most_effective_null_multi(D, theta, phi, thetaN, phiN, WeightN, PhaseCode, MinNumber):
    """
    Identifies the most effective element to adjust for null steering.

    Parameters:
    D (numpy.ndarray): Array element locations.
    theta, phi (float): Signal direction angles.
    thetaN, phiN (numpy.ndarray): Null direction angles.
    WeightN (numpy.ndarray): Weights for null directions.
    PhaseCode (numpy.ndarray): Phase code for the array.
    MinNumber (int): Index of the most effective element.

    Returns:
    tuple: (flag, IndAlter)
        flag (bool): Whether the element adjustment is effective.
        IndAlter (int): Index of the most effective element to adjust.
    """
    ErrorArrayT = error_calculator(D, theta, phi, PhaseCode)
    PhaseMat = np.exp(1j * np.deg2rad(ErrorArrayT))

    Rval = np.real(np.sum(PhaseMat))
    I = np.imag(np.sum(PhaseMat))

    if Rval == 0 and I == 0:
        IndAlter = np.random.randint(len(PhaseCode))
    else:
        DeltaG = Rval * np.real(PhaseMat) + I * np.imag(PhaseMat)
        DeltaNSum = np.zeros_like(DeltaG)

        for i in range(len(WeightN)):
            ErrorArrayN = error_calculator(D, thetaN[i], phiN[i], PhaseCode)
            PhaseMatN = np.exp(1j * np.deg2rad(ErrorArrayN))
            RvalN = np.real(np.sum(PhaseMatN))
            IN = np.imag(np.sum(PhaseMatN))
            DeltaGN = RvalN * np.real(PhaseMatN) + IN * np.imag(PhaseMatN)
            DeltaNSum += WeightN[i] * DeltaGN

        DeltaN = DeltaG - DeltaNSum
        sorted_indices = np.argsort(DeltaN)
        IndAlter = sorted_indices[MinNumber]
        flag = DeltaN[sorted_indices[MinNumber]] >= 0

    return flag, IndAlter

def pert2d_null_multi(D, PhaseTable, theta, phi, R, thetaN, phiN, RN, Noise_level):
    """
    Calculates the signal and interference caused by users based on null steering parameters.

    Parameters:
    D (numpy.ndarray): Array element locations.
    PhaseTable (numpy.ndarray): Phase table for the array.
    theta, phi (float): Signal direction angles.
    R (float): Distance to the signal source.
    thetaN, phiN (numpy.ndarray): Null direction angles.
    RN (numpy.ndarray): Distances to null sources.
    Noise_level (float): Noise level in dB.

    Returns:
    tuple: (Signal, Interference)
        Signal (float): Calculated signal strength.
        Interference (float): Calculated interference strength.
    """
    # Find the initial phase code
    PhaseCodeStart = phase_code_finder(D, PhaseTable, theta, phi)

    # Calculate the initial Figure of Merit (FoM)
    FoMStart, _ = fom_calc_null_multi(D, PhaseCodeStart, theta, phi, thetaN, phiN, RN, Noise_level)

    # Calculate the initial interference in dB
    InterferenceStartdB = find_gain_of_tphi_i(thetaN, phiN, RN, PhaseCodeStart, D)
    #print("InterferenceStartdB:",InterferenceStartdB)
    # Calculate weights based on the interference
    WeightsN = 10 ** ((InterferenceStartdB - np.max(InterferenceStartdB)) / 5)

    # Initialize variables for optimization
    PhaseCodeAltering = PhaseCodeStart.copy()
    PhaseCodeBest = PhaseCodeAltering.copy()
    FOMAltering = FoMStart
    FoMBest = FoMStart
    IndNumber = 1
    IterNumber = 0
    FoMTarget = Noise_level + 3  # Target for stopping
    #print(FOMAltering)
    N=len(PhaseCodeBest)
    # Optimization loop
    while FOMAltering > FoMTarget:
      #print("iter:",IndNumber)
      PhaseCodeAltering = PhaseCodeBest.copy()
      Flag, ElementIndex = find_most_effective_null_multi(D, theta, phi, thetaN, phiN, WeightsN, PhaseCodeAltering, IndNumber)

      PhaseCodeAltering[ElementIndex] += 180  # Flip phase
      FOMAltering, _ = fom_calc_null_multi(D, PhaseCodeAltering, theta, phi, thetaN, phiN, RN, Noise_level)

      if FOMAltering < FoMBest:
          PhaseCodeBest = PhaseCodeAltering.copy()
          FoMBest = FOMAltering
          IndNumber=1;
      else:
        IndNumber +=1
        if IndNumber >= N:  # Safety limit to avoid infinite loops
          break
      IterNumber += 1
    Signal = find_gain_of_tphi(theta, phi, PhaseCodeBest, D) -total_path_loss(R)
    Interference= np.zeros_like(InterferenceStartdB)
    for i in range(len(RN)):
      Interference[i] = find_gain_of_tphi(thetaN[i],phiN[i], PhaseCodeBest, D)-total_path_loss(RN[i])# Placeholder for actual interference calculation

    return Signal, Interference

def error_calculator(D, theta, phi, PhaseCode):
    """
    Calculates the error array based on the input parameters.

    Parameters:
    D (numpy.ndarray): Array element locations.
    theta (float): Signal direction angle in degrees.
    phi (float): Signal azimuth angle in degrees.
    PhaseCode (numpy.ndarray): Phase code for the array.

    Returns:
    numpy.ndarray: Calculated error array.
    """
    lambda_wave = 1  # Wavelength
    K = 360 / lambda_wave  # Wave number (2 * pi / lambda)

    # Calculate the incoming phase wave
    Phase_wave_in = K * np.sin(np.radians(theta)) * D @ np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])

    # Calculate the error array
    ErrorArray = np.mod(PhaseCode.astype(float) - Phase_wave_in, 360)

    return ErrorArray

def db(x):
  return 20*np.log10(x)

def calculate_noise_level_db(bandwidth_ghz, temperature_kelvin=290):
    """
    Calculate the noise level in dBm given the bandwidth in GHz.

    Parameters:
    bandwidth_ghz (float): Bandwidth in GHz.
    temperature_kelvin (float): System temperature in Kelvin. Default is 290 K (room temperature).

    Returns:
    float: Noise level in dBm.
    """
    # Constants
    k = 1.38e-23  # Boltzmann constant in Joules per Kelvin

    # Convert bandwidth from GHz to Hz
    bandwidth_hz = bandwidth_ghz * 1e9

    # Calculate noise power in watts
    noise_power_watts = k * temperature_kelvin * bandwidth_hz

    # Convert noise power to dBm
    noise_level_db = 10 * np.log10(noise_power_watts)  # Multiply by 1e3 to convert watts to milliwatts

    return noise_level_db

import random
import math

MAX_RANGE = 100  # Maximum range of users around the UAV in meters

def random_user_distribution():
    users = []
    radius_min = 0
    radius_max = MAX_RANGE/2

    # Define segments
    segments = [
        (0, 90, radius_min, radius_max),  # Segment 1
        (90, 180, radius_min, radius_max),  # Segment 2
        (180, 270, radius_min, radius_max),  # Segment 3
        (270, 360, radius_min, radius_max),  # Segment 4
        (0, 90, radius_max, 2 * radius_max),  # Segment 5
        (90, 180, radius_max, 2 * radius_max),  # Segment 6
        (180, 270, radius_max, 2 * radius_max),  # Segment 7
        (270, 360, radius_max, 2 * radius_max)  # Segment 8
    ]

    for segment in segments:
        angle_min, angle_max, r_min, r_max = segment

        # Randomly generate polar coordinates within the segment
        angle = random.uniform(math.radians(angle_min), math.radians(angle_max))
        #print(math.degrees( angle))
        radius = random.uniform(r_min, r_max)

        # Convert polar coordinates to cartesian coordinates
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        users.append((x, y))

    # Sort users as per the specified order of segments
    return np.array(users)
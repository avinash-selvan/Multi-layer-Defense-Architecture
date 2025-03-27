import gym
from gym import spaces
import numpy as np

#############################################################
# TODO: need to make compatible with different architectures
#############################################################

# these are specific to the default CAGE 2 Environment -----------------------------------------------------

# attacker and defender actions
RED_ACTIONS = ['sleep', 'remote', 'network', 'exploit', 'escalate', 'impact']
BLUE_ACTIONS = ['sleep', 'analyse', 'decoy', 'remove', 'restore']

# specify network configuration
NUM_SUBNETS = 3
HOSTS = ['def', 'ent0', 'ent1', 'ent2', 'ophost0', 
    'ophost1', 'ophost2', 'opserv', 'user0', 'user1', 'user2', 'user3', 'user4']

# which hosts are connected
CONNECTED_HOSTS = [
    None, None, None, 'opserv', None, None, 
    None, None, None, 'ent1', 'ent1', 'ent0', 'ent0']

# what services are already running on the machines
# -> highlights the exploit they correspond to
HOST_EXPLOITS = [
    
    # ent
    ['Brute'],
    ['Brute'],
    ['Brute', 'Eternal', 'Keep', 'HTTPRFI', 'HTTPSRFI'],
    ['Brute', 'Eternal', 'Keep', 'HTTPRFI', 'HTTPSRFI'],

    # op
    ['Brute'],
    ['Brute'],
    ['Brute'],
    ['Brute'],

    ##############################################
    # BUG: user3 is meant to possess SQL exploit, 
    # but is in fact replace by bluekeep
    ##############################################

    # user
    [], 
    ['Brute', 'FTP'],
    ['Eternal', 'Keep'],
    ['Keep', 'HTTPSRFI', 'HTTPRFI', 'Haraka'],
    ['Keep', 'HTTPSRFI', 'HTTPRFI', 'Haraka', 'SQL']
]

# only some user exploits are rewarded
# this highlights the exploits which are
REWARDED_EXPLOITS = [

    # ent
    [],
    [],
    ['Keep', 'Eternal'],
    ['Keep', 'Eternal'],

    # op
    [],
    [],
    [],
    [],

    # user
    [],
    ['FTP'],
    ['Eternal'],
    ['Keep', 'Haraka'],
    ['SQL', 'Haraka']

]

# what decoy options are available for each host
# ordered based on the Cardiff implementation
HOST_DECOYS = [

    # ent
    ['Haraka', 'Tomcat', 'Apache', 'Vsftpd'],
    ['Haraka', 'Tomcat', 'Vsftpd', 'Apache'],
    ['Femitter'],
    ['Femitter'],

    # op
    [],
    [],
    [],
    ['Haraka', 'Apache', 'Tomcat', 'Vsftpd'], 

    # user
    [],
    ['Apache', 'Tomcat', 'SMSS', 'Svchost'],
    ['Femitter', 'Tomcat', 'Apache', 'SSHD'],
    ['Vsftpd', 'SSHD'],
    ['Vsftpd']
]


# list all the decoy and exploit options
# ranked in order of priority (high to low)
EXPLOITS = ['FTP', 'Haraka', 'SQL', 'HTTPSRFI', 'HTTPRFI', 'Eternal', 'Keep', 'Brute'] 
DECOYS = ['Femitter', 'Vsftpd', 'Apache', 'Haraka', 'SSHD', 'SMSS', 'Tomcat', 'Svchost']

def exploits_to_decoys(remove_bugs):
    '''Give an exploit index and return the compatible decoys.'''

    ######################################################
    # BUG: vsftp has the wrong port attached to it
    # -> so in fact it actually stops HTTPRFI
    # -> it basically deploying an apache server instead
    ######################################################
    ftp_decoys = [0]
    sql_decoys = [2, 6, 1]
    httprfi_decoys = [2, 1]
    if remove_bugs:
        ftp_decoys = [0, 1]
        sql_decoys = [2, 6]
        httprfi_decoys = [2]

    # maps the exploit to the decoys that can stop it
    mapping = np.zeros((len(EXPLOITS), len(DECOYS)))
    mapping[0, ftp_decoys] = 1         # FTP       ->  Femitter (-Vsftp)
    mapping[1, [3]] = 1         # Haraka    ->  Haraka
    mapping[2, sql_decoys] = 1   # SQL       ->  Apache, Tomcat (+Vsftp)
    mapping[3, [6]] = 1         # HTTPSRFI  ->  Tomcat
    mapping[4, httprfi_decoys] = 1      # HTTPRFI   ->  Apache (+ Vsftp)
    mapping[5, [5]] = 1         # Eternal   ->  SMSS
    mapping[6, [7]] = 1         # Keep      ->  Svchost
    mapping[7, [4]] = 1         # Brute     ->  SSHD
    return mapping   


def construct_exploit_rew():
    ''''''
    mapping = np.zeros((len(HOSTS), len(EXPLOITS)))
    for idx, hosts in enumerate(REWARDED_EXPLOITS):
        for h in hosts:
            h_idx = EXPLOITS.index(h)
            mapping[idx, h_idx] = 1
    return mapping

def create_subnets(num_nodes=13):
    '''
    Divide the nodes into subnets.
    '''
    subnets = np.zeros(num_nodes)
    subnets[4:8] = 1
    subnets[8:] = 2
 
    return subnets


def get_host_priority(hosts):
    '''
    Designate the hosts for reward classification.
    '''
    hosts = np.array(hosts)
    host_priority = np.zeros_like(hosts, dtype=np.int32).reshape(-1)

    # user and op hosts
    # low priority hosts
    user_idxs = np.char.find(hosts, 'user')
    host_priority[np.nonzero(user_idxs+1)[0]] = 1
    op_idxs = np.char.find(hosts, 'ophost')
    host_priority[np.nonzero(op_idxs+1)[0]] = 1

    # enterprise and defender
    # medium priority
    ent_idxs = np.char.find(hosts, 'ent')
    host_priority[np.nonzero(ent_idxs+1)[0]] = 2
    def_idxs = np.char.find(hosts, 'def')
    host_priority[np.nonzero(def_idxs+1)[0]] = 2

    # opserver
    # lowest priority
    opserv_idxs = np.char.find(hosts, 'opserv')
    host_priority[np.nonzero(opserv_idxs+1)[0]] = 3

    return host_priority

# --------------------------------------------------------------------------------------------

def default_host_exploits(remove_bug=False):
    '''Get the default exploits for each network host.'''
    mapping = np.zeros((len(HOST_EXPLOITS), len(EXPLOITS)))
    
    # add in SQL exploit on user3
    host_exploits = HOST_EXPLOITS
    if remove_bug:
        host_exploits[11].append('SQL')

    for i, host in enumerate(host_exploits):
        for _, exploit in enumerate(host):
            mapping[i][EXPLOITS.index(exploit)] = 1
    return mapping


def default_defender_decoys():
    '''Return the mapping of hosts to decoys.'''
    mapping = np.zeros((len(HOST_DECOYS), len(DECOYS)))
    for i, host in enumerate(HOST_DECOYS):
        for j, decoy in enumerate(host):
            mapping[i][DECOYS.index(decoy)] = (len(host) - j)
    return mapping


def action_mapping():
    '''
    Return the integer to string mapping for red/blue agent.
    '''

    # populate the red map
    red_a_map = [RED_ACTIONS[0]]
    for i in range(NUM_SUBNETS):
        red_a_map.append(f'{RED_ACTIONS[1]}_sub{i}')
    for a in RED_ACTIONS[2:]:
        for h in HOSTS:
            red_a_map.append(f'{a}_{h}')

    # populate the blue map
    blue_a_map = [BLUE_ACTIONS[0]]
    for a in BLUE_ACTIONS[1:]:
        for h in HOSTS:
            blue_a_map.append(f'{a}_{h}')

    return {'Blue': blue_a_map, 'Red': red_a_map}


def check_red_access(obs):
    '''
    Determine the access the of the red agent within 
    the network, including known hosts, scanned_hosts and host_access.
    '''

    # extract the host information
    batch_size = obs.shape[0]
    host_access = obs.reshape(batch_size, -1, 3)[:, :, 1:]

    # check which networks have been scanned
    # -> used to identify exploitable networks
    scanned = obs.reshape(batch_size, -1, 3)[:, :, 0] == 1

    # check subnet privlege
    # -> used to enable remote service scan
    priv_1 = np.any(host_access[:, :4, 1]==1, axis=-1)
    priv_2 = np.any(host_access[:, 4:8, 1]==1, axis=-1)
    priv_3 = np.any(host_access[:, 8:, 1]==1, axis=-1)
    subnet_priv = np.concatenate([
        priv_1.reshape(-1, 1),
        priv_2.reshape(-1, 1), 
        priv_3.reshape(-1, 1)], axis=-1)

    # check user and privlege access
    # -> known used for remote network scan
    # -> user used for escalate privleges
    # -> privleged used to impact
    known_hosts = host_access[:, :, 0] != -1
    user_access = np.any(host_access == 1, axis=-1)
    privleged_access = host_access[:, :, 1] == 1

    return known_hosts, scanned, user_access, privleged_access, subnet_priv


def get_possible_red_actions(
    user_access, priv_access, known_hosts, subnets, scanned):
    '''Return a list of valid red team actions.'''

    # add discover remote services actions
    # remove when all networks on host are known
    known_1 = np.all(known_hosts[:, :4], axis=-1)
    known_2 = np.all(known_hosts[:, 4:8], axis=-1)
    known_3 = np.all(known_hosts[:, 8:], axis=-1)
    known_subnets = np.concatenate([
        known_1.reshape(-1, 1),
        known_2.reshape(-1, 1),
        known_3.reshape(-1, 1)], axis=-1)*1
    
    # create a mask
    # ensure that sleep action is always allowed
    batch_size = user_access.shape[0]
    action_mask = np.zeros(
        (batch_size, NUM_SUBNETS+len(HOSTS)*len(RED_ACTIONS[2:])+1))
    action_mask[:, 0] = 1

    # keep track of actions including those not already taken
    full_action_mask = action_mask.copy()

    # add discover remote services
    subnet_indices = np.nonzero(subnets * (1-known_subnets))
    action_mask[subnet_indices[0], subnet_indices[1]+1] = 1
    full_action_mask[np.nonzero(subnets)[0], np.nonzero(subnets)[1]+1] = 1
    added_actions = NUM_SUBNETS+1

    # add discover network services
    # remove if already scanned
    known_hosts_indices = np.nonzero(known_hosts * (1-scanned))
    action_mask[known_hosts_indices[0], known_hosts_indices[1]+added_actions] = 1
    full_action_mask[np.nonzero(known_hosts)[0], np.nonzero(known_hosts)[1]+added_actions] = 1
    added_actions += len(HOSTS)
    
    # add exploit remote services
    # remove if already exploited
    scanned_indices = np.nonzero(scanned * (1-user_access))
    action_mask[scanned_indices[0], scanned_indices[1]+added_actions] = 1
    full_action_mask[np.nonzero(scanned)[0], np.nonzero(scanned)[1]+added_actions] = 1
    added_actions += len(HOSTS)

    # escalate privileges
    # remove if already escalated
    user_access_indices = np.nonzero(user_access * (1-priv_access))
    action_mask[user_access_indices[0], user_access_indices[1]+added_actions] = 1
    full_action_mask[np.nonzero(user_access)[0], np.nonzero(user_access)[1]+added_actions] = 1
    added_actions += len(HOSTS)
    
    # impact components
    priv_access_indices = np.nonzero(priv_access)
    action_mask[priv_access_indices[0], priv_access_indices[1]+added_actions] = 1
    full_action_mask[priv_access_indices[0], priv_access_indices[1]+added_actions] = 1
    added_actions += len(HOSTS)

    #############################################
    # NOTE: this is specific to this environment
    #############################################
    # defender can never exploited
    action_mask[:, 17] = 0 

    return action_mask, full_action_mask


test = np.zeros(3)
def update_red(state, action, subnet_loc, processes, impacted, femitter_placed, remove_bug=False):
    '''
    Improved Red Agent Action Selection and Exploit Application
    '''
    
    # Copy current state
    action_reward = np.zeros((state.shape[0], 1))
    next_state = state.copy()
    
    # Get possible actions
    known, scanned, user, priv, subnet = check_red_access(state)
    red_mask, full_red_mask = get_possible_red_actions(user, priv, known, subnet, scanned)
    
    # Validate actions
    action_filter = red_mask[np.arange(len(action)), action.reshape(-1).astype(int)]
    success = np.ones((state.shape[0], 1))
    new_impacted = impacted.copy()

    # Store exploit selection
    selected_exploit_idx = -np.ones(state.shape[0])

    if np.any(action_filter):
        # Get targeted host and action type
        host_alloc = ((action - (NUM_SUBNETS + 1)) % len(HOSTS)).reshape(-1).astype(int)
        action_alloc = np.floor((action - (NUM_SUBNETS + 1)) / len(HOSTS)).reshape(-1).astype(int)

        # Determine attackable processes
        valid = np.logical_and(action_filter, (action_alloc == 1))  # Only consider exploit actions
        if np.any(valid):
            proc = processes[valid]
            host = host_alloc[valid]

            # Get process data on target host
            host_processes = proc[np.arange(len(proc)), host].reshape(-1, proc.shape[-1])

            # Select exploit based on available processes
            exploit_match = []
            for i, host_proc in enumerate(host_processes):
                possible_exploits = [
                    idx for idx, exploit in enumerate(EXPLOITS) if host_proc[idx] == 1
                ]
                if possible_exploits:
                    best_exploit = max(possible_exploits)  # Choose the highest-value exploit
                else:
                    best_exploit = -1  # No valid exploits
                exploit_match.append(best_exploit)

            selected_exploit_idx[valid] = np.array(exploit_match)

            # Calculate success probability
            exploit_success = np.zeros((valid.shape[0])).astype(bool)
            for i, exp in enumerate(selected_exploit_idx[valid]):
                if exp >= 0:
                    exploit_success[i] = np.random.rand() < 0.85  # 85% success if valid exploit

            # Apply successful exploit updates
            if np.any(exploit_success):
                mod_state = next_state[valid].copy().reshape(-1, len(HOSTS), 3)
                mod_state[np.arange(len(host_alloc[valid])), host_alloc[valid]] = np.array([1, 1, 0])
                next_state[valid] = mod_state.reshape(mod_state.shape[0], -1)

            # Handle failures
            exploit_fail = np.logical_and(valid, np.invert(exploit_success))
            if np.any(exploit_fail):
                success[exploit_fail] = -1

    # Apply rewards
    action_reward[success == 1] += 10  # Success
    action_reward[success == -1] -= 5  # Failure

    # print(f"🚀 Selected Exploit: {selected_exploit_idx}")
    # print(f"✅ Exploit Success: {success}")
    
    return next_state, action_reward, success, new_impacted, selected_exploit_idx

def check_blue_action(observation, decoys):
    '''
    Check which blue actions are available.
    '''
    
    if len(observation.shape) == 1:
        observation = observation.reshape(1, -1)
    batch_size = observation.shape[0]
    action_mask = np.zeros((batch_size, len(HOSTS)*len(BLUE_ACTIONS[1:])+1))
    action_mask[:, 0] = 1

    # add analyse actions
    action_mask[:, range(1, len(HOSTS)+1)] = 1
    added_actions = len(HOSTS)+1

    # add decoy actions for those that still have decoys
    subnet_decoy_idxs = np.nonzero(decoys > 0)
    action_mask[subnet_decoy_idxs[0], subnet_decoy_idxs[1]+added_actions] = 1
    added_actions += len(HOSTS)

    # add remove actions
    action_mask[:, range(added_actions, len(HOSTS)+added_actions)] = 1
    added_actions += len(HOSTS)

    # add restore actions
    action_mask[:, range(added_actions, len(HOSTS)+added_actions)] = 1

    #############################################
    # NOTE: this is specific to this environment
    #############################################
    # user0 cannot be restored
    action_mask[:, 48] = 0

    return action_mask

def update_blue(state, updated_state, action, impacted):
    """
    Update the environment for Blue Team, incorporating attack detection and defensive actions.
    """
    if isinstance(action, np.ndarray):
        action = action.flatten()  # Convert multi-dimensional arrays to 1D
    else:
        action = np.array([action])  # Convert scalar to array

    print(f"🔍 Blue Action Values: {np.unique(action)}")
    
    # Copy the state
    action_reward = np.zeros((state.shape[0], 1))  
    next_state = updated_state.copy()

    # Log attacks
    detection_log = np.zeros_like(impacted)

    for i in range(len(action)):  
        if action[i] in [1, 2]:  # Ensure other detection actions are considered
            detection_log[i] = impacted[i]  # Log detected attacks
            action_reward[i] += 5  # Reward Blue for detecting

    
    print(f"🏆 Detection Reward: {action_reward}")
    print(f"🔍 Detection Log: {detection_log}")
    
    return next_state, action_reward, detection_log



class SimplifiedCAGE(gym.Env):
    '''
    A simplified version of the CAGE 2 Challenge environment 
    with faster execution speed and parallelism.
    '''

    def __init__(self, num_envs, num_nodes=13, remove_bugs=False):

        super(SimplifiedCAGE, self).__init__()
        # basic parameters
        self.num_envs = num_envs
        self.observation_space = spaces.Box(low=0, high=1, shape=(52,), dtype=np.float32)  # Adjust if needed
        self.action_space = spaces.Discrete(3)  # 3 actions: Block, Alert, Ignore
        self.num_nodes = num_nodes
        self.remove_bugs = remove_bugs

        # map integer in host_alloc[valid] exes to action name
        self.action_mapping = action_mapping()
        self.detection_log = np.zeros((self.num_envs, self.num_nodes))  # Ensure it's initialized

        # reset all the parameters
        self.reset()

    def _set_init(
        self, num_envs, num_nodes, decoys=None, impacted=None, 
        state=None, current_processes=None, detection=None):
        '''Set the initialisation parameters.'''
        
        # map host allocation to subnet
        # identify host priority
        self.subnets = np.tile(
            create_subnets(num_nodes).reshape(1, -1), (num_envs, 1))
        self.host_priority = np.tile(
            get_host_priority(HOSTS).reshape(1, -1), (num_envs, 1))

        # decoy and exploit information
        # -> given exploit index return compatible decoys
        # -> for each host return built in exploits
        # -> for each host return compatible decoys
        self.exploit_map = exploits_to_decoys(remove_bugs=self.remove_bugs)
        self.default_exploits = default_host_exploits(remove_bug=self.remove_bugs)
        self.default_decoys = np.tile(
            np.expand_dims(default_defender_decoys(),
            axis=0), (self.num_envs, 1, 1))

        # set the initial state
        # add a privleged access to user0
        self.state = state
        if state is None:
            self.state = -np.ones((num_envs, num_nodes*3))
            self.state[:, 24:27] = np.array([0, 0, 1])
        self.proc_states = None

        # keep track of action success
        self.blue_success = -np.ones((num_envs, 1))
        self.red_success = -np.ones((num_envs, 1))

        # keep track of impacts
        self.impacted = impacted
        if impacted is None:
            self.impacted = np.zeros((num_envs, num_nodes))

        # keep track of exploitable process and available decoys
        # legit process are marked as 1, decoys are -1
        # decoys are numbered  by priority for highest to lowest
        self.current_processes = current_processes
        if current_processes is None:
            self.current_processes = np.tile(
                np.expand_dims(self.default_exploits.copy(),
                axis=0), (num_envs, 1, 1))

        # add placeholder selected exploit
        self.selected_exploit = -np.ones(num_envs)
        
        # log the decoys
        self.current_decoys = decoys
        if decoys is None:
            self.current_decoys = self.default_decoys.copy()

        # keep track of previously failed detection
        self.detection = detection
        if self.detection is None:
            self.detection = np.zeros((
                num_envs, num_nodes)).astype(bool)

        # get blue observation of the state
        state = self._process_state(
            state=self.state, 
            logged_decoys=self.current_decoys)

        # keep track of the exploits used
        self.exploit_rewards = np.tile(
            construct_exploit_rew()[None], (num_envs, 1, 1))
        self.host_exploits = -np.ones((num_envs, num_nodes)) 

        # in bugged version femitter is stuck after being placed
        self.femitter_placed = np.zeros((
            num_envs, num_nodes)).astype(bool)

        return state


    def _get_info(self):
        info = {
            'impacted': self.impacted, 
            'current_processes': self.current_processes,
            'current_decoys': self.current_decoys}
        return info


    def reset(self):
        state = self._set_init(num_envs=self.num_envs, num_nodes=self.num_nodes)
        blue_state = state["Blue"].squeeze()

        # ✅ Convert to a NumPy array
        blue_state = np.array(blue_state, dtype=np.float32)

        # ✅ Ensure it's exactly 52 elements
        blue_state = blue_state[:52] if blue_state.shape[0] > 52 else np.pad(blue_state, (0, 52 - blue_state.shape[0]), 'constant')

        # blue_state = blue_state.reshape(1, -1)
        print(f"🔍 Reset Output Shape: {blue_state.shape}, Type: {type(blue_state)}")

        # 🔥 Debug what SB3 is receiving
        # import traceback
        # print(f"🔍 Stack Trace: {traceback.format_stack()}")
        blue_state = np.ascontiguousarray(blue_state, dtype=np.float32)

        print(f"🔍 Final Reset Output Shape: {blue_state.shape}, Type: {type(blue_state)}, Dtype: {blue_state.dtype}")
        import sys
        print(f"🔍 Buffer Check: {sys.getsizeof(blue_state)} bytes")


        return blue_state, {}  # ✅ Only return the NumPy array (not a tuple)


    def _generate_red_action(self):
        return np.random.randint(0, self.action_space.n, size=(self.num_envs,))



    def step(self, blue_action, red_action):
        print(f"PRE-STATE: {self.state}")
        red_action = self._generate_red_action()

        if isinstance(blue_action, np.ndarray):
            blue_action = blue_action.item()  # Convert from array to scalar
        
        true_state, reward_dict = self._process_actions(
            self.state, red_action, blue_action, self.subnets)
        
        self.state = true_state.copy()

        next_state = self._process_state(
            state=true_state, logged_decoys=self.current_decoys, 
            red_action=red_action, blue_action=blue_action)
        
        self.proc_states = next_state
        info = self._get_info()

        blue_state = next_state["Blue"].squeeze()
        blue_state = np.array(blue_state, dtype=np.float32)
        blue_state = blue_state[:52] if blue_state.shape[0] > 52 else np.pad(blue_state, (0, 52 - blue_state.shape[0]), 'constant')
        
        # Pass blue_action to reward processing
        reward_dict = self._process_reward(self.state, reward_dict, self.impacted, blue_action)
        
        reward = float(reward_dict["Blue"] - reward_dict["Red"])
        done = False
        
        print(f"POST-STATE: {self.state}")
        return blue_state, reward, done, info


        

    def get_mask(self, state, decoys):
        '''
        Get the action mask for the current state configuration.
        '''
        
        # get the blue mask
        blue_mask = check_blue_action(state, decoys)

        # get the red action mask
        known, scanned, user, priv, subnet = check_red_access(state)
        red_mask, _ = get_possible_red_actions(
            user, priv, known, subnet, scanned)

        return {'Red': red_mask, 'Blue': blue_mask}
    

    def _process_actions(
        self, state, red_action, blue_action, subnets):
        '''
        Update the internal states based on blue/red actions
        '''

        ############################################
        # TODO: are the success values usable  
        # -> red success is the only important one

        # -> success can be cancelled out via restore
        #   -> if restore occurs with priv
        ############################################

        # get next state and corresponding reward
        # add probability of failure
        print(f"🔍 Processing Actions - Blue: {blue_action}, Red: {red_action}")

        true_state, red_reward, success, impacted, selected_exploit = update_red(
            state=state, action=red_action, subnet_loc=subnets, 
            processes=self.current_processes, 
            impacted=self.impacted,
            femitter_placed=self.femitter_placed,
            remove_bug=self.remove_bugs)
        # self.red_success = success
        self.red_success = np.where(self.detection_log > 0, 0, success)  

        self.selected_exploit = selected_exploit

        # update the host exploits
        host_selected = ((red_action - 4) % self.num_nodes).astype(int)
        self.host_exploits[
            np.arange(len(host_selected)), host_selected] = selected_exploit

        # now perform blue update
        # perform the blue action first
        true_state, blue_reward, detection_log = update_blue(
            state=state, 
            updated_state=true_state, 
            action=blue_action, 
            # decoys=self.current_decoys, 
            # processes=self.current_processes, 
            # proc_map=self.exploit_map,
            impacted=self.red_success,
            # femitter_placed=self.femitter_placed
            )
        self.blue_success = success
        self.impacted = impacted
        self.detection_log = detection_log

        #self.femitter_placed = femitter_placed

        # # reset the decoys
        # if np.any(decoy_reset):
        #     decoys[decoy_reset.astype(bool)] = self.default_decoys[
        #         decoy_reset.astype(bool)]
        #     def_exploits = np.tile(
        #         self.default_exploits[None], (decoy_reset.shape[0], 1, 1))
        #     proc[decoy_reset.astype(bool)] = def_exploits[decoy_reset.astype(bool)]
        # self.current_processes = proc
        # self.current_decoys = decoys


        # impact action should also influence blue but negatively
        # blue_reward -= red_reward
        
        # blue_reward -= red_reward * (1 - self.detection_log)  # Reduce impact if detected
        detection_factor = detection_log.mean(axis=-1, keepdims=True)  # Reduce 13 values into 1
        blue_reward -= red_reward * (1 - detection_factor)*5
        
        print("we now keep the  ddimensions: detection_factor = detection_log.mean(axis=-1, keepdims=True)  # Reduce 13 values into 1 blue_reward -= red_reward * (1 - detection_factor)")
        print(f"🚨 Red Agent Success: {self.red_success}, Selected Exploits: {self.selected_exploit}")

        print(f"🔍 Reward Debug: Blue Reward = {blue_reward}, Red Reward = {red_reward}")
        print(f"🔍 Final Reward - Blue: {blue_reward}, Red: {red_reward}")

        return true_state, {'Blue': blue_reward, 'Red': red_reward}


    def _process_reward(self, state, action_reward, impacted, blue_action=None):  # Added blue_action parameter
        print("🔍 I'm processing the reward!!!!")
        
        state_info = state.reshape(-1, self.num_nodes, 3).copy()
        state_info[:, 8] = 0  # mask out user0

        user_access = state_info[:, :, 1].reshape(-1) > 0
        priv_access = state_info[:, :, 2].reshape(-1) > 0
        
        flat_host = self.host_priority.reshape(-1)
        reward = np.zeros((state.shape[0], 1))
        
        # Reward calculations...
        
        # Scale down rewards
        action_reward['Blue'] = action_reward['Blue'] * 0.1
        action_reward['Red'] = action_reward['Red'] * 0.1
        
        # Penalty for doing nothing (using the passed blue_action)
        if blue_action is not None and np.all(blue_action == 0):
            action_reward['Blue'] -= 0.1
        
        # Clip rewards
        action_reward['Blue'] = np.clip(action_reward['Blue'], -10, 10)
        action_reward['Red'] = np.clip(action_reward['Red'], -5, 5)
        
        print(f"After scaling: {action_reward}")
        return action_reward


    def _process_state(
        self, state, logged_decoys, red_action=None, blue_action=None):
        '''
        Convert the true state into observations of each agent.
        '''
        
        #############################################
        # TODO: host should analyse unless otherwise
        #############################################
        
        #################################################
        # TODO: host should say removed unless otherwise
        #################################################

        # add success to red state
        # red_state = np.concatenate([self.red_success.reshape(-1, 1), state], axis=-1)
        #self.red_success = np.where(self.detection_log > 0, 0, success).reshape(-1, 1)
        if not hasattr(self, "red_success"):  # Ensure it's defined
            self.red_success = np.zeros((self.num_envs, self.num_nodes))  # Initialize it

        self.red_success = np.where(self.detection_log > 0, 0, self.red_success).reshape(-1, 1)

        # Ensure red_success has the correct batch size (match state.shape[0])
        # if self.red_success.shape[0] != state.shape[0]:  
        #     self.red_success = np.tile(self.red_success, (state.shape[0], 1))  

        # red_state = np.concatenate([self.red_success, state.reshape(state.shape[0], -1)], axis=-1)
        # self.red_success = np.asarray(self.red_success).reshape(-1, 1)  # Ensure it's 2D
        # if self.red_success.shape[0] != state.shape[0]:
        #     self.red_success = np.tile(self.red_success, (state.shape[0], 1))
                # red_state = np.concatenate([self.red_success, state.reshape(state.shape[0], -1)], axis=-1)
        # Create a projection matrix to match dimensions
        if self.red_success.shape[0] != state.shape[0]:
            T = np.eye(state.shape[0], self.red_success.shape[0])  # Identity projection
            self.red_success = T @ self.red_success  # Project to correct shape
        red_state = np.concatenate([self.red_success, state.reshape(state.shape[0], -1)], axis=-1)



        # initialise blank state
        blue_state = np.zeros((state.shape[0], 6*self.num_nodes))
        activity_info = np.zeros((state.shape[0], self.num_nodes, 2))
        safety_info = np.zeros((state.shape[0], self.num_nodes, 2))
        scan_info = np.zeros((state.shape[0], self.num_nodes))
        decoy_info = np.zeros((state.shape[0], self.num_nodes))

        # extract the previous processed blue state
        # divide into scan activity, host safety, prior scans and decoy info
        if self.proc_states is not None:
            prev_state = self.proc_states['Blue']
            def_info = prev_state[:, :self.num_nodes*4]
            added_info = prev_state[:, self.num_nodes*4:]
            def_info = def_info.reshape(
                def_info.shape[0], self.num_nodes, -1)
            activity_info = def_info[:, :, :2]
            safety_info = def_info[:, :, 2:]
            scan_info = added_info[:, :self.num_nodes]
            decoy_info = added_info[:, self.num_nodes:]

        # add a green action
        # 50% chance to scan a host
        green_mask = np.zeros((activity_info.shape[0], self.num_nodes))
        selected_green_host = np.ones(activity_info.shape[0]) * -1
        green_scan = np.random.choice(
            [0, 1], size=activity_info.shape[0], p=[0.5, 0.5])
        if np.any(green_scan):
            green_host = np.random.choice(
                [1, 2, 3, 8, 9, 10, 11, 12], size=np.sum(green_scan))
            temp_activity = activity_info[green_scan.astype(bool)].copy()
            temp_activity[np.arange(len(green_host)), green_host.astype(int)] = np.array([1, 0])
            green_mask[np.arange(len(green_host)), green_host.astype(int)] = 1
            activity_info[green_scan.astype(bool)] = temp_activity
            selected_green_host[green_scan.astype(bool)] = green_host

        # determine consequences of red actions
        # also to a lesser extent the green actions
        if red_action is not None:
            
            # update prior activity
            # remove previous scans and other activity
            prev_scanned = np.logical_and(
                activity_info[:, :, 0].astype(bool), np.invert(green_mask.astype(bool)))
            if np.any(prev_scanned):
                temp_info = activity_info[np.any(prev_scanned, axis=-1)].copy().reshape(-1, 2)
                temp_info[:] = np.array([0, 0])
                temp_info = temp_info.reshape(-1, self.num_nodes, 2)
                activity_info[np.any(prev_scanned, axis=-1)] = temp_info

            # extract the host information
            host_alloc = ((red_action-4) % self.num_nodes).reshape(-1).astype(int) 

            # add in previously exploited hosts
            prev_exploited = state.reshape(-1, self.num_nodes, 3)[:, :, 1] == 1
            prev_exploited = np.logical_or(
                prev_exploited, state.reshape(-1, self.num_nodes, 3)[:, :, 2] == 1)
            prev_exploited = np.logical_and(
                prev_exploited.astype(bool), self.detection.astype(bool))
            if np.any(prev_exploited):

                # check if exploit is current host
                exp_idxs = np.where(prev_exploited)
                is_host = host_alloc[exp_idxs[0]] == exp_idxs[1]
                is_host = np.logical_or(is_host, exp_idxs[1] == 8)

                # ensure this is not the current exploit
                # or if it is user0 as this shouldn't appear
                curr_exploit = np.logical_and(
                    red_action > 3+self.num_nodes, 
                    red_action < (self.num_nodes*2+4)).reshape(-1)
                curr_exploit = curr_exploit[exp_idxs[0]]
                ignore_exp = np.logical_and(is_host, np.invert(curr_exploit))   
                ignore_exp = np.logical_or(ignore_exp, exp_idxs[1] == 8)
                if np.any(ignore_exp):
                    prev_exploited[exp_idxs[0][ignore_exp], exp_idxs[1][ignore_exp]] = False

            if np.any(prev_exploited):
                safety_copy = safety_info[np.any(prev_exploited, axis=-1)].copy()
                exp_idxs = np.where(prev_exploited[np.any(prev_exploited, axis=-1)])
                safety_copy[exp_idxs[0], exp_idxs[1]] = np.array([0, 1])
                safety_info[np.any(prev_exploited, axis=-1)] = safety_copy
    
            # only scan and exploits trigger alert
            # set scan to [1, 0, 0, 0]
            # add 5% chance it wont be recorded -> now 100%
            scan = np.logical_and(
                red_action > 3, red_action < (self.num_nodes+4)).reshape(-1)
            scan_mask = np.random.choice([0, 1], size=scan.shape, p=[0.0, 1.0])    
            scan = np.logical_and(scan, scan_mask)
            if np.any(scan):
                host = host_alloc[scan]
                activity_copy = activity_info[scan].copy()
                activity_copy[np.arange(len(host)), host] = np.array([1, 0])
                activity_info[scan] = activity_copy

            # check for exploits
            # set to [1, 1, 0, 1] if exploited
            # add 5% chance it goes unrecorded
            # if exploit is sshbruteforce this has 100% visibility
            # if green scanning with ssh -> access granted
            exploit = np.logical_and(
                red_action > 3+self.num_nodes, 
                red_action < (self.num_nodes*2+4)).reshape(-1)

            exploit_mask = np.random.choice([0, 1], size=scan.shape[0], p=[0.05, 0.95])
            green_scanning = (selected_green_host == host_alloc).reshape(-1)
            ssh_selected = self.selected_exploit == (len(EXPLOITS)-1)
            detected = np.logical_and(
                np.logical_or(exploit_mask, ssh_selected), np.invert(green_scanning))
            
            #####################################################
            # BUG: if host is ent0 -> always visible exploit
            # -> but undetectable if there is a decoy on host
            #####################################################
            decoys_on_ent0 = np.any(self.current_processes[:, 1] == -1, axis=-1)
            ent0_detected = np.logical_and((host_alloc == 1), np.invert(decoys_on_ent0))
            if not self.remove_bugs:
                detected = np.logical_or(detected, ent0_detected)
            exploit = np.logical_and(exploit, detected)

            # add action success
            # keep track of host detection
            exploit = np.logical_and(
                self.red_success.reshape(-1) == 1, exploit)     

            self.detection[np.arange(len(host_alloc)), host_alloc] = exploit.astype(bool)
            if np.any(exploit):
                host = host_alloc[exploit]
                activity_copy = activity_info[exploit].copy()
                activity_copy[np.arange(len(host)), host] = np.array([1, 1])
                activity_info[exploit] = activity_copy
                safety_copy = safety_info[exploit].copy()
                safety_copy[np.arange(len(host)), host] = np.array([0, 1])
                safety_info[exploit] = safety_copy

        # determine consequnces of blue actions
        if blue_action is not None:
            
            # extract the host action is applied to
            host_alloc = np.array((blue_action-1) % self.num_nodes).reshape(-1).astype(int)  

            # use analyse to reveal if privileged access achieved
            # [0, 0, 0, 0] to [0, 0, 1, 1]
            analyse = np.logical_and(
                blue_action > 0, blue_action < self.num_nodes+1).reshape(-1).astype(bool)
            if np.any(analyse):
                host = host_alloc[analyse]
                mod_state = state.reshape(-1, self.num_nodes, 3)[analyse]
                
                is_priv = np.zeros_like(analyse)
                is_p = (mod_state[np.arange(len(host)), host, -1] == 1).reshape(-1)
                is_priv[analyse] = is_p
                analyse = np.logical_and(analyse, is_priv)

                host = host_alloc[analyse]
                safety_copy = safety_info[analyse].copy()
                safety_copy[np.arange(len(host)), host] = np.array([1, 1]) 
                safety_info[analyse] = safety_copy

            # remove processes from the network
            # -> if removed show [0, 0, 1, 0] -> however this is no guarantee it was removed
            remove = np.logical_and(
                blue_action > self.num_nodes*2, 
                blue_action < self.num_nodes*3+1).astype(bool).reshape(-1)
            remove = np.logical_and(self.blue_success.reshape(-1), remove)
            if np.any(remove):
                host = host_alloc[remove]
                temp_safety = safety_info[remove].copy()
                temp_safety[np.arange(len(host)), host] = np.array([1, 0])
                safety_info[remove] = temp_safety 

            # restore the machine to default 
            # i.e. 0 0 0 0
            restore = np.array(blue_action, dtype=int).reshape(-1).astype(bool)
            restore = np.logical_and(restore, self.blue_success.reshape(-1))
            if np.any(restore):
                host = host_alloc[restore]
                temp_safety = safety_info[restore].copy()
                temp_safety[np.arange(len(host)), host] = np.array([0, 0])
                safety_info[restore] = temp_safety

                activity_copy = activity_info[restore].copy()
                activity_copy[np.arange(len(host)), host] = np.array([0, 0])
                activity_info[restore] = activity_copy

        # extract scan information
        # reset prior highlighted scans            
        # highlight the most recent scans 
        scan_info[(scan_info == 2)] = 1
        scan_info += activity_info[:, :, 0] * 2
        scan_info = np.clip(scan_info, a_min=0, a_max=2)

        # set the decoy information
        decoy_info = logged_decoys.copy()
        decoy_info = np.sum(decoy_info > 0, axis=-1)
        decoy_info = decoy_info.reshape(scan_info.shape[0], -1)

        # combine all the components 
        default_state = np.concatenate([
            activity_info, safety_info], axis=-1)
        blue_state = np.concatenate([
            default_state.reshape(scan_info.shape[0], -1),
            scan_info, decoy_info], axis=-1)

        obs_state = {
            'Red': red_state, 'Blue': blue_state
        }

        return obs_state


    def set_game_state(
        self, state, impacted, current_processes, current_decoys, detection):
        '''Set the state of the environment.'''

        # update the environmental number and configuration
        self.num_envs = state.shape[0]
        self.num_nodes = state.shape[-1]//3

        # reset the necessary parameters
        state = self._set_init(
            num_envs=self.num_envs, num_nodes=self.num_nodes, 
            decoys=current_decoys, state=state, impacted=impacted,
            current_processes=current_processes, detection=detection)
        
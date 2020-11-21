import torch


class Node:
    def __init__(self, state, reward, policy, value, action_size, device, batch_size):
        self.q_values = torch.zeros([batch_size, action_size]).to(device)

        self.visits = torch.zeros([batch_size, action_size]).to(device)

        self.reward = reward
        self.state = state
        self.policy = policy
        self.value = value

        self.action_size = action_size

    def number_visits(self, action):
        return self.visits[torch.arange(action.size(0)), action]

    def update_num_visit(self, action):
        self.visits[torch.arange(action.size(0)), action] += 1

    def get_total_visits(self):
        return torch.sum(self.visits, dim=1)

    def q_value(self, action):
        return self.q_values[torch.arange(action.size(0)), action]

    def set_q_value(self, q_val, action):
        self.q_values[torch.arange(action.size(0)), action] = q_val

    def min_q(self):
        return torch.min(self.q_values)

    def max_q(self):
        return torch.max(self.q_values)

    def update(self, gain, action, min_q, max_q):

        n_a = self.number_visits(action)
        q_old = self.q_value(action)
        q_val = (n_a * q_old + gain) / (n_a + 1)

        q_val = (q_val - min_q) / (max_q - min_q)

        self.set_q_value(q_val, action)
        self.update_num_visit(action)

    def action_distribution(self):
        total = self.get_total_visits().unsqueeze(1).repeat(1, self.action_size)  # batch x action size

        return self.visits / total


class MCTS:
    def __init__(self, model, mcts_param, action_length, device, advantaged=False):
        self.model = model
        self.device = device
        self.k = mcts_param["k_sims"]
        self.c1 = torch.Tensor([mcts_param["c1"]]).to(self.device)
        self.c2 = torch.Tensor([mcts_param["c2"]]).to(self.device)
        self.gamma = torch.Tensor([mcts_param["gamma"]]).to(self.device)
        self.action_space = action_length
        self.advantaged = advantaged

        assert self.k > 1, "K simulations length must be greater than 1"

        self.lookup_table = {}
        self.state_node_dict = {}

    def get_action(self, node):
        total_visits = node.get_total_visits().unsqueeze(1).repeat(1, self.action_space)
        term = (self.c1 + torch.log(total_visits + self.c2 + torch.ones([1]).to(self.device)) - torch.log(self.c2))

        values = node.q_values + node.policy * term * torch.sqrt(total_visits) / (torch.Tensor([1]).to(self.device) + node.visits)

        if self.advantaged:
            values = values - node.value

        return torch.argmax(values, dim=1)  # argmax over action dim

    def lookup(self, state, action):
        if (state, action) in self.lookup_table:
            return self.lookup_table[(state, action)]
        else:
            reward, new_state = self.model.dynamics_function(state, action)
            self.lookup_table[(state, action)] = (new_state, reward)
            return new_state, reward

    def state_to_node(self, state, reward=None):
        if state in self.state_node_dict:
            return self.state_node_dict[state]
        else:
            if reward is None:
                reward = torch.zeros([self.batch_size, 1]).to(self.device)
            policy, value = self.model.prediction_function(state)
            new_node = Node(state, reward, policy, value,
                            action_size=self.action_space, device=self.device, batch_size=self.batch_size)
            self.state_node_dict[state] = new_node
            return new_node

    def store(self, state, reward, policy, value):
        new_node = Node(state, reward, policy, value,
                        action_size=self.action_space, device=self.device, batch_size=self.batch_size)
        self.state_node_dict[state] = new_node

    def reset_nodes(self):
        self.lookup_table = {}
        self.state_node_dict = {}

    def compute_gain(self, rewards, v_l, i, l):
        reward_step = rewards[i:]
        reward_chunk = torch.stack(reward_step, 1).squeeze(2)  # squeeze out the 1D reward dim, so is now batch x length
        exponents = torch.arange(len(reward_step)).to(self.device)
        discounts = torch.pow(self.gamma * torch.ones_like(reward_chunk).to(self.device), exponents)
        v_l = v_l.squeeze(1)  # make v_l just batch size
        return torch.pow(self.gamma, l - i) * v_l + torch.sum(reward_chunk * discounts, dim=1)  # sum over the step length dim

    def get_root_policy(self):
        """
        Only call after running simulations
        """
        root_node = self.state_to_node(self.s0)

        return root_node.action_distribution()

    def setup_simulation(self, obs, k=None):
        if k is None:
            k = self.k

        self.reset_nodes()

        self.batch_size = obs.shape[0]

        s0 = self.model.representation_function(obs)
        p0, v0 = self.model.prediction_function(s0)
        current_node = Node(s0, torch.zeros([self.batch_size, 1]).to(self.device), p0, v0,
                            self.action_space, self.device, self.batch_size)  # root node
        self.state_node_dict[s0] = current_node
        self.s0 = s0
        s_i = s0

        return k, current_node, s_i

    def run_simulation(self, k, current_node, s_i):
        state_action = []
        rewards = []
        min_q = torch.Tensor([float("inf")]).to(self.device)
        max_q = torch.Tensor([float("-inf")]).to(self.device)

        for i in range(k - 1):
            a_i = self.get_action(current_node)
            state_action.append((s_i, a_i))

            s_i_prime, r_i = self.lookup(s_i, a_i)
            rewards.append(r_i)

            current_node = self.state_to_node(s_i_prime, r_i)
            s_i = s_i_prime

            min_q = torch.min(min_q, current_node.min_q())
            max_q = torch.max(max_q, current_node.max_q())

        a_l = self.get_action(current_node)
        state_action.append((s_i, a_l))

        r_l, s_l = self.model.dynamics_function(s_i, a_l)
        p_l, v_l = self.model.prediction_function(s_l)

        rewards.append(r_l)

        self.store(s_l, r_l, p_l, v_l)

        self.backup(state_action, rewards, v_l, k, min_q, max_q)

    def backup(self, states, rewards, v_l, k, min_q, max_q):
        for i in reversed(range(1, k)):
            g_i = self.compute_gain(rewards, v_l, i, k)

            s_i_1, a_i = states[i - 1]

            self.state_to_node(s_i_1, rewards[i - 1]).update(g_i, a_i, min_q, max_q)

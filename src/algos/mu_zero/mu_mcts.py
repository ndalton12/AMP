import torch


class Node:
    def __init__(self, state, reward, policy, value, action_size, device):
        self.q_values = torch.zeros(action_size).to(device)

        self.visits = torch.zeros(action_size).to(device)

        self.reward = reward
        self.state = state
        self.policy = policy
        self.value = value

    def number_visits(self, action):
        return self.visits[action]

    def update_num_visit(self, action):
        self.visits[action] += 1

    def get_total_visits(self):
        return torch.sum(self.visits)

    def q_value(self, action):
        return self.q_values[action]

    def set_q_value(self, q_val, action):
        self.q_values[action] = q_val

    def min_q(self):
        return torch.min(self.q_values)

    def max_q(self):
        return torch.max(self.q_values)

    def update(self, gain, action, min_q, max_q):

        action_index = action.item()

        n_a = self.number_visits(action_index)
        q_old = self.q_value(action_index)
        q_val = (n_a * q_old + gain) / (n_a + 1)

        q_val = (q_val - min_q) / (max_q - min_q)

        self.set_q_value(q_val, action_index)
        self.update_num_visit(action_index)

    def action_distribution(self):
        total = torch.sum(self.visits)

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
        total_visits = node.get_total_visits()
        term = (self.c1 + torch.log(total_visits + self.c2 + torch.Tensor([1]).to(self.device)) - torch.log(self.c2))

        if len(node.policy.shape) > 1:
            policy = torch.mean(node.policy, dim=0)
        else:
            policy = node.policy

        values = node.q_values + policy * term * torch.sqrt(total_visits) / (torch.Tensor([1]).to(self.device) + node.visits)

        if self.advantaged:
            values = values - node.value.squeeze(1)

        return torch.argmax(values)

    def lookup(self, state, action):
        if (state, action) in self.lookup_table:
            return self.lookup_table[(state, action)]
        else:
            reward, new_state = self.model.dynamics_function(state, action)
            self.lookup_table[(state, action)] = (new_state, reward)
            return new_state, reward

    def state_to_node(self, state, reward=0):
        if state in self.state_node_dict:
            return self.state_node_dict[state]
        else:
            policy, value = self.model.prediction_function(state)
            new_node = Node(state, reward, policy, value, action_size=self.action_space, device=self.device)
            self.state_node_dict[state] = new_node
            return new_node

    def store(self, state, reward, policy, value):
        new_node = Node(state, reward, policy, value, action_size=self.action_space, device=self.device)
        self.state_node_dict[state] = new_node

    def reset_nodes(self):
        self.lookup_table = {}
        self.state_node_dict = {}

    def compute_gain(self, rewards, v_l, i, l):
        reward_step = rewards[i:]
        reward_chunk = torch.flatten(torch.hstack(reward_step))
        exponents = torch.arange(len(reward_chunk)).to(self.device)
        discounts = torch.pow(self.gamma * torch.ones_like(reward_chunk).to(self.device), exponents)
        reward_chunk = reward_chunk.squeeze()
        discounts = discounts.squeeze()
        v_l = v_l.squeeze()
        return torch.sum(torch.pow(self.gamma, l - i) * v_l) + torch.dot(reward_chunk, discounts)

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

        s0 = self.model.representation_function(obs)
        p0, v0 = self.model.prediction_function(s0)
        current_node = Node(s0, torch.Tensor([0]).to(self.device), p0, v0, self.action_space, self.device)  # root node
        self.s0 = s0
        self.state_node_dict[s0] = current_node
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

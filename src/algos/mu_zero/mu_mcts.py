"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
and Alpha Zero implementation from rllib
"""
import torch


class Node:
    def __init__(self, state, reward, policy, action_size, device):
        self.q_values = torch.zeros(action_size).to(device)  # Q

        self.visits = torch.zeros(action_size).to(device)

        self.reward = reward
        self.state = state
        self.policy = policy

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

    def update(self, gain, action_index, min_q, max_q):

        n_a = self.number_visits(action_index)
        q_old = self.q_value(action_index)
        q_val = (n_a * q_old + gain) / (n_a + 1)

        q_val = (q_val - min_q) / (max_q - min_q)

        self.set_q_value(q_val, action_index)
        self.update_num_visit(action_index)

    def action_distribution(self):
        total = torch.sum(self.visits)

        return self.visits / total


class RootParentNode(Node):

    def __init__(self, initial_state, action_size, device):
        self.q_values = torch.zeros(action_size).to(device)  # Q
        self.visits = torch.zeros(action_size).to(device)  # N

        self.state = initial_state


class MCTS:
    def __init__(self, model, mcts_param, action_length):
        self.model = model
        self.k = mcts_param["k_sims"]
        self.c1 = mcts_param["c1"]
        self.c2 = mcts_param["c2"]
        self.gamma = mcts_param["gamma"]
        self.action_space = action_length
        self.device = model.device

        assert self.k > 1, "K simulations must be greater than 1"

        self.lookup_table = {}
        self.state_node_dict = {}

    def get_action(self, node):
        total_visits = node.get_total_visits()
        term = (self.c1 + torch.log(total_visits + self.c2 + 1) - torch.log(self.c2))

        values = node.q_values + node.policy * term * torch.sqrt(total_visits) / (1 + node.visits)

        return torch.argmax(values)

    def lookup(self, state, action):
        if (state, action) in self.lookup_table:
            return self.lookup_table[(state, action)]
        else:
            reward, new_state = self.model.dynamics_function(state, action)
            self.lookup_table[(state, action)] = (new_state, reward)

    def state_to_node(self, state, reward=0):
        if state in self.state_node_dict:
            return self.state_node_dict[state]
        else:
            policy, _ = self.model.prediction_function(state)
            new_node = Node(state, reward, policy, action_size=self.action_space, device=self.device)
            self.state_node_dict[state] = new_node
            return new_node

    def store(self, state, reward, policy):
        new_node = Node(state, reward, policy, action_size=self.action_space, device=self.device)
        self.state_node_dict[state] = new_node

    def reset_nodes(self):
        self.lookup_table = {}
        self.state_node_dict = {}

    def compute_gain(self, rewards, v_l, i, l):
        reward_step = rewards[i + 1:]
        exponents = torch.arange(len(reward_step))
        discounts = torch.pow(self.gamma * torch.ones(len(reward_step)), exponents)
        return torch.pow(torch.Tensor(self.gamma).to(self.device), l - i) * v_l + torch.dot(reward_step, discounts)

    def get_root_policy(self, obs):
        s0 = self.model.representation_function(obs)
        root_node = self.state_to_node(s0)

        return root_node.action_distribution()

    def simulation(self, obs, k=None):
        if k is None:
            k = self.k

        s0 = self.model.representation_function(obs)
        current_node = RootParentNode(s0, self.action_space, self.device)
        self.state_node_dict[s0] = current_node
        s_i = s0

        state_action = []
        rewards = []
        min_q = torch.Tensor(float("inf"))
        max_q = torch.Tensor(float("-inf"))

        for i in range(k):
            a_i = self.get_action(current_node)
            state_action.append((s_i, a_i))

            s_i_prime, r_i = self.lookup(s_i, a_i)
            rewards.append((rewards, r_i))

            current_node = self.state_to_node(s_i_prime, r_i)
            s_i = s_i_prime

            min_q = torch.min(min_q, current_node.min_q())
            max_q = torch.max(max_q, current_node.max_q())

        a_l = self.get_action(current_node)
        state_action.append((s_i, a_l))

        r_l, s_l = self.model.dynamics_function(s_i, a_l)
        p_l, v_l = self.model.prediction_function(s_l)

        rewards.append((rewards, r_l))

        self.store(s_l, r_l, p_l)

        self.backup(state_action, rewards, v_l, k, min_q, max_q)

    def backup(self, states, rewards, v_l, k, min_q, max_q):
        for i in reversed(range(1, k + 1)):
            g_i = self.compute_gain(rewards, v_l, i, k)

            s_i_1, a_i = states[i - 1]

            self.state_to_node(s_i_1, rewards[i - 1]).update(g_i, a_i, min_q, max_q)

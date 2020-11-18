import torch

from src.algos.mu_zero.mu_mcts import MCTS


class NomadMCTS(MCTS):
    def __init__(self, model, mcts_param, action_length, device, advantaged=True):
        super().__init__(model, mcts_param, action_length, device, advantaged)

        self.order = mcts_param["order"]

        assert self.k > self.order, "k length must be greater than the order"

    def lookup(self, states, action):
        state = states[-1]

        if (state, action) in self.lookup_table:
            return self.lookup_table[(state, action)]
        else:
            reward, new_state = self.model.dynamics_function(states, action, evolving=True)
            self.lookup_table[(state, action)] = (new_state, reward)
            return new_state, reward

    def run_simulation(self, k, current_node, s_i):
        state_action = []
        rewards = []
        cache = [s_i] * self.order
        min_q = torch.Tensor([float("inf")]).to(self.device)
        max_q = torch.Tensor([float("-inf")]).to(self.device)

        for i in range(k - 1):
            a_i = self.get_action(current_node)
            state_action.append((s_i, a_i))

            s_i_prime, r_i = self.lookup(cache, a_i)
            rewards.append(r_i)
            cache.append(s_i_prime)
            cache.pop(0)

            current_node = self.state_to_node(s_i_prime, r_i)
            s_i = s_i_prime

            min_q = torch.min(min_q, current_node.min_q())
            max_q = torch.max(max_q, current_node.max_q())

        a_l = self.get_action(current_node)
        state_action.append((s_i, a_l))

        r_l, s_l = self.model.dynamics_function(cache, a_l, evolving=True)
        p_l, v_l = self.model.prediction_function(s_l)

        rewards.append(r_l)

        self.store(s_l, r_l, p_l, v_l)

        self.backup(state_action, rewards, v_l, k, min_q, max_q)


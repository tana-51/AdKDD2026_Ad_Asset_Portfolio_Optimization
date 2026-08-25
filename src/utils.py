from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataclasses import dataclass
from dataset import diversity_cosine, generate_combinations
import itertools

seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

class GradientBasedPolicyDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        mask: np.ndarray,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()
        self.pscore = torch.from_numpy(pscore).float()
        self.q_hat = torch.from_numpy(q_hat).float()
        self.mask = torch.from_numpy(mask).long()
    
    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.mask[index],
        )
    
    
class RegBasedPolicyDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()

    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )
        

class PseudoInversePolicyDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        mask: np.ndarray,
        pscore_PI: np.ndarray,
        mask_PI: np.ndarray,
        action_binary: np.ndarray = None,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()
        self.pscore = torch.from_numpy(pscore).float()
        self.q_hat = torch.from_numpy(q_hat).float()
        self.mask = torch.from_numpy(mask).long()
        self.pscore_PI = torch.from_numpy(pscore_PI).float()
        self.mask_PI = torch.from_numpy(mask_PI).long()
        if action_binary is not None:
            self.action_binary = torch.from_numpy(action_binary).float()
    
    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        if hasattr(self, "action_binary"):
            return (
                self.context[index],
                self.action[index],
                self.reward[index],
                self.pscore[index],
                self.q_hat[index],
                self.mask[index],
                self.pscore_PI[index],
                self.mask_PI[index],
                self.action_binary[index],
            )
        else:
            return (
                self.context[index],
                self.action[index],
                self.reward[index],
                self.pscore[index],
                self.q_hat[index],
                self.mask[index],
                self.pscore_PI[index],
                self.mask_PI[index],
            )

class OursDataset(Dataset):
    def __init__(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: np.ndarray,
        pscore: np.ndarray,
        q_hat: np.ndarray,
        mask: np.ndarray,
        meta_info: np.ndarray,
        f_hat_pi_phi_2nd: np.ndarray,
    ):
        self.context = torch.from_numpy(context).float()
        self.action = torch.from_numpy(action).long()
        self.reward = torch.from_numpy(reward).float()
        self.pscore = torch.from_numpy(pscore).float()
        self.q_hat = torch.from_numpy(q_hat).float()
        self.mask = torch.from_numpy(mask).long()
        self.meta_info = torch.from_numpy(meta_info).long()
        self.f_hat_pi_phi_2nd = torch.from_numpy(f_hat_pi_phi_2nd).float()
    
    def __len__(self):
        return self.context.shape[0]
    
    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.mask[index],
            self.meta_info[index],
            self.f_hat_pi_phi_2nd[index],
        )

def obtain_f_x_a_from_f_x_b(
    f_hat_x_b: np.ndarray,
    dataset_train: dict,
) -> np.ndarray:
    f_hat_x_a = np.zeros((dataset_train["context"].shape[0], dataset_train["n_action_max"]))
    
    for i in range(f_hat_x_a.shape[0]):
        n_creative = dataset_train["mask_PI"][i].sum()
        all_comb_actions = np.array(generate_combinations([], n_creative))
        f_hat_x_a[i,:all_comb_actions.shape[0]] = (f_hat_x_b[i,:n_creative] * all_comb_actions).sum(axis=1)
    return f_hat_x_a

def obtain_pi_phi_2nd(
    f_hat_x_b: np.ndarray,
    dataset_train: dict,
    independent_reward_type: str = "ctr",
    is_lack: bool = False, 
):
    pi_phi_2nd = np.zeros((dataset_train["context"].shape[0], dataset_train["n_m_max"], dataset_train["n_action_max"]))
    group_effect_coeff_candidate = dataset_train["group_effect_coeff_candidate"]
    cost_effect_coeff_candidate = dataset_train["cost_effect_coeff_candidate"]
    fixed_unique_action_context = dataset_train["fixed_unique_action_context"]
    for i in range(pi_phi_2nd.shape[0]):
        n_creative = dataset_train["mask_PI"][i].sum()
        d_candidate = np.arange(1, n_creative+1)
        all_comb_actions = np.array(generate_combinations([], n_creative))
        m_candidate = np.array(list(itertools.product(d_candidate, group_effect_coeff_candidate, cost_effect_coeff_candidate)))
        action_context = fixed_unique_action_context[dataset_train["product_idx"][i]]
        for j in range(m_candidate.shape[0]):
            d = m_candidate[j,0]
            group_effect_coeff = m_candidate[j,1]
            action_index = np.zeros(int(d), dtype=int)
            unique_action_set = np.arange(n_creative)
            for k in range(int(d)):
                if dataset_train["group_effect_type"] == "cos_similarity":
                    if k==0:
                        select_idx = unique_action_set[np.argmax(f_hat_x_b[i,unique_action_set])]
                        
                    else:
                        div_list = []
                        for b in unique_action_set:
                            temp_action_context = np.concatenate((action_context[action_index[:k]], action_context[b][None, :]), axis=0)
                            if is_lack:
                                diversity = mean_pairwise_distance(np.array(temp_action_context)[:,5:])
                            else:
                                avg_sim, diversity = diversity_cosine(np.array(temp_action_context)[:,5:])
                            div_list.append(diversity)
                        if independent_reward_type == "ctr":
                            score = group_effect_coeff*np.array(div_list) + (1 - group_effect_coeff)*f_hat_x_b[i,unique_action_set]
                        elif independent_reward_type == "sum_click":
                            score = f_hat_x_b[i,unique_action_set] + group_effect_coeff*np.array(div_list)
                        select_idx = unique_action_set[np.argmax(score)]
                    
                    action_index[k] = select_idx
                    unique_action_set = unique_action_set[unique_action_set != select_idx]
            
            selected_action_context = np.zeros(all_comb_actions.shape[1])
            selected_action_context[action_index] = 1
            pi_phi_2nd[i,j,np.where(np.all(all_comb_actions == selected_action_context, axis=1))[0][0]]  = 1
    
    return pi_phi_2nd


def obtain_pi_phi_2nd_lack(
    f_hat_x_b: np.ndarray,
    dataset_train: dict,
):
    pi_phi_2nd = np.zeros((dataset_train["context"].shape[0], dataset_train["n_m_max_lack"], dataset_train["n_action_max"]))
    group_effect_coeff_candidate = dataset_train["group_effect_coeff_candidate"]
    cost_effect_coeff_candidate = dataset_train["cost_effect_coeff_candidate"]
    fixed_unique_action_context = dataset_train["fixed_unique_action_context"]
    for i in range(pi_phi_2nd.shape[0]):
        n_creative = dataset_train["mask_PI"][i].sum()
        d_candidate = np.arange(1, n_creative+1)
        all_comb_actions = np.array(generate_combinations([], n_creative))
        m_candidate = np.array(list(itertools.product(d_candidate, cost_effect_coeff_candidate)))
        action_context = fixed_unique_action_context[dataset_train["product_idx"][i]]
        for j in range(m_candidate.shape[0]):
            d = m_candidate[j,0]

            action_index = np.zeros(int(d), dtype=int)
            unique_action_set = np.arange(n_creative)
            for k in range(int(d)):
                if dataset_train["group_effect_type"] == "cos_similarity":
                    if k==0:
                        select_idx = unique_action_set[np.argmax(f_hat_x_b[i,unique_action_set])]
                        
                    else:
                        div_list = []
                        for b in unique_action_set:
                            temp_action_context = np.concatenate((action_context[action_index[:k]], action_context[b][None, :]), axis=0)
                            avg_sim, diversity = diversity_cosine(np.array(temp_action_context)[:,5:])
                            div_list.append(diversity)
                        score = f_hat_x_b[i,unique_action_set]
                        select_idx = unique_action_set[np.argmax(score)]
                    
                    action_index[k] = select_idx
                    unique_action_set = unique_action_set[unique_action_set != select_idx]
            
            selected_action_context = np.zeros(all_comb_actions.shape[1])
            selected_action_context[action_index] = 1
            pi_phi_2nd[i,j,np.where(np.all(all_comb_actions == selected_action_context, axis=1))[0][0]]  = 1
    
    return pi_phi_2nd




    
    
def obtain_meta_info_hist(m_candidate_list, pi_ours_x_m, mask_m, meta_info_candidate, meta_info_name):
    m_ratio = np.zeros((pi_ours_x_m.shape[0], len(meta_info_candidate)))
    if meta_info_name == "d":
        col = 0
    elif meta_info_name == "group_effect_coeff":
        col = 1
    elif meta_info_name == "cost_effect_coeff":
        col = 2

    for i in range(pi_ours_x_m.shape[0]):
        pi_for_x = pi_ours_x_m[i,:] * mask_m[i,:]
        for j, m in enumerate(meta_info_candidate):
            m_idx = np.where((m_candidate_list[i][:,col] == m))[0]
            m_ratio[i,j] = pi_for_x[m_idx].sum()
    return m_ratio.mean(axis=0)


def mean_pairwise_distance(X):
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    triu_idx = np.triu_indices_from(dist_matrix, k=1)
    return dist_matrix[triu_idx].mean()

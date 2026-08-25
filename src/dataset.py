from dataclasses import dataclass
import os
import joblib
from pathlib import Path
import pickle
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from obp.dataset import BaseRealBanditDataset
from obp.utils import softmax
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from scipy.stats import gmean
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from PIL import Image
import torch
from scipy.spatial.distance import pdist


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def generate_combinations(current_combination, n):
    if len(current_combination) == n:
        return [current_combination]

    combinations = []
    combinations.extend(generate_combinations(current_combination + [0], n))
    combinations.extend(generate_combinations(current_combination + [1], n))

    return combinations

def diversity_cosine(X):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    sim_matrix = X_norm @ X_norm.T

    N = len(X)
    sim_matrix_no_diag = sim_matrix - np.eye(N)

    if N <= 1:
        avg_sim = 1.0
    else:
        avg_sim = sim_matrix_no_diag.sum() / (N*(N-1))

    diversity = 1 - avg_sim

    return avg_sim, diversity



@dataclass
class ExtremeBanditDataset():
    n_components: int = 10
    reward_std: float = 1.0
    n_train: int = 1500
    random_state: int = 12345
    dim_context: int = 10
    n_unique_action_threshold: int = 7
    independent_reward_type: str = "ctr"
    group_effect_coeff: float = 0.5
    cost_effect_coeff: float = 0.1
    group_effect_type: str = "cos_similarity"
    group_effect_coeff_candidate: np.ndarray = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cost_effect_coeff_candidate: np.ndarray = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    beta_first: float = 0.0

    def __post_init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.data_path = project_root
        self.cache_dir = project_root / "saved_objects"
        self.feature_path = project_root / "image_features.pkl"
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.sc = StandardScaler()
        
        self.random_ = check_random_state(self.random_state)


    def do_pre_process(self,) -> None:
        cache_path = self.cache_dir / f"saved_objects_{self.n_unique_action_threshold}_{self.independent_reward_type}.joblib"

        if cache_path.exists():
            data = joblib.load(cache_path)
            self.df_all = data["df_all"]
            self.creative_dict = data["creative_dict"]
            self.independent_reward_dict = data["independent_reward_dict"]
            print("loaded saved_objects.joblib")
        else:
            self.df_all, self.creative_dict, self.independent_reward_dict = self.pre_process(self.data_path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "df_all": self.df_all,
                    "creative_dict": self.creative_dict,
                    "independent_reward_dict": self.independent_reward_dict,
                },
                cache_path,
            )


        path = self.feature_path
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.fixed_unique_action_context = []
        embeddings = []
        for product_id in self.creative_dict.keys():
            for image_name in self.creative_dict[product_id]:
                embeddings.append(data[image_name].reshape(1, -1))

        embeddings_10d = np.concatenate(embeddings, axis=0)
        

        self.fixed_product_context = np.zeros((len(self.creative_dict), self.n_components))
        idx = 0
        for i, product_id in enumerate(self.creative_dict.keys()):
            n_creative = len(self.creative_dict[product_id])
            self.fixed_unique_action_context.append(
                embeddings_10d[idx:idx+n_creative,:]
            )
            self.fixed_product_context[i,:] = embeddings_10d[idx:idx+n_creative,:].mean(axis=0)

            idx += n_creative

            
            
        
        

    def pre_process(
        self, file_path: Path, 
    ):
        """Preprocess raw dataset."""
        df_all = pd.read_csv(file_path / "list/train_data_list.txt", sep="\t", header=None, names=["product", "creative", "date", "impression", "click"])
        count_df = df_all.groupby("product")["creative"].nunique().reset_index()
        count_df.columns = ["product", "creative_count"]
        use_df = count_df[(count_df['creative_count'] <= 10) & (count_df['creative_count'] >= self.n_unique_action_threshold)]

        creative_dict = {}
        independent_reward_dict = {}
        for idx, row in use_df.iterrows():
            product_id = row['product']
            df_product = df_all[df_all['product'] == product_id]
            creative_list = df_product['creative'].unique().tolist()
            creative_dict[product_id] = creative_list

            independent_reward = np.zeros(len(creative_list))
            for i, creative_id in enumerate(creative_list):
                df_creative = df_product[df_product['creative'] == creative_id]
                if self.independent_reward_type == "ctr":
                    ctr = df_creative['click'].sum() / df_creative['impression'].sum()
                    independent_reward[i] = ctr
                elif self.independent_reward_type == "sum_click":
                    independent_reward[i] = df_creative['click'].mean()
            independent_reward_dict[product_id] = independent_reward
        
        return df_all, creative_dict, independent_reward_dict
        
        

    def train_pi_b(
        self,
    ) -> None:
        idx = self.random_.choice(self.contexts.shape[0], size=self.n_train, replace=False)
        contexts = self.contexts[idx]
        contexts = self.sc.fit_transform(self.pca.fit_transform(contexts))
        idx = self.random_.choice(self.n_train, size=self.n_train, replace=False)
        expected_rewards = self.train_data[idx]
        
        rewards = np.zeros(self.n_train)
        actions = np.zeros(self.n_train , dtype=int)
        
        expected_rewards_  = expected_rewards.copy()
        expected_rewards_[expected_rewards_!=0] = 1

        
        for i in range(expected_rewards.shape[0]):
            x = expected_rewards[i][np.nonzero(expected_rewards[i])]
            if len(x)==0:
                x = np.array([0])
            rewards[i] = gmean(x) 
            sampled_action = np.where(np.all(self.action_context == expected_rewards_[i],axis=1))
            actions[i] = sampled_action[0][0]
        
        q_x_m = np.zeros(self.n_train*self.n_comb_action).reshape(self.n_train,self.n_comb_action)
        q_x_m[np.arange(self.n_train),actions] = rewards
        
        noise = self.random_.uniform(0.0, 1.0, size=(self.n_train,self.n_comb_action))

        q_x_m += noise

        self.regressor = MultiOutputRegressor(Ridge(max_iter=500, random_state=12345))
        self.regressor.fit(contexts, q_x_m)


    def compute_pi_b(
        self,
        contexts: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        r_hat = self.regressor.predict(contexts)
        pi_b = softmax(r_hat * beta)
        return pi_b
    
    def caluculate_pi_0(
        self,
        independent_reward: np.ndarray,
        beta: float = -1.0,
        product_idx: int = 0,
        w: np.ndarray = None,
        group_effect_coeff_candidate: np.ndarray = None,
        cost_effect_coeff_candidate: np.ndarray = None,
    ):
        n_creative = len(independent_reward)
        d_candidate = np.arange(1, n_creative+1)
        m_candidate = np.array(list(itertools.product(d_candidate, group_effect_coeff_candidate, cost_effect_coeff_candidate)))
        all_comb_actions = np.array(generate_combinations([], n_creative))
        pi_0_a_given_x_m = np.zeros((m_candidate.shape[0], all_comb_actions.shape[0]))
        estimated_independent_reward = independent_reward + 0*self.random_.normal(0, 0.2, size=independent_reward.shape)
        for m_idx in range(m_candidate.shape[0]):
            d = m_candidate[m_idx][0]
            g = m_candidate[m_idx][1]
            c = m_candidate[m_idx][2]
            d_comb_actions = all_comb_actions[all_comb_actions.sum(axis=1) == d]
            expected_rewards = []
            for comb in d_comb_actions:
                if self.independent_reward_type == "ctr":
                    expected_reward = (1 - g)*np.sum(estimated_independent_reward[comb==1])
                    expected_reward += g * self.calc_group_effect(product_id=product_idx, comb=comb)
                    expected_reward -= c * comb.sum()
                elif self.independent_reward_type == "sum_click":
                    expected_reward = np.sum(estimated_independent_reward[comb==1])
                    expected_reward += g * self.calc_group_effect(product_id=product_idx, comb=comb)
                    expected_reward -= c * comb.sum()
                expected_rewards.append(expected_reward)
            expected_rewards = np.array(expected_rewards).reshape(1,-1)
            pi_0_a_given_x_m[m_idx,all_comb_actions.sum(axis=1) == d] = softmax(beta*expected_rewards).ravel()

        m_candidate_context = m_candidate.copy()
        logits = self.beta_first*(m_candidate_context @ w)
        logits = logits.reshape(1, -1)
        pi_0_m = softmax(logits)
        pi_0 = (pi_0_a_given_x_m * pi_0_m.T).sum(axis=0)
        
        return pi_0_m, pi_0_a_given_x_m, pi_0, m_candidate, all_comb_actions

    
    def obtain_pi_0_all(
        self,
        independent_reward: np.ndarray,
        beta: float = -1.0,
    ):
        n_creative = len(independent_reward)
        all_comb_actions = np.array(generate_combinations([], n_creative))
        d_candidate = np.arange(1, n_creative+1)
        m_candidate = np.array(list(itertools.product(d_candidate, self.group_effect_coeff_candidate, self.cost_effect_coeff_candidate)))

        w = self.random_.uniform(-1,1,m_candidate.shape[1])
        logits = self.beta_first*(m_candidate @ w)
        logits = logits.reshape(1, -1)
        p_first = softmax(logits)

        pi_0_all = np.zeros(len(all_comb_actions))
        for d in range(n_creative):
            p_d = p_first[0, m_candidate[:, 0] == d].sum()
            comb_actions = all_comb_actions[all_comb_actions.sum(axis=1) == d]
            expected_rewards = []
            index_list = []
            for idx, comb in enumerate(comb_actions):             
                expected_reward = np.sum(independent_reward[comb==1])
                expected_rewards.append(expected_reward)
                index_list.append(np.where(np.all(all_comb_actions == comb_actions[idx], axis=1))[0][0])
            if comb.sum() ==0:
                pi_0_all[np.array(index_list)] = 0.0
                continue
            else:
                expected_rewards = np.array(expected_rewards).reshape(1,-1)
                prob = softmax(beta*expected_rewards).ravel() 
                pi_0_all[np.array(index_list)] = prob*p_d
        return pi_0_all

    def obtain_batch_bandit_feedback(
        self, n_rounds: Optional[int] = None, beta: float = -1.0,
    ) -> dict:
        """Obtain batch logged bandit data."""
        
        n_product = len(self.creative_dict)
        product_idx = self.random_.choice(n_product, size=n_rounds, replace=True)
        context = self.fixed_product_context[product_idx,:]

        self.group_effect_coeff_x = (self.fixed_product_context * self.random_.normal(0,3.0,(len(self.fixed_product_context), 1))).sum(axis=1)
        self.group_effect_coeff_x = sigmoid(self.group_effect_coeff_x)




        fixed_q_x_a = self.obtain_expected_reward_matrix()
        q_x_a = []
        independent_reward = []
        n_creative_max = 0
        for i in range(len(product_idx)):
            q_x_a.append(fixed_q_x_a[product_idx[i]])
            independent_reward_arr = self.independent_reward_dict[list(self.creative_dict.keys())[product_idx[i]]]
            independent_reward.append(independent_reward_arr)
            if len(independent_reward_arr) > n_creative_max:
                n_creative_max = len(independent_reward_arr)

        

        actions = np.zeros(n_rounds , dtype=int)
        meta_info = np.zeros(n_rounds , dtype=int)
        meta_info_lack = np.zeros(n_rounds , dtype=int)
        pscore = np.zeros(n_rounds)
        pi_m_score = np.zeros(n_rounds)
        pi_m_score_lack = np.zeros(n_rounds)
        action_binary = np.zeros((n_rounds, n_creative_max))
        g_ = np.zeros(n_rounds)
        d_ = np.zeros(n_rounds)
        c_ = np.zeros(n_rounds)
        w = self.random_.uniform(-1,1,3)
        for i in np.arange(n_rounds):    

            (
                pi_0_m, 
                pi_0_a_given_x_m, 
                pi_0,
                m_candidate,
                all_comb_actions,
                ) = self.caluculate_pi_0(
                                    independent_reward=independent_reward[i], 
                                    beta=beta, 
                                    product_idx=product_idx[i],
                                    w=w,
                                    group_effect_coeff_candidate=self.group_effect_coeff_candidate,
                                    cost_effect_coeff_candidate=self.cost_effect_coeff_candidate,
                                )
            sample_m = self.random_.choice(np.arange(pi_0_m.shape[1]), p=pi_0_m.ravel())
            m_score = pi_0_m[0, sample_m]
            sample_action = self.random_.choice(np.arange(pi_0_a_given_x_m.shape[1]), p=pi_0_a_given_x_m[sample_m,:].ravel())
            score = pi_0[sample_action]
            action_binary_sample = all_comb_actions[sample_action]

            d = m_candidate[sample_m,0]
            g = m_candidate[sample_m,1]
            c = m_candidate[sample_m,2]
            m_candidate_lack = np.array(list(itertools.product(np.arange(1,len(independent_reward[i])+1), self.cost_effect_coeff_candidate)))
            m_score_lack = pi_0_m[0,(m_candidate[:,0]==d) & (m_candidate[:,2]==c)].sum()
            sample_m_lack  = np.where((m_candidate_lack[:,0]==d) & (m_candidate_lack[:,1]==c))[0][0]

            meta_info[i] = sample_m
            actions[i] = sample_action
            pscore[i] = score
            pi_m_score[i] = m_score
            pi_m_score_lack[i] = m_score_lack
            meta_info_lack[i] = sample_m_lack
            action_binary[i, :len(action_binary_sample)] = action_binary_sample
            g_[i] = g
            d_[i] = d
            c_[i] = c
        
        q_x_a_factual = np.zeros(n_rounds)
        for i in range(n_rounds):
            q_x_a_factual[i] = q_x_a[i][actions[i]]

        rewards = self.random_.normal(
                loc=q_x_a_factual,
                scale=self.reward_std,
            )
        
        pi_0_all_list = []
        n_action_max = 0
        for i , content in enumerate(self.independent_reward_dict.items()):
            product_id, independent_reward_arr = content
            (
                pi_0_m, 
                pi_0_a_given_x_m, 
                pi_0,
                m_candidate,
                all_comb_actions,
                ) = self.caluculate_pi_0(
                                    independent_reward=independent_reward_arr, 
                                    beta=beta, 
                                    product_idx=i,
                                    w=w,
                                    group_effect_coeff_candidate=self.group_effect_coeff_candidate,
                                    cost_effect_coeff_candidate=self.cost_effect_coeff_candidate,
                                )
            pi_0_all_list.append(pi_0)
            if len(fixed_q_x_a[i]) > n_action_max:
                n_action_max = len(fixed_q_x_a[i])
        
        mask = np.zeros((n_rounds,n_action_max), dtype=int)
        for i in range(n_rounds):
            mask[i,:len(fixed_q_x_a[product_idx[i]])] = 1

        pi_0_all_arr = np.zeros((n_product, n_action_max))
        fixed_q_x_a_arr = np.zeros((n_product, n_action_max))
        for i in range(n_product):
            pi_0_all_arr[i,:len(pi_0_all_list[i])] = pi_0_all_list[i]
            fixed_q_x_a_arr[i,:len(fixed_q_x_a[i])] = fixed_q_x_a[i]
        
        pscore_PI = np.zeros((n_rounds, n_creative_max))
        mask_PI = np.zeros((n_rounds, n_creative_max), dtype=int)
        for i  in range(n_rounds):
            n_creative = len(independent_reward[i])
            all_comb_actions = np.array(generate_combinations([], n_creative))
            action_indicator= all_comb_actions[actions[i]]
            for j in range(n_creative):
                ind = action_indicator[j]
                indices = np.where(all_comb_actions[:, j] == ind)[0]
                pscore_PI[i,j] = pi_0_all_arr[product_idx[i], indices].sum() 
                mask_PI[i,j] = 1
        
        m_max = np.array(list(itertools.product(np.arange(n_creative_max), self.group_effect_coeff_candidate, self.cost_effect_coeff_candidate))).shape[0]
        m_max_lack = np.array(list(itertools.product(np.arange(n_creative_max), self.cost_effect_coeff_candidate))).shape[0]
        mask_m = np.zeros((n_rounds, m_max), dtype=int)
        mask_m_lack = np.zeros((n_rounds, m_max_lack), dtype=int)
        m_candidate_list = []
        m_candidate_list_lack = []
        for i in range(n_rounds):
            n_creative = len(independent_reward[i])
            m_candidate = np.array(list(itertools.product(np.arange(n_creative), self.group_effect_coeff_candidate, self.cost_effect_coeff_candidate)))
            num_m_candidate = m_candidate.shape[0]
            mask_m[i,:num_m_candidate] = 1
            m_candidate_list.append(m_candidate)

            m_candidate_lack = np.array(list(itertools.product(np.arange(n_creative), self.cost_effect_coeff_candidate)))
            num_m_candidate_lack = m_candidate_lack.shape[0]
            mask_m_lack[i,:num_m_candidate_lack] = 1
            m_candidate_list_lack.append(m_candidate_lack)

        
        f_x_b = np.zeros((len(independent_reward), n_creative_max))
        for i in range(f_x_b.shape[0]):
            independent_reward_arr = independent_reward[i]
            f_x_b[i,:len(independent_reward_arr)] = independent_reward_arr
        


        return dict(
            n_rounds=n_rounds,
            n_product=n_product,
            n_creative_max=n_creative_max,
            product_idx=product_idx,
            fixed_unique_action_context=self.fixed_unique_action_context,
            context=context,
            fixed_product_context=self.fixed_product_context,
            action=actions,
            meta_info=meta_info,
            position=None,
            reward=rewards,
            fixed_q_x_a=fixed_q_x_a,
            df_all=self.df_all, 
            creative_dict=self.creative_dict, 
            independent_reward_dict=self.independent_reward_dict,
            pscore=pscore,
            pscore_PI=pscore_PI,
            pi_m_score=pi_m_score,
            pi_0_all_list=pi_0_all_list,
            n_action_max=n_action_max,
            n_m_max=m_max,
            mask=mask,
            mask_PI=mask_PI,
            mask_m=mask_m,
            pi_0_all_arr=pi_0_all_arr,
            fixed_q_x_a_arr=fixed_q_x_a_arr,
            q_x_a_arr=fixed_q_x_a_arr[product_idx],
            group_effect_type=self.group_effect_type,
            group_effect_coeff_candidate=self.group_effect_coeff_candidate,
            cost_effect_coeff_candidate=self.cost_effect_coeff_candidate,
            action_binary=action_binary,
            m_candidate_list=m_candidate_list,
            meta_info_lack=meta_info_lack,
            mask_m_lack=mask_m_lack,
            n_m_max_lack=m_max_lack,
            m_candidate_list_lack=m_candidate_list_lack,
            pi_m_score_lack=pi_m_score_lack,
            f_x_b=f_x_b,
            g_=g_,
            d_=d_,
            c_=c_,
        )
    
    def obtain_expected_reward_matrix(self,):
        expected_reward_list = []

        for i, content in enumerate(self.independent_reward_dict.items()):
            _, independent_reward_arr = content
            n_creative = len(independent_reward_arr)
            comb_actions = generate_combinations([], n_creative)
            expected_reward_sublist = []
            for comb in comb_actions:
                comb = np.array(comb)
                if comb.sum() ==0:
                    expected_reward = 0.0
                else:
                    if self.independent_reward_type == "ctr":
                        expected_reward = (1 - self.group_effect_coeff)*np.sum(independent_reward_arr[comb==1])
                        expected_reward += self.group_effect_coeff * self.calc_group_effect(product_id=i, comb=comb)
                        expected_reward -= self.cost_effect_coeff * comb.sum()
                    elif self.independent_reward_type == "sum_click":
                        expected_reward = np.sum(independent_reward_arr[comb==1])
                        expected_reward += self.group_effect_coeff * self.calc_group_effect(product_id=i, comb=comb)
                        expected_reward -= self.cost_effect_coeff * comb.sum()
                expected_reward_sublist.append(expected_reward)
            expected_reward_list.append(np.array(expected_reward_sublist))
            
        return expected_reward_list
    
    def calc_group_effect(
        self, product_id: int, comb: np.ndarray
    ) -> float:
        if self.group_effect_type == "cos_similarity":
            action_context = self.fixed_unique_action_context[product_id][comb==1]
            avg_sim, diversity = diversity_cosine(action_context[:,5:])
        return diversity


    @staticmethod
    def calc_ground_truth_policy_value(
        expected_reward: list, action_dist: list
    ) -> float:
        score = np.zeros(len(expected_reward))
        for i in range(len(action_dist)):
            score[i] = (expected_reward[i] * action_dist[i]).sum()
        
        return np.average(score)
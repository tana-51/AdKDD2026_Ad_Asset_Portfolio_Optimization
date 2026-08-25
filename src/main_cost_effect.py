"""
vary group cost coef
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import os

from dataset import ExtremeBanditDataset
from obp.dataset import linear_reward_function
from pandas import DataFrame
import pandas as pd
from sklearn.neural_network import MLPRegressor
from obp.ope import RegressionModel
from sklearn.utils import check_random_state
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from policylearners import (
    GradientBasedPolicyLearner,
    RegBasedPolicyLearner,
    PseudoInversePolicyLearner,
    IndependentRegBasedPolicyLearner,
    Ours,
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import wandb
from plot import plot
from utils import obtain_f_x_a_from_f_x_b, obtain_pi_phi_2nd, obtain_meta_info_hist

seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

@hydra.main(config_path="../conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    num_runs = cfg.setting.base_setting.num_runs
    epoch = cfg.setting.base_setting.epoch
    cost_effect_coeff_list = cfg.setting.vary.cost_effect_coeff_list
    use_wandb = cfg.setting.base_setting.use_wandb

    print("===== Configuration =====")
    print(cost_effect_coeff_list)

    dataset = ExtremeBanditDataset(
        dim_context = cfg.setting.dataset.dim_context,
        n_unique_action_threshold = cfg.setting.dataset.n_unique_action_threshold,
        independent_reward_type = cfg.setting.dataset.independent_reward_type,
        group_effect_coeff = cfg.setting.dataset.group_effect_coeff,
        cost_effect_coeff = cost_effect_coeff_list[0],
        group_effect_type = cfg.setting.dataset.group_effect_type,
        group_effect_coeff_candidate = np.array(cfg.setting.dataset.group_effect_coeff_candidate),
        cost_effect_coeff_candidate = np.array(cfg.setting.dataset.cost_effect_coeff_candidate),
        random_state = cfg.setting.dataset.random_state,
        beta_first = cfg.setting.dataset.beta_first,
        reward_std=cfg.setting.dataset.reward_std,
    )

    result_df_list = []
    result_df = DataFrame()
    meta_info_df_list = []
    for cost_effect_coeff in cost_effect_coeff_list:

        dataset.cost_effect_coeff = cost_effect_coeff
        if cost_effect_coeff == cost_effect_coeff_list[0]:
            dataset.do_pre_process()

        test_bandit_data = dataset.obtain_batch_bandit_feedback(
            n_rounds = cfg.setting.base_setting.test_data_size, 
            beta = cfg.setting.base_setting.beta,
        )
            
        pi_0_value = dataset.calc_ground_truth_policy_value(
                expected_reward=test_bandit_data["fixed_q_x_a"], 
                action_dist=test_bandit_data["pi_0_all_list"],
            )
        print(f"pi_0_value: {pi_0_value}")
        row_max = np.max(test_bandit_data["fixed_q_x_a_arr"], axis=1)
        max_value = row_max.mean()
        print(f"max_value: {max_value}")

        test_policy_value_list = []
        m_ratio_list = []
        for _ in tqdm(range(num_runs), desc=f"cost_effect_coeff={cost_effect_coeff}..."):
            test_value_of_learned_policies = dict()
            
            train_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds = cfg.setting.base_setting.train_data_size, 
                beta = cfg.setting.base_setting.beta,
            )

            test_value_of_learned_policies["pi_0"] = pi_0_value
            
            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="IPS"
                )
            ips = GradientBasedPolicyLearner(
                dim_context = dataset.dim_context, 
                n_action_max=train_bandit_data["n_action_max"], 
                epoch = epoch,
                device=device,
                use_wandb=use_wandb,
            )
            ips.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_ips = ips.predict(test_bandit_data)
            mask = test_bandit_data["mask"]
            ips_value = ((test_bandit_data["q_x_a_arr"] * pi_ips)*mask).sum(1).mean()
            test_value_of_learned_policies["ips"] = ips_value
            if use_wandb:
                wandb.finish()

            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="RegBased"
                )
            reg = RegBasedPolicyLearner(
                dim_context = dataset.dim_context, 
                n_action_max=train_bandit_data["n_action_max"],  
                epoch = epoch,
                device=device,
                use_wandb=use_wandb,
            )
            reg.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_reg = reg.predict(test_bandit_data)
            reg_value = ((test_bandit_data["q_x_a_arr"] * pi_reg)*mask).sum(1).mean()
            test_value_of_learned_policies["reg"] = reg_value
            if use_wandb:
                wandb.finish()

            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="PI"
                )
            PI = PseudoInversePolicyLearner(
                dim_context = dataset.dim_context, 
                n_action_max=train_bandit_data["n_action_max"],  
                epoch = epoch,
                device=device,
                use_wandb=use_wandb,
            )
            PI.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_PI = PI.predict(test_bandit_data)
            mask = test_bandit_data["mask"]
            PI_value = ((test_bandit_data["q_x_a_arr"] * pi_PI)*mask).sum(1).mean()
            test_value_of_learned_policies["pi"] = PI_value
            if use_wandb:
                wandb.finish()

            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="IndepReg"
                )
            indep_reg = IndependentRegBasedPolicyLearner(
                dim_context = dataset.dim_context, 
                n_creative_max=train_bandit_data["n_creative_max"],  
                epoch = epoch,
                device=device,
                d=cfg.setting.base_setting.d_independent_reg,
                use_wandb=use_wandb,
            )

            indep_reg.fit(dataset=train_bandit_data, dataset_test=test_bandit_data)
            pi_indep_reg = indep_reg.predict(dataset_test=test_bandit_data, d=cfg.setting.base_setting.d_independent_reg)
            mask = test_bandit_data["mask"]
            indep_reg_value = ((test_bandit_data["q_x_a_arr"] * pi_indep_reg)*mask).sum(1).mean()
            test_value_of_learned_policies["indep_reg"] = indep_reg_value
            if use_wandb:
                wandb.finish()


            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="Ours"
                )
            ours = Ours(
                dim_context = dataset.dim_context, 
                n_m_max=train_bandit_data["n_m_max"],  
                epoch = epoch,
                device=device,
                use_wandb=use_wandb,
            )
            f_hat_x_b = indep_reg.predict_q_hat(dataset_test=train_bandit_data)
            f_hat_x_a = obtain_f_x_a_from_f_x_b(f_hat_x_b=f_hat_x_b, dataset_train=train_bandit_data)
            pi_phi_2nd = obtain_pi_phi_2nd(f_hat_x_b=f_hat_x_b, dataset_train=train_bandit_data)
            f_hat_pi_phi_2nd = (pi_phi_2nd*f_hat_x_a[:, None, :]).sum(axis=2)

            f_hat_x_b_test = indep_reg.predict_q_hat(dataset_test=test_bandit_data)
            pi_phi_2nd_test = obtain_pi_phi_2nd(f_hat_x_b=f_hat_x_b_test, dataset_train=test_bandit_data)
            ours.fit(
                dataset=train_bandit_data, 
                dataset_test=test_bandit_data, 
                q_hat=f_hat_x_a, 
                f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
                pi_phi_2nd=pi_phi_2nd,
                pi_phi_2nd_test=pi_phi_2nd_test,
            )
            pi_ours_x_m = ours.predict(dataset_test=test_bandit_data)
            pi_ours_x_a = (pi_ours_x_m[:, :, None] * pi_phi_2nd_test).sum(axis=1)
            mask = test_bandit_data["mask"]
            ours_value = ((test_bandit_data["q_x_a_arr"] * pi_ours_x_a)*mask).sum(1).mean()
            test_value_of_learned_policies["ours"] = ours_value
            meta_info_hist = obtain_meta_info_hist(
                m_candidate_list=test_bandit_data["m_candidate_list"],
                pi_ours_x_m=pi_ours_x_m,
                mask_m=test_bandit_data["mask_m"],
                meta_info_candidate=[i+1 for i in range(10)],
                meta_info_name="d",
            )
            m_ratio_list.append({"cost_effect_coeff": cost_effect_coeff, "meta_info_hist": meta_info_hist})
            if use_wandb:
                wandb.finish()

            if use_wandb:
                wandb.init(
                    project=cfg.setting.wandb.project_name, 
                    name=f"c_coef{cost_effect_coeff}_seed{_}",
                    group="Ours_lack_group_effect"
                )
            ours_lack = Ours(
                dim_context = dataset.dim_context, 
                n_m_max=train_bandit_data["n_m_max"],  
                epoch = epoch,
                device=device,
                use_wandb=use_wandb,
                hidden_dim=cfg.setting.base_setting.hidden_dim,
            )
            f_hat_x_b = indep_reg.predict_q_hat(dataset_test=train_bandit_data)
            f_hat_x_a = obtain_f_x_a_from_f_x_b(f_hat_x_b=f_hat_x_b, dataset_train=train_bandit_data)
            pi_phi_2nd = obtain_pi_phi_2nd(f_hat_x_b=f_hat_x_b, dataset_train=train_bandit_data, is_lack=True)
            f_hat_pi_phi_2nd = (pi_phi_2nd*f_hat_x_a[:, None, :]).sum(axis=2)

            f_hat_x_b_test = indep_reg.predict_q_hat(dataset_test=test_bandit_data)
            pi_phi_2nd_test = obtain_pi_phi_2nd(f_hat_x_b=f_hat_x_b_test, dataset_train=test_bandit_data, is_lack=True)
            ours_lack.fit(
                dataset=train_bandit_data, 
                dataset_test=test_bandit_data, 
                q_hat=f_hat_x_a, 
                f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
                pi_phi_2nd=pi_phi_2nd,
                pi_phi_2nd_test=pi_phi_2nd_test,
            )
            pi_ours_x_m = ours_lack.predict(dataset_test=test_bandit_data)
            pi_ours_x_a = (pi_ours_x_m[:, :, None] * pi_phi_2nd_test).sum(axis=1)
            mask = test_bandit_data["mask"]
            ours_value_lack = ((test_bandit_data["q_x_a_arr"] * pi_ours_x_a)*mask).sum(1).mean()
            test_value_of_learned_policies["ours_lack"] = ours_value_lack
            if use_wandb:
                wandb.finish()

            test_policy_value_list.append(test_value_of_learned_policies)


        result_df = DataFrame(test_policy_value_list).stack().reset_index(1)\
            .rename(columns={"level_1": "method", 0: "value"})
        result_df["cost_effect_coeff"] = cost_effect_coeff
        result_df["pi_0_value"] = pi_0_value
        result_df["max_value"] = max_value
        result_df["rel_value"] = result_df["value"] / pi_0_value
        result_df["rel_value_max"] = result_df["value"] / max_value
        result_df_list.append(result_df)

        meta_info_df = DataFrame(m_ratio_list)
        meta_info_df_list.append(meta_info_df)

        tqdm.write("=====" * 15)

        save_path = f"./result/{cfg.setting.wandb.train_name}/{cfg.setting.dataset.independent_reward_type}/beta_first={cfg.setting.dataset.beta_first}/"
        result_df = pd.concat(result_df_list).reset_index(level=0)
        result_df.to_csv(f"{save_path}cost_effect_coeff.csv")
        meta_info_df = pd.concat(meta_info_df_list).reset_index(level=0)
        meta_info_df.to_csv(f"{save_path}meta_info_cost_effect_coeff.csv")
        plot(
            result_df=result_df,
            varying_name="cost_effect_coeff",
            varying_list=cost_effect_coeff_list,
            axis_name="Cost Effect Coefficient",
            log_scale=False,
            save_path=save_path,
        )



if __name__ == "__main__":
    print("======= Starting main =======")
    main()

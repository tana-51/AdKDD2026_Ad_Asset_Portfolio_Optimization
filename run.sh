# 仮想環境を有効化
source .venv/bin/activate

# main_group_effect.py
uv run ./src/main_group_effect.py \
    setting.wandb.train_name="group_effect_coeff" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=1 \
    setting.dataset.cost_effect_coeff=0.02  \
    setting.vary.group_effect_coeff_list="[0.0, 0.25, 0.5, 0.75, 1.0]" \
    setting.dataset.group_effect_coeff_candidate="[0.0, 0.25, 0.5, 0.75, 1.0]"  \
    setting.dataset.cost_effect_coeff_candidate="[0.0, 0.01, 0.02, 0.03, 0.04]"  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6 \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64

# main_cost_effect.py
uv run ./src/main_cost_effect.py \
    setting.wandb.train_name="cost_effect_coeff" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25 \
    setting.dataset.group_effect_coeff=0.5  \
    setting.vary.group_effect_coeff_list="[0.0, 0.25, 0.5, 0.75, 1.0]" \
    setting.dataset.group_effect_coeff_candidate="[0.0, 0.25, 0.5, 0.75, 1.0]"  \
    setting.vary.cost_effect_coeff_list="[0.0, 0.01, 0.02, 0.03, 0.04]" \
    setting.dataset.cost_effect_coeff_candidate="[0.0, 0.01, 0.02, 0.03, 0.04]"  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6 \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64

# main_n_creative.py
uv run ./src/main_n_creative.py \
    setting.wandb.train_name="n_creative_threshold" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25 \
    setting.dataset.group_effect_coeff=0.5  \
    setting.dataset.cost_effect_coeff=0.02  \
    setting.vary.n_unique_action_threshold_list="[5,6,7,8,9]" \
    setting.dataset.group_effect_coeff_candidate="[0.0, 0.25, 0.5, 0.75, 1.0]"  \
    setting.dataset.cost_effect_coeff_candidate="[0.0, 0.01, 0.02, 0.03, 0.04]"  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6  \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64

# main_beta.py
uv run ./src/main_beta.py \
    setting.wandb.train_name="beta" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25 \
    setting.dataset.group_effect_coeff=0.5  \
    setting.dataset.cost_effect_coeff=0.02  \
    setting.vary.beta_list="[-5.0, -2.5, 0.0, 2.5, 5.0]" \
    setting.dataset.group_effect_coeff_candidate="[0.0, 0.25, 0.5, 0.75, 1.0]"  \
    setting.dataset.cost_effect_coeff_candidate="[0.0, 0.01, 0.02, 0.03, 0.04]"  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6  \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64

# main_beta1.py
uv run ./src/main_beta1.py \
    setting.wandb.train_name="beta1" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25 \
    setting.dataset.group_effect_coeff=0.5  \
    setting.dataset.cost_effect_coeff=0.02  \
    setting.vary.beta_list="[-2.0, -1.0, 0.0, 1.0, 2.0]" \
    setting.dataset.group_effect_coeff_candidate="[0.0, 0.25, 0.5, 0.75, 1.0]"  \
    setting.dataset.cost_effect_coeff_candidate="[0.0, 0.01, 0.02, 0.03, 0.04]"  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6  \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64

# main_random_sample_linspace.py
uv run ./src/main_random_sample_linspace.py \
    setting.wandb.project_name="Hakuhodo_test_last" \
    setting.wandb.train_name="random_sample_linspace_num_runs25_8" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25 \
    setting.dataset.group_effect_coeff=0.5  \
    setting.dataset.cost_effect_coeff=0.02  \
    setting.dataset.independent_reward_type="ctr" \
    setting.dataset.n_unique_action_threshold=6 \
    setting.base_setting.train_data_size=2000 \
    setting.base_setting.test_data_size=10000 \
    setting.base_setting.d_independent_reg=4 \
    setting.dataset.beta_first=-2.0 \
    setting.base_setting.beta=5.0 \
    setting.base_setting.epoch=100 \
    setting.base_setting.hidden_dim=64
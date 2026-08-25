from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from dataclasses import dataclass
from utils import (
    GradientBasedPolicyDataset,
    RegBasedPolicyDataset,
    PseudoInversePolicyDataset,
    OursDataset,
)
from typing import List
from obp.utils import softmax
import wandb
from dataset import generate_combinations

seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

@dataclass
class RegBasedPolicyLearner:
    dim_context : int
    n_action_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_action_max)
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test):
        
            
        mydataset = RegBasedPolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_ = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                q_hat = self.nn_model(x_)
                idx = torch.arange(a_.shape[0], dtype=torch.long)
                loss = ((r_ - q_hat[idx, a_]) ** 2).mean()
                loss.backward()
                optimizer.step()

                minibatch_train_loss += loss.item()
            
            if self.use_wandb:
                pi_train = self.predict(dataset)
                train_value = ((q_x_a_train * pi_train)*dataset["mask"]).sum(1).mean()
                pi_test = self.predict(dataset_test)
                test_value = ((q_x_a_test * pi_test)*dataset_test["mask"]).sum(1).mean()
                minibatch_train_loss /= (i+1)
                wandb.log(
                    {
                        "train_loss": minibatch_train_loss,
                        "train_value": train_value,
                        "test_value": test_value,
                        "epoch": e
                    }
                )

    def predict(self, dataset_test: np.ndarray, beta: float = 10):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask = torch.from_numpy(dataset_test["mask"]).long().to(self.device)
        q_hat = self.nn_model(x)
        masked_logits = q_hat.masked_fill(mask == 0, float('-inf'))
        pi = self.softmax(beta*masked_logits)

        return pi.detach().cpu().numpy()

    
    def predict_q_hat(self, dataset_test):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float()

        return self.nn_model(x).detach().cpu().numpy()
    
    
    
@dataclass
class GradientBasedPolicyLearner:
    dim_context : int
    n_action_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_action_max),
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat: np.ndarray = None):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_action_max))
            
        mydataset = GradientBasedPolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pscore"], 
            q_hat = q_hat, 
            mask = dataset["mask"],
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_,p,q_hat_,mask = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                logits = self.nn_model(x_)
                masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                pi = self.softmax(masked_logits)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi=pi,
                    mask=mask,
                ).mean()
                loss.backward()
                optimizer.step()
                minibatch_train_loss += loss.item()
            
            if self.use_wandb:
                pi_train = self.predict(dataset)
                train_value = ((q_x_a_train * pi_train)*dataset["mask"]).sum(1).mean()
                pi_test = self.predict(dataset_test)
                test_value = ((q_x_a_test * pi_test)*dataset_test["mask"]).sum(1).mean()
                minibatch_train_loss /= (i+1)
                wandb.log(
                    {
                        "train_loss": minibatch_train_loss,
                        "train_value": train_value,
                        "test_value": test_value,
                        "epoch": e
                    }
                )
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, a] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, a]
        estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, a]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask = torch.from_numpy(dataset_test["mask"]).long().to(self.device)

        logits = self.nn_model(x)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        pi = self.softmax(masked_logits)
        return pi.detach().cpu().numpy()
    

@dataclass
class PseudoInversePolicyLearner:
    dim_context : int
    n_action_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_action_max),
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat: np.ndarray = None):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_action_max))
            
        mydataset = PseudoInversePolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pscore"], 
            q_hat = q_hat, 
            mask = dataset["mask"],
            pscore_PI = dataset["pscore_PI"],
            mask_PI = dataset["mask_PI"],
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_,p,q_hat_,mask, pscore_PI, mask_PI = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                logits = self.nn_model(x_)
                masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                pi = self.softmax(masked_logits)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi=pi,
                    mask=mask,
                    pscore_PI=pscore_PI,
                    mask_PI=mask_PI,
                )
                loss.backward()
                optimizer.step()
                minibatch_train_loss += loss.item()
            
            if self.use_wandb:
                pi_train = self.predict(dataset)
                train_value = ((q_x_a_train * pi_train)*dataset["mask"]).sum(1).mean()
                pi_test = self.predict(dataset_test)
                test_value = ((q_x_a_test * pi_test)*dataset_test["mask"]).sum(1).mean()
                minibatch_train_loss /= (i+1)
                wandb.log(
                    {
                        "train_loss": minibatch_train_loss,
                        "train_value": train_value,
                        "test_value": test_value,
                        "epoch": e
                    }
                )
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        mask: torch.Tensor,
        pscore_PI: torch.Tensor,
        mask_PI: torch.Tensor,
    ) -> torch.Tensor:



        current_pi = pi.detach()
        pi_0_list = []
        action_dist_PI_all_list = []
        action_dist_PI_with_grad_all_list = []
        reward_list = []
        for i in range(mask_PI.shape[0]):
            n_creative = mask_PI[i].sum()
            all_comb_actions = torch.tensor(generate_combinations([], n_creative)).to(self.device)
            action_dist_PI_list = []
            action_dist_PI_with_grad_list = []
            action_indicator= all_comb_actions[a[i]]
            for j in range(n_creative):
                ind = action_indicator[j]
                indices = (all_comb_actions[:, j] == ind).nonzero(as_tuple=True)[0]

                action_dist_PI_list.append(current_pi[i,indices].sum() / n_creative)
                action_dist_PI_with_grad_list.append(pi[i,indices].sum() / n_creative)
            action_dist_PI_all_list.append(torch.tensor(action_dist_PI_list).to(self.device))
            action_dist_PI_with_grad_all_list.append(torch.stack(action_dist_PI_with_grad_list))
            pi_0_list.append(pscore_PI[i,:n_creative])
            reward_list.append(torch.tensor([r[i]]*n_creative).to(self.device))
        
        pi_0_PI = torch.cat(pi_0_list)
        action_dist_PI = torch.cat(action_dist_PI_all_list)
        action_dist_PI_with_grad = torch.cat(action_dist_PI_with_grad_all_list)
        rewards_PI = torch.cat(reward_list)

        iw_PI = action_dist_PI / pi_0_PI
        log_prob = torch.log(action_dist_PI_with_grad + self.log_eps)
        estimated_policy_grad_arr = (iw_PI * log_prob * rewards_PI).sum()

        return estimated_policy_grad_arr / a.shape[0]

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask = torch.from_numpy(dataset_test["mask"]).long().to(self.device)

        logits = self.nn_model(x)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        pi = self.softmax(masked_logits)
        return pi.detach().cpu().numpy()
    

@dataclass
class IndependentRegBasedPolicyLearner:
    dim_context : int
    n_creative_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    d: int = 5
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_creative_max)
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test):

        mydataset = PseudoInversePolicyDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pscore"], 
            q_hat = np.zeros((dataset["context"].shape[0], self.n_creative_max)), 
            mask = dataset["mask"],
            pscore_PI = dataset["pscore_PI"],
            mask_PI = dataset["mask_PI"],
            action_binary = dataset["action_binary"],
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_,p,q_hat_,mask, pscore_PI, mask_PI, action_binary = [t.to(self.device) for t in batch]

                optimizer.zero_grad()
                q_hat = self.nn_model(x_)
                q_sum = (q_hat * action_binary).sum(1)
                loss = ((r_ - q_sum) ** 2).mean()
                loss.backward()
                optimizer.step()

                minibatch_train_loss += loss.item()
            if self.use_wandb:
                if e in [0, 6, 12, 18, 24, 29]:
                    pi_train = self.predict(dataset, d=self.d)
                    train_value = ((q_x_a_train * pi_train)*dataset["mask"]).sum(1).mean()
                    pi_test = self.predict(dataset_test, d=self.d)
                    test_value = ((q_x_a_test * pi_test)*dataset_test["mask"]).sum(1).mean()
                    minibatch_train_loss /= (i+1)
                    wandb.log(
                        {
                            "train_loss": minibatch_train_loss,
                            "train_value": train_value,
                            "test_value": test_value,
                            "epoch": e
                        }
                    )

    def predict(self, dataset_test: np.ndarray, beta: float = 10, d = 5):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask_PI = torch.from_numpy(dataset_test["mask_PI"]).long().to(self.device)
        mask = torch.from_numpy(dataset_test["mask"]).long().to(self.device)
        q_hat = self.nn_model(x)

        indices_list = []
        for i in range(mask_PI.shape[0]):
            n_creative = mask_PI[i].sum()
            all_comb_actions = torch.tensor(generate_combinations([], n_creative)).to(self.device)
            if n_creative < d:
                raise AssertionError("n_creative < d")
            else:
                d_use = d
            topd_indices = torch.topk(q_hat[i,:n_creative], d_use).indices
            action_indicator = torch.zeros(n_creative).long().to(self.device)
            action_indicator[topd_indices] = 1

            indices = (all_comb_actions == action_indicator).all(dim=1).nonzero(as_tuple=True)[0]

            indices_list.append(indices)
        
        indices = torch.cat(indices_list)
        pi = torch.zeros((x.shape[0], mask.shape[1])).to(self.device)
        pi[torch.arange(pi.shape[0]), indices] = 1

        return pi.detach().cpu().numpy()

    
    def predict_q_hat(self, dataset_test):
        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)

        return self.nn_model(x).detach().cpu().numpy()


@dataclass
class Ours:
    dim_context : int
    n_m_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_m_max),
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat, f_hat_pi_phi_2nd, pi_phi_2nd, pi_phi_2nd_test):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_m_max))
            
        mydataset = OursDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pi_m_score"], 
            q_hat = q_hat, 
            mask = dataset["mask_m"],
            meta_info = dataset["meta_info"],
            f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_,p,q_hat_,mask, meta_info, f_hat_pi_phi_2nd = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                logits = self.nn_model(x_)
                masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                pi = self.softmax(masked_logits)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi=pi,
                    mask=mask,
                    meta_info=meta_info,
                    f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
                ).mean()
                loss.backward()
                optimizer.step()
                minibatch_train_loss += loss.item()
            
            if self.use_wandb:
                pi_train_x_m = self.predict(dataset)
                pi_train_x_a = (pi_train_x_m[:, :, None] * pi_phi_2nd).sum(axis=1)
                train_value = ((q_x_a_train * pi_train_x_a)*dataset["mask"]).sum(1).mean()
                pi_test_x_m = self.predict(dataset_test)
                pi_test_x_a = (pi_test_x_m[:, :, None] * pi_phi_2nd_test).sum(axis=1)
                test_value = ((q_x_a_test * pi_test_x_a)*dataset_test["mask"]).sum(1).mean()
                minibatch_train_loss /= (i+1)
                wandb.log(
                    {
                        "train_loss": minibatch_train_loss,
                        "train_value": train_value,
                        "test_value": test_value,
                        "epoch": e
                    }
                )
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        mask: torch.Tensor,
        meta_info: torch.Tensor,
        f_hat_pi_phi_2nd: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, meta_info] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, meta_info]
        estimated_policy_grad_arr += torch.sum(f_hat_pi_phi_2nd * current_pi * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, meta_info]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask = torch.from_numpy(dataset_test["mask_m"]).long().to(self.device)

        logits = self.nn_model(x)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        pi = self.softmax(masked_logits)
        return pi.detach().cpu().numpy()
    

@dataclass
class OursLack:
    dim_context : int
    n_m_max : int
    batch_size : int = 16
    epoch : int = 30
    imit_reg: float = 0.0
    log_eps: float = 1e-10
    device: str = "cpu"
    use_wandb: bool = False
    hidden_dim: int = 30
    
    def __post_init__(self, ):
        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        
        self.nn_model = nn.Sequential(
            nn.Linear(self.dim_context, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_m_max),
        ).to(self.device)

        self.softmax = nn.Softmax(dim=1)
        
        self.train_loss = []
        self.train_value = []
        self.test_value = []
    
    def fit(self,dataset, dataset_test, q_hat, f_hat_pi_phi_2nd, pi_phi_2nd, pi_phi_2nd_test):
        
        if q_hat is None:
            q_hat = np.zeros((dataset["context"].shape[0], self.n_m_max))
            
        mydataset = OursDataset(
            context = dataset["context"], 
            action = dataset["action"], 
            reward = dataset["reward"], 
            pscore = dataset["pi_m_score_lack"], 
            q_hat = q_hat, 
            mask = dataset["mask_m_lack"],
            meta_info = dataset["meta_info_lack"],
            f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
        )
        
        train_dataloader = DataLoader(mydataset, batch_size=self.batch_size, num_workers=0)
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        q_x_a_train, q_x_a_test = dataset["q_x_a_arr"], dataset_test["q_x_a_arr"]
        
        
        for e in range(self.epoch):
            self.nn_model.train()
            minibatch_train_loss = 0
            train_value = 0
            test_value = 0
            for i, batch in enumerate(train_dataloader):
                x_,a_,r_,p,q_hat_,mask, meta_info, f_hat_pi_phi_2nd = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                logits = self.nn_model(x_)
                masked_logits = logits.masked_fill(mask == 0, float('-inf'))
                pi = self.softmax(masked_logits)
                loss = -self._estimate_policy_gradient(
                    a=a_,
                    r=r_,
                    pscore=p,
                    q_hat=q_hat_,
                    pi=pi,
                    mask=mask,
                    meta_info=meta_info,
                    f_hat_pi_phi_2nd=f_hat_pi_phi_2nd,
                ).mean()
                loss.backward()
                optimizer.step()
                minibatch_train_loss += loss.item()
            
            if self.use_wandb:
                pi_train_x_m = self.predict(dataset)
                pi_train_x_a = (pi_train_x_m[:, :, None] * pi_phi_2nd).sum(axis=1)
                train_value = ((q_x_a_train * pi_train_x_a)*dataset["mask"]).sum(1).mean()
                pi_test_x_m = self.predict(dataset_test)
                pi_test_x_a = (pi_test_x_m[:, :, None] * pi_phi_2nd_test).sum(axis=1)
                test_value = ((q_x_a_test * pi_test_x_a)*dataset_test["mask"]).sum(1).mean()
                minibatch_train_loss /= (i+1)
                wandb.log(
                    {
                        "train_loss": minibatch_train_loss,
                        "train_value": train_value,
                        "test_value": test_value,
                        "epoch": e
                    }
                )
    
    def _estimate_policy_gradient(
        self,
        a: torch.Tensor,
        r: torch.Tensor,
        pscore: torch.Tensor,
        q_hat: torch.Tensor,
        pi: torch.Tensor,
        mask: torch.Tensor,
        meta_info: torch.Tensor,
        f_hat_pi_phi_2nd: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx = torch.arange(a.shape[0], dtype=torch.long)

        q_hat_factual = q_hat[idx, a]
        iw = current_pi[idx, meta_info] / pscore
        estimated_policy_grad_arr = iw * (r - q_hat_factual) * log_prob[idx, meta_info]
        estimated_policy_grad_arr += torch.sum(f_hat_pi_phi_2nd * current_pi * log_prob, dim=1)

        estimated_policy_grad_arr += self.imit_reg * log_prob[idx, meta_info]

        return estimated_policy_grad_arr

    def predict(self, dataset_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(dataset_test["context"]).float().to(self.device)
        mask = torch.from_numpy(dataset_test["mask_m_lack"]).long().to(self.device)

        logits = self.nn_model(x)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))
        pi = self.softmax(masked_logits)
        return pi.detach().cpu().numpy()
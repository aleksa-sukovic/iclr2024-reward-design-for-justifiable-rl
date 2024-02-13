import lime
import torch
import tqdm
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F

from os.path import exists
from typing import List, Tuple, Optional, Literal
from typing import Optional
from tensordict import TensorDict
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from tianshou.data import PrioritizedReplayBuffer, Batch
from shap import DeepExplainer

from argo.models.judge import Judge
from argo.models.clinician import ClinicianPolicy
from argo.models.protagonist import DQNPolicy
from argo.models.argumentator import MaskedPPO
from argo.library.evaluation import run_debate


SEPSIS_FEATURES_IGNORED = ["m:presumed_onset", "m:charttime", "m:icustayid"]
SEPSIS_FEATURES_DEMOGRAPHIC = ["o:gender", "o:mechvent", "o:age", "o:Weight_kg"]
SEPSIS_FEATURES_OBSERVATIONAL = ["o:SOFA", "o:SIRS", "o:Shock_Index", "o:output_4hourly", "o:output_total",
                        "o:cumulated_balance", "o:GCS", "o:HR", "o:SysBP", "o:MeanBP", "o:DiaBP",
                        "o:RR", "o:Temp_C", "o:FiO2_1", "o:Potassium", "o:Sodium", "o:Chloride",
                        "o:Glucose", "o:Magnesium", "o:Calcium", "o:Hb", "o:WBC_count", "o:Platelets_count",
                        "o:PTT", "o:PT", "o:Arterial_pH", "o:paO2", "o:paCO2", "o:Arterial_BE",
                        "o:HCO3", "o:Arterial_lactate","o:PaO2_FiO2", "o:SpO2", "o:BUN", "o:Creatinine",
                        "o:SGOT", "o:SGPT", "o:Total_bili", "o:INR", "o:input_total"]
SEPSIS_FEATURES_ACTION = "a:action"
SEPSIS_FEATURES_REWARD = "r:reward"


def get_clinician_dataset(
    data_dict: TensorDict, batch_size: int, use_dem: bool = True, weighted_sampling: bool = False, scaler: Optional[StandardScaler] = None,
    mask: bool = True, rewards: bool = False, device: torch.device = torch.device("cpu"), trajectory_ids: bool = False, filter_ids: List[int] = [],
    perform_scaling: bool = False, debug: bool = False, limit: Optional[int] = None, num_workers: int = 1, shuffle: bool = True, reward_multiplier: float = 1.0,
) -> Tuple[TensorDataset, DataLoader, StandardScaler]:
    """Returns the Torch dataset and loader representing patents in the form (s_t, a_r) or (s_t, a_t, r_t)"""
    if filter_ids:
        indices = torch.tensor(filter_ids)
        indices = pd.DataFrame(data_dict["trajectory_id"].cpu().int(), columns=["id"])
        indices = indices[indices["id"].isin(filter_ids)].index.tolist()
        indices = torch.tensor(indices)
        data_dict = data_dict[indices]

    s_obs, s_dem, a_t, r_t, ids = data_dict["obs"], data_dict["dem"], data_dict["actions"].long(), data_dict["rewards"] * reward_multiplier, data_dict["trajectory_id"].int()

    if mask:
        mask = (s_obs == 0).all(dim=-1)
        s_obs, s_dem = s_obs[~mask], s_dem[~mask]
        a_t = data_dict["actions"][~mask]
        r_t = r_t[~mask]
        ids = ids.repeat_interleave(data_dict["obs"].size(1), dim=1)[~mask]
        if "r_hat" in locals(): r_hat = r_hat[~mask]

    s_t = s_obs if not use_dem else torch.cat((s_dem, s_obs), dim=-1)
    s_t = s_t if limit is None else s_t[:limit]
    a_t = a_t.argmax(dim=-1)
    a_t = a_t if limit is None else a_t[:limit]
    r_t = r_t if limit is None else r_t[:limit]
    ids = ids if limit is None else ids[:limit]

    if perform_scaling and scaler is not None:
        # Standardizes the dataset with existing scaler, usually derived from train set
        s_t = scaler.transform(s_t.cpu().numpy())
        s_t = torch.tensor(s_t, device=device)
    elif perform_scaling:
        # Standardizes the dataset by fitting a new scaler
        scaler = StandardScaler()
        s_t = scaler.fit_transform(s_t.cpu().numpy())
        s_t = torch.tensor(s_t, device=device)

    dataset = TensorDataset(s_t.float(), a_t.long(), r_t.float()) if rewards else TensorDataset(s_t.float(), a_t.long())
    dataset = TensorDataset(s_t.float(), a_t.long(), r_t.float(), r_hat.float()) if dense_reward else dataset
    dataset = TensorDataset(s_t.float(), a_t.long(), r_t.float(), ids) if trajectory_ids else dataset

    if not weighted_sampling:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        num_samples, num_actions = s_t.size(0), len(torch.unique(a_t))
        _, counts = torch.unique(a_t.cpu(), return_counts=True)
        actions_weight = torch.tensor([num_samples] * num_actions) / counts
        samples_weight = [actions_weight[a] for a in a_t.cpu().tolist()]
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    if debug:
        dataset = TensorDataset(next(iter(dataloader)))
        dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataset, dataloader, scaler


def get_debate_dataset(
    data_dict: Optional[TensorDict] = None, scaler: Optional[StandardScaler] = None, perform_scaling: bool = False, use_dem: bool = True,
    limit: Optional[int] = None, batch_size: int = 1, num_workers: int = 1, load_path: Optional[str] = None,  device: str = "cpu"
):
    return get_judge_dataset(data_dict=data_dict, use_dem=use_dem, batch_size=batch_size,
                             weighted_sampling=False, scaler=scaler, perform_scaling=perform_scaling,
                             limit=limit, num_workers=num_workers, device=device, load_path=load_path)


def get_xai_debate_dataset(
    method: Literal["lime", "shap"], clinician: ClinicianPolicy, background_dataset_path: Optional[str] = None, background_samples: int = 100, load_path: Optional[str] = None,
    num_arguments: int = 3, data_dict: Optional[TensorDict] = None, scaler: Optional[StandardScaler] = None, perform_scaling: bool = False, use_dem: bool = True,
    limit: Optional[int] = None, batch_size: int = 1, num_workers: int = 1, preferences_path: Optional[str] = None,  device: str = "cpu", seed: int = None
):
    if load_path and exists(load_path):
        dataset = torch.load(load_path, map_location=device)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return dataset, dataloader, None

    def lime_predict_fn(data: np.ndarray):
        data = torch.tensor(data, dtype=torch.float32, device=device)
        logits = clinician(data)
        return F.softmax(logits, dim=-1).detach().cpu().numpy()

    judge_dataset, judge_dataloader, scaler = get_judge_dataset(data_dict=data_dict, use_dem=use_dem, batch_size=1,
                             weighted_sampling=False, scaler=scaler, perform_scaling=perform_scaling,
                             limit=limit, num_workers=num_workers, device=device, load_path=preferences_path)

    if background_dataset_path and exists(background_dataset_path):
        background_dataset = torch.load(background_dataset_path).to(device)
    else:
       background_dataset = judge_dataset[:][0][torch.randint(0, len(judge_dataset), (background_samples, ))].to(device)
       torch.save(background_dataset, background_dataset_path)

    state_dim = background_dataset.shape[-1]
    arguments = SEPSIS_FEATURES_DEMOGRAPHIC + SEPSIS_FEATURES_OBSERVATIONAL
    total_items = min(len(judge_dataloader), limit if limit else len(judge_dataloader))

    s_full = torch.empty((0, state_dim), dtype=torch.float).to(device)
    a1_full = torch.empty((0,), dtype=torch.long).to(device)
    a2_full = torch.empty((0,), dtype=torch.long).to(device)
    xai_args_full = torch.empty((0, state_dim), dtype=torch.float).to(device)
    pref_full = torch.empty((0, 2), dtype=torch.float).to(device)

    for state, a1, a2, pref in tqdm.tqdm(judge_dataloader, desc="Creating XAI dataset", total=total_items):
        if limit and len(s_full) >= limit:
            break

        state, a1, a2, pref = state.to(device), a1.to(device), a2.to(device), pref.to(device)

        a_t = a1 if pref.flatten()[0].item() == 1.0 else a2
        a_r = a1 if pref.flatten()[0].item() == 0.0 else a2

        if method == "shap":
            shap_explainer = DeepExplainer(clinician.model, background_dataset)
            shap_values = shap_explainer.shap_values(state)
            shap_indices = torch.tensor(shap_values[a_t.item()])
            shap_indices = torch.topk(shap_indices, k=num_arguments)[1].flatten()
            args_mask = torch.zeros_like(state).to(device)
            args_mask[:, shap_indices] = 1
        elif method == "lime":
            explainer = lime.lime_tabular.LimeTabularExplainer(training_data=background_dataset.cpu().numpy(), mode="classification",
                                                               feature_names=arguments, discretize_continuous=False, random_state=seed)
            explanation = explainer.explain_instance(data_row=state.cpu().flatten().numpy(), predict_fn=lime_predict_fn,
                                                     labels=[a_t.item(), a_r.item()], num_features=num_arguments)
            indices = [arguments.index(name) for name, _ in explanation.as_list(label=a_t.item())]
            args_mask = torch.zeros_like(state).to(device)
            args_mask[:, indices] = 1
        else:
            raise ValueError(f"Unknown XAI method {method}")

        s_full = torch.cat((s_full, state))
        a1_full = torch.cat((a1_full, a1))
        a2_full = torch.cat((a2_full, a2))
        pref_full = torch.cat((pref_full, pref))
        xai_args_full = torch.cat((xai_args_full, args_mask))

    dataset = TensorDataset(s_full, a1_full, a2_full, pref_full, xai_args_full)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataset, dataloader, scaler


def get_judge_dataset(
    data_dict: TensorDict, batch_size: int, use_dem: bool = True, weighted_sampling: bool = False,
    scaler: Optional[StandardScaler] = None, perform_scaling: bool = False, num_workers: int = 1,
    limit: Optional[int] = None, device: torch.device = torch.device("cpu"), debug: bool = False,
    load_path: Optional[str] = None, method: Literal["random", "exhaustive", "offset"] = "random",
):
    """Returns the Torch dataset and loader representing the preference dataset in the form (s_t, a_1, a_2, pref),
       where s_t is the patient state, a_1 and a_2 are two actions, one of which corresponds to the action a_t from
       the dataset, indicated by the pref \\in [0, 1] variable.
    """
    if load_path and exists(load_path):
        if not weighted_sampling:
            # loads and returns dataset from provided path
            dataset = torch.load(load_path)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            return dataset, dataloader, scaler

        # weighted sampling is enabled, so we need to calculate the sample weights
        dataset = torch.load(load_path)
        a_true = torch.empty((0, ), dtype=torch.long)

        # the weights are calculated based on actions that were preferred
        for _, a1, a2, pref in DataLoader(dataset, batch_size=128, num_workers=num_workers):
            a_true_batch = torch.zeros_like(a1, device=device)
            a_true_batch[pref[:, 0] == 1.0] = a1[pref[:, 0] == 1.0]
            a_true_batch[pref[:, 1] == 1.0] = a2[pref[:, 1] == 1.0]
            a_true = torch.cat((a_true, a_true_batch.long()))

        # calculate sample weights
        _, counts = torch.unique(a_true.cpu(), return_counts=True)
        num_actions, num_samples = torch.unique(a_true).size(0), a_true.size(0)
        actions_weight = torch.tensor([a_true.size(0)] * num_actions) / counts
        samples_weight = [actions_weight[a] for a in a_true.cpu().tolist()]
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=num_samples) if samples_weight else None

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        return dataset, dataloader, scaler

    _, in_dataloader, _ = get_clinician_dataset(data_dict=data_dict, use_dem=use_dem, batch_size=128, limit=limit, scaler=scaler, weighted_sampling=False,
                                                perform_scaling=perform_scaling, device=device, num_workers=num_workers, debug=debug)
    state_dim = data_dict["obs"].size(-1) if not use_dem else data_dict["obs"].size(-1) + data_dict["dem"].size(-1)
    num_actions = data_dict["actions"].size(-1)

    s = torch.empty((0, state_dim), dtype=torch.float)
    a_true = torch.empty((0,), dtype=torch.long)
    a_other = torch.empty((0,), dtype=torch.long)
    pref = torch.empty((0, 2), dtype=torch.float)

    # creates preference comparisons
    for s_t, a_t in tqdm.tqdm(in_dataloader, desc="Creating preference dataset"):
        if method == "random":
            a_r = torch.randint(low=0, high=num_actions, size=a_t.shape)
            while torch.any(a_r == a_t): a_r = torch.randint(low=0, high=num_actions, size=a_t.shape)
        elif method == "offset":
            offsets = {0: [0, 1], 1: [-1, 0, 1], 2: [-1, 0, 1], 3: [-1, 0, 1], 4: [-1, 0]}
            a_r = torch.empty_like(a_t, device=device)
            for i, act in enumerate(a_t):
                iv, vc = act.item() // 5, act.item() % 5
                offset_iv, offset_vc = random.choice(offsets[iv]), random.choice(offsets[vc])
                while offset_iv == 0 and offset_vc == 0: offset_iv, offset_vc = random.choice(offsets[iv]), random.choice(offsets[vc])
                a_r[i] = (iv + offset_iv) * 5 + (vc + offset_vc)
        elif method == "exhaustive":
            a_r = torch.arange(num_actions).repeat(a_t.shape[0])
            s_t, a_t = s_t.repeat_interleave(num_actions, dim=0), a_t.repeat_interleave(num_actions)
            s_t, a_r, a_t = s_t[a_t != a_r], a_r[a_t != a_r], a_t[a_t != a_r]

        s = torch.cat((s, s_t))
        a_true = torch.cat((a_true, a_t))
        a_other = torch.cat((a_other, a_r))
        pref = torch.cat((pref, torch.tensor([1.0, 0.0]).repeat((s_t.shape[0], 1))))

    # calculates the sample weights, if needed
    if not weighted_sampling:
        samples_weight = None
    else:
        _, counts = torch.unique(a_true.cpu(), return_counts=True)
        actions_weight = torch.tensor([s.size(0)] * num_actions) / counts
        samples_weight = [actions_weight[a] for a in a_true.cpu().tolist()]

    # shuffles the data, so that the true action is not always the first one
    data_shuffled = torch.empty((0, 4))
    data_iter = DataLoader(TensorDataset(a_true, a_other, pref), batch_size=128)

    for a_true, a_other, pref in tqdm.tqdm(data_iter, desc="Shuffling preference dataset"):
        data = torch.cat((a_true.reshape((-1, 1)), a_other.reshape((-1, 1)), pref), dim=1)
        perm_mat = torch.eye(2)[torch.randperm(2)]
        perm = torch.cat((perm_mat, torch.zeros((2, 2))), dim=1)
        perm = torch.cat((perm, torch.cat((torch.zeros((2, 2)), perm_mat), dim=1)), dim=0)
        data = data @ perm
        data_shuffled = torch.cat((data_shuffled, data))

    # constructs the dataset and dataloader
    a1, a2, pref = data_shuffled[:, 0].flatten().long(), data_shuffled[:, 1].flatten().long(), data_shuffled[:, 2:].float()
    dataset = TensorDataset(s, a1, a2, pref)
    if load_path:
        torch.save(dataset, load_path)
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=s.size(0)) if samples_weight else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    return dataset, dataloader, scaler


def get_patient_dataset(data_dict: TensorDict, batch_size: int, use_dem: bool = True, mask: bool = True, shuffle: bool = True, num_workers: int = 1, reward_multiplier: float = 1.0,
                             baseline: Optional[DQNPolicy] = None, judge: Optional[Judge] = None, argumentator: Optional[MaskedPPO] = None, lmbd_justifiability: float = 0.0,
                             num_arguments: int = 6, debate_multiplier: float = 1.0, limit: Optional[int] = None, filter_ids: List[float] = [], device: str = "cpu", dense_reward: bool = False):
    """Returns the Torch dataset and loader representing patients in the form (s_t, a_t, r_t, done, s_t+1)"""
    if filter_ids:
        indices = pd.DataFrame(data_dict["trajectory_id"].cpu().int(), columns=["id"])
        indices = indices[indices["id"].isin(filter_ids)].index.tolist()
        indices = torch.tensor(indices)
        data_dict = data_dict[indices]

    a_t = data_dict["actions"].to(device)
    r_t = data_dict["rewards"].to(device) * reward_multiplier
    s_t_obs, s_t_dem = data_dict["obs"].to(device), data_dict["dem"].to(device)
    s_t = torch.cat((s_t_dem, s_t_obs), dim=-1) if use_dem else s_t_obs
    s_t_dummy = torch.zeros((s_t.shape[0], 1, s_t.shape[-1])).to(device)

    max_length = data_dict["trajectory_length"].max().int().item()
    s_t, a_t, r_t = s_t[:, :max_length, :], a_t[:, :max_length, :], r_t[:, :max_length]

    s_t_curr, s_t_next = s_t, torch.cat((s_t[:, 1:, :], s_t_dummy), dim=1)
    a_t_curr, r_t_curr = a_t, r_t
    done = (r_t_curr != 0).int()

    if dense_reward:
        c0, c1, c2 = -0.025, -0.125, -2.0
        sofa_ind, lactate_ind = SEPSIS_FEATURES_OBSERVATIONAL.index("o:SOFA") + s_t_dem.shape[-1], SEPSIS_FEATURES_OBSERVATIONAL.index("o:Arterial_lactate") + s_t_dem.shape[-1]
        rew = torch.zeros_like(r_t, device=device)

        sofa_curr, sofa_next = s_t_curr[:, :, sofa_ind], s_t_next[:, :, sofa_ind]
        lactate_curr, lactate_next = s_t_curr[:, :, lactate_ind], s_t_next[:, :, lactate_ind]

        rew = torch.where((sofa_curr == sofa_next) & (sofa_next > 0), c0, rew)
        rew += c1 * (sofa_next - sofa_curr)
        rew += c2 * torch.tanh(lactate_next - lactate_curr)

        r_t_curr = torch.where(r_t_curr != 0, r_t_curr, rew)

    if baseline and argumentator and judge and lmbd_justifiability:
        a_p1 = a_t_curr.argmax(dim=-1)
        a_p2 = torch.tensor(baseline(Batch(obs=s_t_curr.flatten(end_dim=1), info={})).act).reshape(a_p1.shape).to(device)

        s_t_curr_flat, a_p1_flat, a_p2_flat = s_t_curr.flatten(end_dim=1), a_p1.flatten(), a_p2.flatten()
        rew_p1, rew_p2 = run_debate(s_t=s_t_curr_flat, a_p1=a_p1_flat, a_p2=a_p2_flat, judge=judge, argumentator=argumentator, num_arguments=num_arguments, device=device)
        r_t_debate = debate_multiplier * torch.where(rew_p1 > rew_p2, +1.0, 0.0).flatten()
        r_t_debate = r_t_debate.reshape(r_t_curr.shape)

        r_t_env = r_t_curr
        r_t_curr = (1 - lmbd_justifiability) * r_t_env + lmbd_justifiability * r_t_debate

    if mask:
        s_t_mask = (s_t_curr == 0).all(dim=-1)
        s_t_curr, s_t_next = s_t_curr[~s_t_mask], s_t_next[~s_t_mask]
        a_t_curr, r_t_curr = a_t_curr[~s_t_mask], r_t_curr[~s_t_mask]
        done = done[~s_t_mask]

    if limit:
        s_t_curr, s_t_next = s_t_curr[:limit], s_t_next[:limit]
        a_t_curr, r_t_curr = a_t_curr[:limit], r_t_curr[:limit]
        done = done[:limit]

    dataset = TensorDataset(s_t_curr, a_t_curr, r_t_curr, done, s_t_next)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataset, dataloader, None


def get_patient_buffer(data_dict: TensorDict,  use_dem: bool = True, device: str = "cuda", reward_multiplier: float = 1.0, dense_reward: bool = False, save_path: Optional[str] = None, limit: Optional[int] = None,
                            baseline: Optional[DQNPolicy] = None, judge: Optional[Judge] = None, argumentator: Optional[MaskedPPO] = None, lmbd_justifiability: float = 0.0, num_arguments: int = 6,
                            debate_multiplier: float = 1.0) -> PrioritizedReplayBuffer:
    """Instantiates and creates a replay buffer for training the task policy"""
    if save_path and exists(save_path):
        return PrioritizedReplayBuffer.load_hdf5(save_path, device=device)

    dataset, dataloader, _ = get_patient_dataset(data_dict=data_dict, use_dem=use_dem, reward_multiplier=reward_multiplier, limit=limit, device=device, dense_reward=dense_reward,
                                                      baseline=baseline, judge=judge, argumentator=argumentator, lmbd_justifiability=lmbd_justifiability, num_arguments=num_arguments,
                                                      debate_multiplier=debate_multiplier, mask=True, batch_size=1, shuffle=False, num_workers=0)
    buffer = PrioritizedReplayBuffer(size=dataset[:][0].shape[0], alpha=0.6, beta=0.9)

    for s_t, a_t, r_t, d_t, s_t_next in tqdm.tqdm(dataloader, "Filling replay buffer"):
        batch = Batch(
            obs=s_t.flatten().cpu().numpy(),
            act=a_t.argmax(dim=-1).flatten().cpu().item(),
            rew=r_t.flatten().cpu().item(),
            terminated=d_t.flatten().cpu().bool().item(),
            truncated=False,
            obs_next=s_t_next.flatten().cpu().numpy(),
            info={})
        buffer.add(batch)

    if save_path:
        buffer.save_hdf5(save_path)

    return buffer

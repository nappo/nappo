import torch


@torch.no_grad()
def get_updated_action_log_probs(storage, new_ac, algo):
    """
    Arguments:
        storage ():
        new_policy (ActorCritic):
        algo ():
    """

    len, num_proc = storage.data["act"].shape[0:2]

    # Create batches without shuffling data
    batches = storage.generate_batches(
        algo.num_mini_batch, algo.mini_batch_size,
        num_epochs=1, recurrent_ac=new_ac.is_recurrent, shuffle=False)

    # Obtain new value and log probability predictions
    new_val = []
    new_logp = []
    for batch in batches:
        obs, rhs, act, done = batch["obs"], batch["rhs"], batch["act"], batch["done"]
        (val, logp, _, _) = new_ac.evaluate_actions(obs, rhs, done, act)
        new_val.append(val)
        new_logp.append(logp)

    # Concatenate results
    if new_ac.is_recurrent:
        new_val = [p.view(len, num_proc // algo.num_mini_batch, -1) for p in new_val]
        storage.data["val"][:-1] = torch.cat(new_val, dim=1)
        new_logp = [p.view(len, num_proc // algo.num_mini_batch, -1) for p in new_logp]
        new_logp = torch.cat(new_logp, dim=1)
    else:
        storage.data["val"][:-1] = torch.cat(new_val, dim=0).view(len, num_proc, 1)
        new_logp = torch.cat(new_logp, dim=0).view(len, num_proc, 1)

    return new_logp

@torch.no_grad()
def compute_vtrace(storage, new_policy, algo, clip_rho_thres=1.0, clip_c_thres=1.0):
    """
    Arguments:
        storage ():
        new_policy (Actorritic):
        algo ():
        clip_rho_thres (float):
        clip_c_thres (float):
    """

    l = storage.step if storage.step != 0 else storage.max_size

    with torch.no_grad():
        _ = new_policy.get_action(
            storage.data["obs"][storage.step - 1],
            storage.data["rhs"][storage.step - 1],
            storage.data["done"][storage.step - 1])
        next_value = new_policy.get_value(storage.data["obs"][storage.step - 1])
    storage.data["val"][storage.step] = next_value

    new_action_log_probs = get_updated_action_log_probs(storage, new_policy, algo)

    log_rhos = (new_action_log_probs - storage.data["logp"])
    clipped_rhos = torch.clamp(torch.exp(log_rhos), max=clip_rho_thres)
    clipped_cs = torch.clamp(torch.exp(log_rhos), max=clip_c_thres)

    deltas = clipped_rhos * (storage.data["rew"] + algo.gamma * storage.data["val"][1:]
                             - storage.data["val"][:-1])

    acc = torch.zeros_like(storage.data["val"][-1])
    result = []
    for i in reversed(range(l)):
        acc = deltas[i] + algo.gamma * clipped_cs[i] * acc * (1 - storage.data["done"][i + 1])
        result.append(acc)

    result.reverse()
    result.append(torch.zeros_like(storage.data["val"][-1]))
    vs_minus_v_xs = torch.stack(result)

    vs = torch.add(vs_minus_v_xs, storage.data["val"])
    adv = clipped_rhos * (storage.data["rew"] + algo.gamma * vs[1:] - storage.data["val"][:-1])

    storage.data["ret"] = vs
    storage.data["logp"] = new_action_log_probs
    storage.data["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    return storage



import torch
import torch.nn.functional as F


def _forward_process(batch, mask_token_id, u0, mask_rand, sampling_eps=1e-3):
    b, l = batch.shape
    # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
    indices = torch.arange(b, device=batch.device).float()
    t = (u0 + indices / b) % 1

    p_mask = (1 - sampling_eps) * t + sampling_eps

    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = mask_rand < p_mask
    # always unmask bos and eos
    mask_indices[:, 0] = False
    mask_indices[:, -1] = False

    noisy_batch = torch.where(mask_indices, mask_token_id, batch)
    return noisy_batch, p_mask


@torch.no_grad()
def get_logits(
    model,
    batch,
    prompt_index,
    mask_token_id,
    cfg_scale=1,
    classifier_free_guidance=1,
):
    '''
    prompt_index : 1D bool tensor, length=batch.shape[1]
    '''
    if classifier_free_guidance > 1.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_token_id
        batch = torch.cat([batch, un_batch])

    input = batch

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input).logits
        # since bos always unmask, the first logits will not be used
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    if classifier_free_guidance > 1.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + cfg_scale * (logits - un_logits)
        
    return logits[:, :batch.shape[1]]


@torch.no_grad()
def get_forward_log_likelihood(
    model,
    prefix,
    target,
    mc_num,
    batch_size,
    u0,
    mask_rand,
    mask_token_id,
    device,
    sampling_eps=1e-3,
):
    assert len(u0) == len(mask_rand), "u0 and mask random number length must be equal"

    if prefix is None:
        seq = target[None, :]
        prefix_len = 0
    else:
        seq = torch.concatenate([prefix, target])[None, :]
        prefix_len = len(prefix)
    seq = seq.repeat((batch_size, 1)).to(device)

    prompt_index = torch.arange(seq.shape[1], device=device) < prefix_len

    loss_acc = []
    for i in range(max(mc_num // batch_size, 1)):
        perturbed_seq = seq.clone()
        perturbed_seq_, p_mask = _forward_process(
            seq,
            mask_token_id,
            u0[i],
            mask_rand[i],
            sampling_eps,
        )
        perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]

        mask_indices = perturbed_seq == mask_token_id
        logits = get_logits(model, perturbed_seq, prompt_index, mask_token_id)
        loss = F.cross_entropy(
            logits[mask_indices],
            seq[mask_indices],
            reduction='none',
        ) / p_mask[mask_indices]
        loss = loss.sum() / batch_size
        loss_acc.append(loss.item())

    return - sum(loss_acc) / len(loss_acc)

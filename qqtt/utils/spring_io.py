"""Explicit spring topology in PhysTwin checkpoints.

A checkpoint stores one learned stiffness per spring (``spring_Y``), but until
now it did not store which pair of points each spring connects. On load, the
springs are rebuilt from the config (connection radius and max neighbour
count) and ``spring_Y`` is matched to them by position. That is only correct
if the rebuilt spring list comes out in exactly the same order, and with
exactly the same members, as it did at training time.

New checkpoints additionally store the spring index pairs and rest lengths.
When a checkpoint has them, loading uses them directly, so the stiffness
always stays attached to the spring it was learned for. Checkpoints without
them (all existing releases) load exactly as before.

This module only depends on torch so it can be tested without a GPU.
"""

import torch

SPRINGS_KEY = "springs"
REST_LENGTHS_KEY = "rest_lengths"

# Values returned as the ``status`` of ``resolve_spring_topology``
LEGACY = "legacy"  # checkpoint has no saved topology, use the rebuilt springs
MATCH = "match"  # saved topology is identical to the rebuilt one
SAVED = "saved"  # saved topology differs from the rebuilt one, use the saved one


def spring_topology_state(springs, rest_lengths):
    """Entries to add to a checkpoint dict so the spring topology is saved.

    Tensors are detached and moved to CPU so the checkpoint does not depend on
    the device it was trained on.
    """
    return {
        SPRINGS_KEY: springs.detach().to(device="cpu", dtype=torch.int32).clone(),
        REST_LENGTHS_KEY: rest_lengths.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone(),
    }


def resolve_spring_topology(
    checkpoint, rebuilt_springs, rebuilt_rest_lengths, num_vertices
):
    """Decide which spring topology to use when loading ``checkpoint``.

    Args:
        checkpoint: the dict returned by ``torch.load``.
        rebuilt_springs: (N, 2) int tensor built from the config at start up.
        rebuilt_rest_lengths: (N,) float tensor built alongside it.
        num_vertices: number of vertices in the simulator (object points plus
            controller points), used to check the saved indices are in range.

    Returns:
        ``(springs, rest_lengths, status)``. For ``LEGACY`` and ``MATCH`` the
        rebuilt tensors are returned unchanged. For ``SAVED`` the saved tensors
        are returned on the same device and dtype as the rebuilt ones.
    """
    has_springs = SPRINGS_KEY in checkpoint
    has_rest = REST_LENGTHS_KEY in checkpoint
    if not has_springs and not has_rest:
        return rebuilt_springs, rebuilt_rest_lengths, LEGACY
    if has_springs != has_rest:
        raise ValueError(
            f"Checkpoint has only one of '{SPRINGS_KEY}' and '{REST_LENGTHS_KEY}'; "
            "both are needed to restore the spring topology"
        )

    springs = torch.as_tensor(checkpoint[SPRINGS_KEY])
    rest_lengths = torch.as_tensor(checkpoint[REST_LENGTHS_KEY])

    if springs.dim() != 2 or springs.shape[1] != 2:
        raise ValueError(
            f"Saved springs must have shape (N, 2), got {tuple(springs.shape)}"
        )
    n = springs.shape[0]
    if rest_lengths.shape != (n,):
        raise ValueError(
            f"Saved rest_lengths must have shape ({n},), got {tuple(rest_lengths.shape)}"
        )
    if "spring_Y" in checkpoint and len(checkpoint["spring_Y"]) != n:
        raise ValueError(
            f"Checkpoint has {len(checkpoint['spring_Y'])} spring_Y values "
            f"but {n} saved springs"
        )
    if n > 0 and (int(springs.min()) < 0 or int(springs.max()) >= num_vertices):
        raise ValueError(
            f"Saved springs index vertices in [{int(springs.min())}, "
            f"{int(springs.max())}] but the simulator has {num_vertices} vertices; "
            "the checkpoint does not belong to this data"
        )

    springs = springs.to(device=rebuilt_springs.device, dtype=rebuilt_springs.dtype)
    rest_lengths = rest_lengths.to(
        device=rebuilt_rest_lengths.device, dtype=rebuilt_rest_lengths.dtype
    )

    if torch.equal(springs, rebuilt_springs) and torch.equal(
        rest_lengths, rebuilt_rest_lengths
    ):
        return rebuilt_springs, rebuilt_rest_lengths, MATCH
    return springs, rest_lengths, SAVED

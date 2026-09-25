"""CPU tests for saving and restoring the spring topology in checkpoints."""

import pytest
import torch


def make_topology(num_vertices=12, num_springs=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    springs = torch.randint(0, num_vertices, (num_springs, 2), generator=g)
    springs = springs.to(torch.int32)
    rest_lengths = torch.rand(num_springs, generator=g) + 0.01
    spring_Y = torch.rand(num_springs, generator=g) * 1e4 + 1e3
    return springs, rest_lengths, spring_Y


def legacy_checkpoint(spring_Y, num_object_springs):
    # The keys a checkpoint written before this change contains
    return {
        "epoch": 0,
        "num_object_springs": num_object_springs,
        "spring_Y": spring_Y,
        "collide_elas": torch.tensor([0.5]),
        "collide_fric": torch.tensor([0.3]),
        "collide_object_elas": torch.tensor([0.7]),
        "collide_object_fric": torch.tensor([0.3]),
        "optimizer_state_dict": {},
    }


def new_checkpoint(spring_io, springs, rest_lengths, spring_Y):
    ckpt = legacy_checkpoint(spring_Y, len(springs))
    ckpt.update(spring_io.spring_topology_state(springs, rest_lengths))
    return ckpt


def save_and_load(ckpt, path):
    torch.save(ckpt, path)
    return torch.load(path, map_location="cpu")


def test_round_trip_through_torch_save(spring_io, tmp_path):
    springs, rest_lengths, spring_Y = make_topology()
    loaded = save_and_load(
        new_checkpoint(spring_io, springs, rest_lengths, spring_Y),
        tmp_path / "ckpt.pth",
    )

    assert loaded["springs"].dtype == torch.int32
    assert loaded["rest_lengths"].dtype == torch.float32
    assert torch.equal(loaded["springs"], springs)
    assert torch.equal(loaded["rest_lengths"], rest_lengths)

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        loaded, springs, rest_lengths, num_vertices=12
    )
    assert status == spring_io.MATCH
    assert out_springs is springs and out_rest is rest_lengths


def test_round_trip_with_weights_only_load(spring_io, tmp_path):
    # torch >= 2.6 defaults torch.load to weights_only=True; the new keys are
    # plain tensors so they must load under that setting too
    springs, rest_lengths, spring_Y = make_topology()
    path = tmp_path / "ckpt.pth"
    torch.save(new_checkpoint(spring_io, springs, rest_lengths, spring_Y), path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["springs"], springs)


def test_legacy_checkpoint_keeps_rebuilt_springs(spring_io, tmp_path):
    springs, rest_lengths, spring_Y = make_topology()
    loaded = save_and_load(
        legacy_checkpoint(spring_Y, len(springs)), tmp_path / "legacy.pth"
    )

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        loaded, springs, rest_lengths, num_vertices=12
    )
    assert status == spring_io.LEGACY
    assert out_springs is springs and out_rest is rest_lengths


def test_reordered_rebuild_uses_saved_springs(spring_io):
    # The case this change is for: the springs rebuilt at load time come out
    # in a different order from the ones the stiffness was learned on
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    perm = torch.randperm(len(springs), generator=torch.Generator().manual_seed(1))
    rebuilt_springs, rebuilt_rest = springs[perm], rest_lengths[perm]

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, rebuilt_springs, rebuilt_rest, num_vertices=12
    )
    assert status == spring_io.SAVED
    assert torch.equal(out_springs, springs)
    assert torch.equal(out_rest, rest_lengths)

    # Stiffness per point pair is what was trained with the saved springs...
    def stiffness_by_pair(s, y):
        return {tuple(p.tolist()): float(v) for p, v in zip(s, y)}

    trained = stiffness_by_pair(springs, spring_Y)
    assert stiffness_by_pair(out_springs, ckpt["spring_Y"]) == trained
    # ...and would not be with the reordered rebuild, which is the silent error
    assert stiffness_by_pair(rebuilt_springs, ckpt["spring_Y"]) != trained


def test_rebuild_with_different_spring_count_uses_saved_springs(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    out_springs, _, status = spring_io.resolve_spring_topology(
        ckpt, springs[:-3], rest_lengths[:-3], num_vertices=12
    )
    assert status == spring_io.SAVED
    assert len(out_springs) == len(spring_Y)


def test_same_springs_different_rest_lengths_uses_saved(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)

    _, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, springs, rest_lengths + 1e-3, num_vertices=12
    )
    assert status == spring_io.SAVED
    assert torch.equal(out_rest, rest_lengths)


def test_saved_tensors_are_detached_cpu_copies(spring_io):
    springs, rest_lengths, _ = make_topology()
    springs64 = springs.to(torch.int64)
    rest = rest_lengths.clone().requires_grad_(True)

    state = spring_io.spring_topology_state(springs64, rest)
    assert state["springs"].dtype == torch.int32
    assert state["springs"].device.type == "cpu"
    assert not state["rest_lengths"].requires_grad

    # Later in place changes to the simulator's tensors must not leak into a
    # checkpoint that has already been built
    springs64[0, 0] = 11
    assert state["springs"][0, 0] == springs[0, 0]


def test_output_matches_rebuilt_device_and_dtype(spring_io):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    rebuilt = springs.flip(0).to(torch.int64)

    out_springs, out_rest, status = spring_io.resolve_spring_topology(
        ckpt, rebuilt, rest_lengths.flip(0).double(), num_vertices=12
    )
    assert status == spring_io.SAVED
    assert out_springs.dtype == torch.int64
    assert out_rest.dtype == torch.float64


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("rest_lengths"), "both are needed"),
        (lambda c: c.pop("springs"), "both are needed"),
        (lambda c: c.__setitem__("springs", c["springs"][:, :1]), "shape"),
        (lambda c: c.__setitem__("rest_lengths", c["rest_lengths"][:-1]), "shape"),
        (lambda c: c.__setitem__("spring_Y", c["spring_Y"][:-1]), "spring_Y"),
    ],
)
def test_malformed_checkpoint_is_rejected(spring_io, mutate, message):
    springs, rest_lengths, spring_Y = make_topology()
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    mutate(ckpt)
    with pytest.raises(ValueError, match=message):
        spring_io.resolve_spring_topology(ckpt, springs, rest_lengths, 12)


def test_springs_from_other_data_are_rejected(spring_io):
    # A checkpoint whose springs index more vertices than this data has
    springs, rest_lengths, spring_Y = make_topology(num_vertices=12)
    ckpt = new_checkpoint(spring_io, springs, rest_lengths, spring_Y)
    with pytest.raises(ValueError, match="does not belong"):
        spring_io.resolve_spring_topology(
            ckpt, springs, rest_lengths, num_vertices=int(springs.max())
        )

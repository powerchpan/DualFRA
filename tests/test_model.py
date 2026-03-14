import numpy as np
import torch

from preprocessing import (
    SkeletonMultiModalViews,
    SlidingWindowSampler,
    SkeletonAugmentor,
    FallRiskDataset,
    AugmentedDataset,
    build_windowed_dataset,
    build_dataset_splits,
    make_synthetic_dataset,
)
from model import (
    ATTGCN,
    MSTCN,
    SpatioTemporalLayer,
    GraphEncoder,
    MLPViewEncoder,
    SpatioTemporalEncoder,
    CrossAttentionFusion,
    MLPClassifier,
    FallRiskAssessmentModel,
    _default_channel_schedule,
)

# ── Test configuration (small to keep tests fast) ─────────────────────────────
B            = 2
T            = 40       # small window for unit tests (paper uses 400)
V            = 17       # small joint count (paper uses 49)
C            = 3
CLASSES      = 3
SMALL_SCHED  = [16, 16, 16, 32, 32]   # fast equivalent of paper's [48,48,48,96,96]
PAPER_SCHED  = _default_channel_schedule()   # [48, 48, 48, 96, 96]


def sep(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")



def test_multimodal_views():
    sep("SkeletonMultiModalViews")
    m   = SkeletonMultiModalViews()
    x   = torch.randn(B, T, V, C)
    nf, sj, sm = m(x)
    assert nf.shape == (B, T, V, C)
    assert sj.shape == (B, T, V * C)
    assert sm.shape == (B, T, V * C * 2)
    print(f"  node_feats : {tuple(nf.shape)}")
    print(f"  s_joint    : {tuple(sj.shape)}")
    print(f"  s_motion   : {tuple(sm.shape)}")
    print("  PASSED ✓")


def test_sliding_window_sampler():
    sep("SlidingWindowSampler")
    W    = 10   # small window for testing
    rng  = np.random.default_rng(0)

    # Case 1: raw sequence longer than window
    T_raw = 25
    seq   = rng.standard_normal((T_raw, V, C)).astype(np.float32)
    sampler = SlidingWindowSampler(window_size=W, stride=1)
    windows = sampler(seq, label=1)
    expected = T_raw - W + 1   # 16 windows
    assert len(windows) == expected, f"Expected {expected} windows, got {len(windows)}"
    assert windows[0]["seq"].shape == (W, V, C)
    assert windows[0]["label"] == 1
    print(f"  T_raw={T_raw}, W={W}, stride=1 → {len(windows)} windows  ✓")

    # Case 2: raw sequence shorter than window → padded to W, one sample
    T_short = 5
    seq_short = rng.standard_normal((T_short, V, C)).astype(np.float32)
    windows_short = sampler(seq_short, label=0)
    assert len(windows_short) == 1
    assert windows_short[0]["seq"].shape == (W, V, C)
    print(f"  T_raw={T_short} < W={W} → 1 padded window  ✓")

    # Case 3: stride=2
    sampler2 = SlidingWindowSampler(window_size=W, stride=2)
    windows2 = sampler2(seq, label=2)
    expected2 = len(range(0, T_raw - W + 1, 2))
    assert len(windows2) == expected2
    print(f"  T_raw={T_raw}, W={W}, stride=2 → {len(windows2)} windows  ✓")
    print("  PASSED ✓")


def test_skeleton_augmentor():
    sep("SkeletonAugmentor")
    aug = SkeletonAugmentor(
        noise_std=0.05, rotate_prob=1.0,    # force rotation always
        joint_drop_prob=0.3, coord_mask_prob=0.3,
        seed=0,
    )
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((T, V, C)).astype(np.float32)
    out = aug(seq)

    assert out.shape == seq.shape, f"Shape changed: {out.shape}"
    assert not np.allclose(out, seq), "Augmented output identical to input"

   
    aug_noise_only = SkeletonAugmentor(
        noise_std=0.5, rotate_prob=0.0,
        joint_drop_prob=0.0, coord_mask_prob=0.0, seed=0,
    )
    out_noise = aug_noise_only(seq.copy())
    assert not np.allclose(out_noise, seq), "Noise augmentation had no effect"
    print("  (a) Gaussian noise       ✓")

    aug_rot = SkeletonAugmentor(
        noise_std=0.0, rotate_prob=1.0,
        joint_drop_prob=0.0, coord_mask_prob=0.0, seed=0,
    )
    out_rot = aug_rot(seq.copy())
    assert not np.allclose(out_rot, seq), "Rotation had no effect"
    print("  (b) Random rotation      ✓")

    aug_drop = SkeletonAugmentor(
        noise_std=0.0, rotate_prob=0.0,
        joint_drop_prob=0.9, coord_mask_prob=0.0, seed=0,
    )
    out_drop = aug_drop(seq.copy())
    joint_norms = np.linalg.norm(out_drop, axis=(0, 2))  # [V]
    assert (joint_norms == 0).any(), "No joint was zeroed despite high drop prob"
    print("  (c) Joint removal        ✓")

    aug_mask = SkeletonAugmentor(
        noise_std=0.0, rotate_prob=0.0,
        joint_drop_prob=0.0, coord_mask_prob=0.9, seed=0,
    )
    out_mask = aug_mask(seq.copy())
    coord_norms = np.linalg.norm(out_mask, axis=(0, 1))  # [C]
    assert (coord_norms == 0).any(), "No coordinate was masked despite high prob"
    print("  (d) Coordinate masking   ✓")

    print("  PASSED ✓")


def test_augmented_dataset():
    sep("AugmentedDataset")
    ds  = make_synthetic_dataset(n_samples=10, num_joints=V, coord_dim=C,
                                 seq_len=T, num_classes=CLASSES, seed=0)
    aug = SkeletonAugmentor(noise_std=0.1, seed=1)
    ads = AugmentedDataset(ds, aug)

    assert len(ads) == len(ds)
    tug_orig, ftsts_orig, lbl_orig = ds[0]
    tug_aug,  ftsts_aug,  lbl_aug  = ads[0]

    assert lbl_orig.item() == lbl_aug.item(), "Label changed after augmentation"
    assert not torch.allclose(tug_orig, tug_aug),   "TUG not augmented"
    assert not torch.allclose(ftsts_orig, ftsts_aug), "FTSTS not augmented"
    print(f"  Dataset length unchanged : {len(ads)}  ✓")
    print(f"  Labels preserved         ✓")
    print(f"  Sequences modified       ✓")
    print("  PASSED ✓")


def test_build_windowed_dataset():
    sep("build_windowed_dataset")
    W   = 10
    rng = np.random.default_rng(0)

    subjects = [
        {
            "tug":   rng.standard_normal((25, V, C)).astype(np.float32),
            "ftsts": rng.standard_normal((25, V, C)).astype(np.float32),
            "label": i % CLASSES,
        }
        for i in range(3)
    ]
    ds = build_windowed_dataset(subjects, window_size=W, stride=1, seq_len=W)
    expected_total = 3 * (25 - W + 1)   # 3 * 16 = 48
    assert len(ds) == expected_total, f"Expected {expected_total}, got {len(ds)}"
    tug, ftsts, lbl = ds[0]
    assert tug.shape   == (W, V, C)
    assert ftsts.shape == (W, V, C)
    print(f"  3 subjects × 16 windows = {len(ds)} samples  ✓")
    print(f"  Window shape : {tuple(tug.shape)}  ✓")
    print("  PASSED ✓")


def test_build_dataset_splits():
    sep("build_dataset_splits  (100 test / 9:1 train-val)")
    ds = make_synthetic_dataset(n_samples=200, num_joints=V, coord_dim=C,
                                seq_len=T, num_classes=CLASSES, seed=0)
    train_loader, val_loader, test_loader = build_dataset_splits(
        ds,
        n_test=100,
        train_val_ratio=9.0,
        batch_size=4,
        augment_train=True,
        seed=0,
    )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    total   = n_train + n_val + n_test

    assert n_test  == 100,                f"Expected 100 test samples, got {n_test}"
    assert total   == 200,                f"Total mismatch: {total}"
    assert n_train + n_val == 100,        f"train+val should be 100, got {n_train+n_val}"
    assert abs(n_train / n_val - 9.0) < 1.5, \
        f"Train:Val ratio off: {n_train}:{n_val}"

    tug_b, ftsts_b, lbl_b = next(iter(train_loader))
    assert tug_b.shape   == (4, T, V, C)
    assert ftsts_b.shape == (4, T, V, C)

    print(f"  n_test   = {n_test}   (held-out, unseen)  ✓")
    print(f"  n_train  = {n_train}  (augmented on-the-fly)  ✓")
    print(f"  n_val    = {n_val}  (no augmentation)  ✓")
    print(f"  Train:val ≈ {n_train}:{n_val} ≈ 9:1  ✓")
    print("  PASSED ✓")


# =============================================================================
# Model component tests
# =============================================================================

def test_att_gcn():
    sep("ATTGCN")
    m = ATTGCN(in_channels=C, out_channels=48, num_joints=V)
    z = m(torch.randn(B, T, V, C))
    assert z.shape == (B, T, V, 48)
    print(f"  {(B,T,V,C)} → {tuple(z.shape)}  ✓")
    print("  PASSED ✓")


def test_ms_tcn():
    sep("MSTCN")
    m = MSTCN(in_channels=48, num_joints=V)
    z = m(torch.randn(B, T, V, 48))
    assert z.shape[:3] == (B, T, V)
    print(f"  {(B,T,V,48)} → {tuple(z.shape)}  ✓")
    print("  PASSED ✓")


def test_st_layer():
    sep("SpatioTemporalLayer")
    layer = SpatioTemporalLayer(in_channels=C, out_channels=48, num_joints=V)
    z = layer(torch.randn(B, T, V, C))
    assert z.shape[:3] == (B, T, V)
    print(f"  {(B,T,V,C)} → {tuple(z.shape)}  ✓")
    print("  PASSED ✓")


def test_graph_encoder_paper_schedule():
    sep(f"GraphEncoder — paper schedule {PAPER_SCHED}")
    enc = GraphEncoder(in_channels=C, num_joints=V,
                       channel_schedule=PAPER_SCHED)
    E_G = enc(torch.randn(B, T, V, C))
    assert E_G.shape[0] == B
    print(f"  Schedule : {PAPER_SCHED}")
    print(f"  E_G      : {tuple(E_G.shape)}  ✓")
    print("  PASSED ✓")


def test_full_model_forward_backward():
    sep("FallRiskAssessmentModel — forward + backward")
    model = FallRiskAssessmentModel(
        num_joints=V, coord_dim=C,
        channel_schedule=SMALL_SCHED,
        mlp_hidden=64, mlp_out=32,
        attn_dim=32, cls_hidden=64,
        num_classes=CLASSES,
    )
    tug    = torch.randn(B, T, V, C)
    ftsts  = torch.randn(B, T, V, C)
    labels = torch.randint(0, CLASSES, (B,))

    logits = model(tug, ftsts)
    assert logits.shape == (B, CLASSES)

    loss = model.loss(logits, labels)
    assert loss.item() > 0
    loss.backward()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Logits     : {tuple(logits.shape)}  ✓")
    print(f"  Loss       : {loss.item():.4f}  ✓")
    print(f"  Parameters : {params:,}")
    print("  PASSED ✓")


def test_parameter_breakdown():
    sep("Parameter breakdown — paper-spec model (49 joints)")
    model = FallRiskAssessmentModel(
        num_joints=49, coord_dim=3,
        channel_schedule=PAPER_SCHED,
        mlp_hidden=128, mlp_out=64,
        attn_dim=64, cls_hidden=256,
        num_classes=3,
    )
    rows = [
        ("Encoder (shared)",           model.encoder),
        ("  └─ GraphEncoder",           model.encoder.graph_encoder),
        ("  └─ MLPViewEncoder (joint)",  model.encoder.joint_encoder),
        ("  └─ MLPViewEncoder (motion)", model.encoder.motion_encoder),
        ("CrossAttentionFusion",         model.fusion),
        ("MLPClassifier",                model.classifier),
    ]
    for name, mod in rows:
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:<40}: {n:>10,}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {'TOTAL':<40}: {total:>10,}")


if __name__ == "__main__":
    print("=" * 62)
    print("  Fall Risk Assessment — Smoke Tests")
    print(f"  Paper backbone  : L=5, channels {PAPER_SCHED}")
    print(f"  Test window size: T={T}  (paper uses T=400)")
    print("=" * 62)

    # ── Preprocessing ──────────────────────────────────────────────────────
    test_multimodal_views()
    test_sliding_window_sampler()
    test_skeleton_augmentor()
    test_augmented_dataset()
    test_build_windowed_dataset()
    test_build_dataset_splits()

    # ── Model ──────────────────────────────────────────────────────────────
    test_att_gcn()
    test_ms_tcn()
    test_st_layer()
    test_graph_encoder_paper_schedule()
    test_full_model_forward_backward()
    test_parameter_breakdown()

    print("\n" + "=" * 62)
    print("  All tests PASSED ✓")
    print("=" * 62)

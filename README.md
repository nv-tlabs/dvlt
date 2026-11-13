<div align="center">
<h1>Déjà View: Looping Transformers for Multi-View 3D Reconstruction</h1>
<h3>NeurIPS 2026 · Oral</h3>

<a href="https://research.nvidia.com/labs/dvl/projects/dvlt/"><img src="https://img.shields.io/badge/Project-Page-1f72b1.svg" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.30215"><img src="https://img.shields.io/badge/arXiv-2605.30215-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/nvidia/dvlt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/nvidia/dvlt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-orange" alt="Hugging Face Spaces Demo"></a>

**[NVIDIA](https://www.nvidia.com/)** &nbsp;&nbsp;&nbsp; **[University of Modena and Reggio Emilia](https://www.unimore.it/it)** &nbsp;&nbsp;&nbsp; **[University of Toronto](https://www.utoronto.ca/)** &nbsp;&nbsp;&nbsp; **[ETH Zurich](https://ethz.ch/)**

[Alessandro Burzio*](https://research.nvidia.com/labs/dvl/author/alessandro-burzio/), [Tobias Fischer*](https://tobiasfshr.github.io/), [Sven Elflein](https://selflein.github.io/), [Qunjie Zhou](https://research.nvidia.com/labs/dvl/author/qunjie-zhou/), [Riccardo de Lutio](https://riccardodelutio.github.io/), [Jiawei Ren](https://jiawei-ren.github.io/), [Jiahui Huang](https://huangjh-pub.github.io/), [Shengyu Huang](https://shengyuh.github.io/), [Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/), [Laura Leal-Taixé](https://research.nvidia.com/labs/dvl/author/laura-leal-taixe/), [Zan Gojcic+](https://zgojcic.github.io/), [Haithem Turki+](https://haithemturki.com/)
</div>

<p align="center">
  <img src="assets/nvidia-hq-dvlt.gif" alt="Déjà View demo" width="100%">
</p>

## Overview

DéjàView (DVLT) is a recurrent transformer for multi-view 3D reconstruction. It
loops a shared block of frame/global attention with discrete depth indexing,
producing per-pixel rays, depth, confidence, and camera poses from an unordered
set of images. Trained once, the number of refinement steps `K` becomes an
inference-time compute knob, matching or outperforming substantially larger
feed-forward baselines at a fraction of their parameters.

This repository contains:

- The DVLT model + four configurable ablations (vanilla, decoupled blocks,
  no `s_out` token, no depth-scaling).
- Evaluation wrappers for five baselines: VGGT, VGGT-Omega, Depth-Anything-3,
  MapAnything, and Pi3. Each wrapper imports the upstream package (installed
  separately — see [INSTALL.md](docs/INSTALL.md)).
- A training stack built on `accelerate` + Hydra, with optional W&B logging.
- A Stage-2 fine-tune recipe for the depth-conv head.
- Rerun-based visualization tools.

## Release status

- [x] Inference code
- [x] Model weights
- [x] Evaluation code
  - [x] eval datasets preprocess and loaders
- [x] Training code
  - [x] ScanNet++ training dataset loader
  - [ ] other training dataset loaders

## Quickstart

### Install

See [docs/INSTALL.md](docs/INSTALL.md). The short version:

```bash
conda create -n dvlt python=3.12 && conda activate dvlt
conda install pytorch=2.5.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge
pip install -e .[all]
```

### Quick setup

Quick example script:

```python
import torch
from accelerate import Accelerator

from dvlt.model.dvlt.model import DVLT
from dvlt.util.preprocess import load_sequence, preprocess_images

checkpoint_path = "nvidia/dvlt"  # local dir, HTTPS URL, or HF Hub repo id
# load_sequence accepts a directory, a single video, or an explicit list of files.
input_path = "path/to/scene_dir"
# Or: input_path = "path/to/clip.mp4"
# Or: from glob import glob; input_path = sorted(glob("path/to/scene_dir/*.png"))

accelerator = Accelerator(mixed_precision="bf16")

model = DVLT(img_size=504)
model.load_pretrained(checkpoint_path, strict=True)
model.setup_test(accelerator)

_, frames = load_sequence(input_path)
batch = preprocess_images(frames, img_size=504, patch_size=14, device=accelerator.device)

with torch.no_grad(), accelerator.autocast():
    predictions = model.predict(batch, accelerator)

cameras = predictions["cameras"][0]            # Cameras object with shape [S]
extrinsics_c2w = cameras.camera_to_worlds       # (S, 3, 4) — OpenCV convention [R | t]
intrinsics = cameras.get_intrinsics_matrices()  # (S, 3, 3)

depths = predictions["depths"][0]              # (S, H, W)
world_points = predictions["world_points"][0]  # (S, H, W, 3)
```

### Train

```bash
# Single-GPU
python -m dvlt.scripts.train --config-name dvlt-large data=scannetpp

# Multi-GPU (4 GPUs)
accelerate launch --num-processes 4 -m dvlt.scripts.train --config-name dvlt-large data=scannetpp

# Resume
python -m dvlt.scripts.train \
    --config-dir=outputs/<run> \
    --config-name=config.yaml \
    trainer.resume_from_checkpoint=latest
```

### Evaluate

`benchmark_lite` (DTU, ETH3D, 7Scenes) is a convenience benchmark over the
datasets that don't require heavy preprocessing; the full `benchmark` adds
[ScanNet++](src/dvlt/scripts/preprocess/preprocess_scannetpp.md) and
[NuScenes](src/dvlt/scripts/preprocess/preprocess_nuscenes.md).

```bash
python -m dvlt.scripts.test --config-name dvlt data=benchmark
# multi-GPU: accelerate launch --num-processes <N> -m dvlt.scripts.test --config-name dvlt data=benchmark
python -m dvlt.scripts.test --config-name dvlt data=benchmark_lite
# multi-GPU: accelerate launch --num-processes <N> -m dvlt.scripts.test --config-name dvlt data=benchmark_lite
```

DVLT reference results on the full `benchmark`:

| Dataset | Pose AUC@3 | Pose AUC@30 | Depth inlier@3% | Depth AbsRel |
|---|---|---|---|---|
| DTU | 0.8319 | 0.9880 | 0.9706 | 0.0093 |
| ETH3D | 0.6604 | 0.9536 | 0.7717 | 0.0267 |
| 7Scenes | 0.1393 | 0.8172 | 0.7437 | 0.0349 |
| ScanNet++ | 0.7941 | 0.9803 | 0.9239 | 0.0167 |
| NuScenes | 0.4340 | 0.8534 | 0.5853 | 0.0673 |


### Interactive demo

Browser UI for uploading images / video and exploring the predicted 3D point
cloud, depth maps and camera trajectory. The dropdown switches between DVLT
and the baseline wrappers (VGGT, VGGT-Omega, DA3, Pi3, MapAnything);
each baseline requires its upstream package installed (see
[docs/INSTALL.md](docs/INSTALL.md)).

```bash
# Launch on http://localhost:7860 (DVLT preselected)
python -m dvlt.scripts.gradio_demo
```

The same script also has a headless **offline mode** that skips Gradio
and writes a `.glb` + `.rrd` per (sequence, model) under
`demo_outputs/<sequence_name>/`. `--input` accepts a directory of images, a
single image, or a video file (mp4/mov/gif/...), and may be repeated to
process multiple sequences in one go; `--models` is a comma-separated list of
config names from the curated registry (or `all`).

```bash
# Run two models on two sequences (one image dir, one video)
python -m dvlt.scripts.gradio_demo --offline \
    --input /path/to/scene_dir \
    --input /path/to/clip.mp4 \
    --models dvlt

# Run every registered model on one sequence
python -m dvlt.scripts.gradio_demo --offline --input /path/to/scene_dir --models all
```

## Configuration

DVLT uses [Hydra](https://hydra.cc) for configuration. Top-level experiment
configs live in `src/dvlt/config/experiments/`:

| Config | Description |
|---|---|
| `dvlt-large` | Stage-1 recipe (large model, full training schedule, linear depth head). |
| `dvlt-large-ablation` | Vanilla ablation parent — toggle decoupled blocks, no-`s_out`, no-depthscale via overrides. |
| `dvlt-large-ablation-decoupled` | Fully decoupled blocks (`recurrence_mode=none`, no looping): a distinct block per step, fixed 16 steps. |
| `dvlt-large-depthconv-stage2` | Stage-2 depth-conv head fine-tune (matches the released checkpoint and the model's default `depth_head_type="conv"`). |
| `dvlt` | Inference-only alias for the released stage-2 checkpoint. |
| `vggt`, `vggt_omega`, `da3-{base,large,giant}`, `pi3`, `pi3x`, `mapanything` | Eval-only baseline wrappers. Require the upstream package installed (see [INSTALL.md](docs/INSTALL.md)). |

### User configuration (data paths)

Per-user settings (most importantly, the dataset root) live in
`src/dvlt/config/experiments/user/`. Copy `default.yaml` to `local.yaml`,
edit `data_root`, and select it via `user=local`:

```bash
python -m dvlt.scripts.train --config-name dvlt-large data=scannetpp user=local
```

`user.data_root` can also be overridden inline or via the `DVLT_DATA_ROOT`
environment variable.

### Selecting datasets

Pick a single curated dataset config:

```bash
python -m dvlt.scripts.train --config-name dvlt-large data=scannetpp
python -m dvlt.scripts.train --config-name dvlt-large data=mixed_all
```

## Tab completion

For scripts using the `@cli` decorator (train, test, visualize):

```bash
eval "$(python -m dvlt.scripts.train -sc install)"
# later, to remove:
eval "$(python -m dvlt.scripts.train -sc uninstall)"
```

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) — environment setup + baseline installs
- [docs/data/DATA.md](docs/data/DATA.md) — data pipeline overview + how to
  add a new dataset parser
- [docs/CONTRIB.md](docs/CONTRIB.md) — dev setup, code style, tests
- [docs/TESTING.md](docs/TESTING.md) — full test-runner documentation

## Acknowledgments

We are also grateful to several other open-source repositories that we drew inspiration from or built upon during the development of our pipeline:
- [VGGT](https://github.com/facebookresearch/vggt)
- [Pi3](https://github.com/yyfz/Pi3)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MapAnything](https://github.com/facebookresearch/map-anything)
- [Depth-Anything-3](https://github.com/bytedance-seed/depth-anything-3)

## Citation

If you find this work useful, please cite:

```bibtex
@article{burzio2026dejaview,
  title   = {D\'ej\`a View: Looping Transformers for Multi-View 3D Reconstruction},
  author  = {Burzio, Alessandro and Fischer, Tobias and Elflein, Sven and Zhou, Qunjie and de Lutio, Riccardo and Ren, Jiawei and Huang, Jiahui and Huang, Shengyu and Pollefeys, Marc and Leal-Taix{\'e}, Laura and Gojcic, Zan and Turki, Haithem},
  journal = {arXiv preprint arXiv:2605.30215},
  year    = {2026}
}
```

## License + attribution

The DVLT **code** is released mostly under the **Apache License, Version 2.0** — see
[LICENSE](LICENSE). The **model weights** (the `nvidia/dvlt` checkpoint) are
released under the **NVIDIA License** — non-commercial, research-and-evaluation
use only; see [LICENSES/NVIDIA-LICENSE.txt](LICENSES/NVIDIA-LICENSE.txt).

Portions of the codebase are adapted from third-party open-source projects
(DINOv2, PyTorch3D, MoGe, AnyCalib, MultiNeRF, Depth-Anything-3, VGGT). Each
adapted file carries the upstream copyright + license notice in its header;
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the full attribution
map and full upstream license texts. The VGGT-derived files are distributed
under the VGGT License; see [LICENSES/VGGT-LICENSE.txt](LICENSES/VGGT-LICENSE.txt).

The baseline evaluation wrappers in `src/dvlt/model/{vggt,vggt_omega,da3,mapanything,pi3}/`
import (do not vendor) their respective upstream packages, each of which is
governed by its own license — see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) §"Upstream packages used
for evaluation".

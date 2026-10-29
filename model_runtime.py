"""Faithful single-image inference helpers for Ali Golbaf's retinal models.

No training, downloading, or predictions without user-supplied checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


GRADE_NAMES = (
    "No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR",
)
CONCEPT_NAMES = (
    "Microaneurysms (MA)",
    "Hard exudates (HE)",
    "Intraretinal haemorrhages (IH)",
    "Venous beading / IRMA",
    "Neovascularisation (NV)",
    "Vitreous haemorrhage (VH)",
    "Retinal detachment (RD)",
)
CONCEPT_CODES = ("MA", "HE", "IH", "VB/IRMA", "NV", "VH", "RD")


def discover_checkpoints(folder: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Select only unambiguously named checkpoint files in ``models/``.

    Never guess a fold/model from several candidates: accidentally pairing
    different model types or folds would undermine the comparison.
    """
    matches: dict[str, list[Path]] = {"blackbox": [], "cbm": []}
    selected: dict[str, Path] = {}
    problems: dict[str, str] = {}

    if not folder.is_dir():
        return selected, {kind: "models/ folder not found" for kind in matches}

    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in {".pt", ".pth"}:
            continue
        name = path.stem.casefold().replace("-", "_").replace(" ", "_")
        is_cbm = any(token in name for token in ("cbm", "concept", "bottleneck"))
        is_blackbox = (
            any(token in name for token in ("blackbox", "black_box", "baseline"))
            or name == "best_model" or name.startswith("best_model_")
            or ("resnet" in name and not is_cbm)
        )
        if is_cbm and not is_blackbox:
            matches["cbm"].append(path)
        elif is_blackbox and not is_cbm:
            matches["blackbox"].append(path)

    for kind, candidates in matches.items():
        if len(candidates) == 1:
            selected[kind] = candidates[0]
        elif not candidates:
            expected = "best_model.pt" if kind == "blackbox" else "best_cbm_model.pt"
            problems[kind] = f"Not found — add {expected} to models/"
        else:
            filenames = ", ".join(path.name for path in candidates)
            problems[kind] = f"Multiple checkpoints — keep one: {filenames}"
    return selected, problems


def decode_fundus(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("The selected image is empty.")
    with Image.open(BytesIO(data)) as source:
        source.load()
        if source.width < 64 or source.height < 64:
            raise ValueError("Please select a fundus image at least 64 × 64 pixels.")
        return source.convert("RGB")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max())
    return values / values.sum()


def scenario_probabilities(weights: np.ndarray, bias: np.ndarray, concept_values: np.ndarray) -> np.ndarray:
    """Re-evaluate ONLY the trained linear CBM grading head."""
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    concept_values = np.asarray(concept_values, dtype=np.float64)
    if weights.shape != (5, 7) or bias.shape != (5,) or concept_values.shape != (7,):
        raise ValueError("Expected weights [5,7], bias [5], and seven concept values.")
    if not np.isfinite(concept_values).all() or np.any((concept_values < 0) | (concept_values > 1)):
        raise ValueError("Concept values must be finite probabilities between zero and one.")
    return softmax(weights @ concept_values + bias)


def margin_contributions(weights: np.ndarray, bias: np.ndarray, concepts: np.ndarray,
                         target: int, reference: int) -> tuple[np.ndarray, float]:
    """Exact additive decomposition of target-versus-reference logit margin."""
    weights = np.asarray(weights, dtype=np.float64)
    return (weights[target] - weights[reference]) * np.asarray(concepts), float(bias[target] - bias[reference])


@dataclass
class RuntimeModels:
    device: str = "cpu"
    blackbox: Any = field(default=None, repr=False)
    cbm: Any = field(default=None, repr=False)
    checkpoint_paths: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _require_torch():
        try:
            import torch
            from torchvision import transforms
        except Exception as error:
            raise RuntimeError(
                "PyTorch and torchvision are required for model inference. "
                "Install dependencies from requirements.txt."
            ) from error
        return torch, transforms

    def load(self, kind: str, checkpoint_path: str) -> dict[str, Any]:
        """Load a compatible state dictionary without executing checkpoint code."""
        if kind not in {"blackbox", "cbm"}:
            raise ValueError(f"Unknown model type: {kind}")

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() not in {".pt", ".pth"}:
            raise ValueError("Select an existing .pt or .pth PyTorch checkpoint.")

        torch, _ = self._require_torch()
        from models import BlackBox_Model, Concept_Bottleneck_Model

        model = (BlackBox_Model(num_grade_classes=5, pretrained=False)
                 if kind == "blackbox"
                 else Concept_Bottleneck_Model(num_grade_classes=5, num_concepts=7, pretrained=False))

        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=True)
            if not isinstance(raw, dict):
                raise ValueError("Checkpoint must contain a state dictionary.")
            state = raw.get("model_state_dict", raw.get("state_dict", raw))
            if not isinstance(state, dict):
                raise ValueError("Model state dictionary is missing.")
            if state and all(str(key).startswith("module.") for key in state):
                state = {str(key)[7:]: value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
        except Exception as error:
            raise RuntimeError(f"Checkpoint is incompatible with the {kind} architecture: {error}") from error

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(self.device).eval()
        except Exception:
            self.device = "cpu"
            model.to("cpu").eval()
            for already_loaded in (self.blackbox, self.cbm):
                if already_loaded is not None:
                    already_loaded.to("cpu").eval()

        if kind == "blackbox":
            self.blackbox = model
        else:
            self.cbm = model
        self.checkpoint_paths[kind] = str(path)
        return {
            "kind": kind,
            "name": path.name,
            "device": self.device,
            "parameters": sum(p.numel() for p in model.parameters()),
        }

    def analyse(self, image_bytes: bytes) -> dict[str, Any]:
        """Run deterministic inference using the training-time test transform."""
        if self.blackbox is None and self.cbm is None:
            raise RuntimeError("Load at least one trained model checkpoint first.")
        torch, transforms = self._require_torch()
        image = decode_fundus(image_bytes)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        x = transform(image).unsqueeze(0).to(self.device)
        result: dict[str, Any] = {}

        try:
            with torch.inference_mode():
                if self.blackbox is not None:
                    outputs = self.blackbox(x)
                    result["blackbox"] = {
                        "grade_probs": torch.softmax(outputs["grade_logits"], dim=1)[0].detach().cpu().numpy(),
                    }
                if self.cbm is not None:
                    outputs = self.cbm(x, return_maps=True)
                    result["cbm"] = {
                        "grade_probs": torch.softmax(outputs["grade_logits"], dim=1)[0].detach().cpu().numpy(),
                        "concept_probs": outputs["concept_probs"][0].detach().cpu().numpy(),
                        "concept_maps": outputs["concept_maps"][0].detach().cpu().numpy(),
                        "weights": self.cbm.grade_head.weight.detach().cpu().numpy(),
                        "bias": self.cbm.grade_head.bias.detach().cpu().numpy(),
                    }
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise RuntimeError("Inference ran out of GPU memory. Close GPU applications or use a CPU-only installation.") from error
            raise

        for name, outputs in result.items():
            if outputs["grade_probs"].shape != (5,) or not np.isfinite(outputs["grade_probs"]).all():
                raise RuntimeError(f"Invalid five-grade output from {name}.")
        if "cbm" in result:
            c = result["cbm"]
            if c["concept_probs"].shape != (7,) or c["concept_maps"].shape[0] != 7:
                raise RuntimeError("The CBM did not return the expected seven concepts and evidence maps.")
            if not all(np.isfinite(c[key]).all() for key in ("concept_probs", "concept_maps", "weights", "bias")):
                raise RuntimeError("The CBM produced a non-finite value.")
        return result

"""QMT-PAT model definitions, copied verbatim from notebooks/01_main_5fold_vit_densenet_qmtpat.ipynb
(PatientAttentionAnchor, the task-structured 16-qubit circuit, QuantumCoreV9 and QMT_PAT_V9),
plus the fine-tuned ViT-B/16 image encoder (ViTMultiTask)."""
import copy, math
import torch
import torch.nn as nn
import pennylane as qml

TASKS = {
    "H_pylori_infection_status": ["negative", "positive"],
    "Presence_of_atrophy": ["non_atrophic", "atrophic"],
    "Distribution_of_atrophy": ["normal", "antrum_predominant", "corpus_predominant", "antrum_and_corpus"],
    "Gastritis_type": ["normal", "type_A", "type_B"],
}
TASK_ORDER = list(TASKS.keys())
N_TASKS = len(TASK_ORDER)
LABEL_MAPS = {t: {c: i for i, c in enumerate(cs)} for t, cs in TASKS.items()}
VIT_FEATURE_DIM = 768
MODEL_DIM = 256
CLASSICAL_DIM = 128
ATTN_HEADS = 4
DROPOUT = 0.30
FEATURE_DROPOUT = 0.10
MAX_IMAGES_EVAL = 108
QUBITS_PER_TASK = 4
N_QUBITS = N_TASKS * QUBITS_PER_TASK
N_QLAYERS = 3
ANGLES_PER_TASK = N_QLAYERS * QUBITS_PER_TASK * 2
N_ENCODING_ANGLES = N_TASKS * ANGLES_PER_TASK
N_Q_OBSERVABLES = 3 * N_QUBITS
Q_WEIGHT_STD = 0.035
Q_ENCODING_STD = 0.015
ANGLE_HIDDEN_DIM = 96
FUSION_DIM = 128

class PatientAttentionAnchor(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.feature_adapter = (
            nn.Sequential(
                nn.LayerNorm(
                    VIT_FEATURE_DIM
                ),
                nn.Linear(
                    VIT_FEATURE_DIM,
                    MODEL_DIM,
                ),
                nn.GELU(),
                nn.Dropout(
                    FEATURE_DROPOUT
                ),
                nn.LayerNorm(
                    MODEL_DIM
                ),
            )
        )

        self.task_queries = (
            nn.Parameter(
                torch.randn(
                    1,
                    N_TASKS,
                    MODEL_DIM,
                ) * 0.02
            )
        )

        self.patient_attention = (
            nn.MultiheadAttention(
                embed_dim=MODEL_DIM,
                num_heads=ATTN_HEADS,
                dropout=0.10,
                batch_first=True,
            )
        )

        self.attn_norm = (
            nn.LayerNorm(
                MODEL_DIM
            )
        )

        self.classical_branches = (
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(
                        MODEL_DIM,
                        CLASSICAL_DIM,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(
                        CLASSICAL_DIM
                    ),
                    nn.Dropout(
                        DROPOUT
                    ),
                )
                for _ in TASK_ORDER
            ])
        )

        self.classical_heads = (
            nn.ModuleList([
                nn.Linear(
                    CLASSICAL_DIM,
                    len(TASKS[task]),
                )
                for task
                in TASK_ORDER
            ])
        )

    def forward(
        self,
        padded_features,
        padding_mask,
    ):
        x = self.feature_adapter(
            padded_features
        )

        queries = (
            self.task_queries.expand(
                x.shape[0],
                -1,
                -1,
            )
        )

        attended, attn_weights = (
            self.patient_attention(
                query=queries,
                key=x,
                value=x,
                key_padding_mask=(
                    padding_mask
                ),
                need_weights=True,
                average_attn_weights=False,
            )
        )

        task_embeddings = (
            self.attn_norm(
                queries
                + attended
            )
        )

        logits = []

        for task_idx in range(
            N_TASKS
        ):
            hidden = (
                self.classical_branches[
                    task_idx
                ](
                    task_embeddings[
                        :,
                        task_idx,
                        :,
                    ]
                )
            )

            logits.append(
                self.classical_heads[
                    task_idx
                ](
                    hidden
                )
            )

        return {
            "task_embeddings": (
                task_embeddings
            ),
            "logits": logits,
            "attention_weights": (
                attn_weights
            ),
        }
# ---- cell 14

def _angle_index(task_idx, layer_idx, local_wire, axis_idx):
    return (
        (((task_idx * N_QLAYERS + layer_idx) * QUBITS_PER_TASK + local_wire) * 2)
        + axis_idx
    )

def _make_quantum_device():
    try:
        return qml.device("lightning.qubit", wires=N_QUBITS)
    except Exception:
        print("lightning.qubit unavailable; falling back to default.qubit")
        return qml.device("default.qubit", wires=N_QUBITS)

def make_v9_quantum_qnode(cross_task=True):
    qdev = _make_quantum_device()

    @qml.qnode(
        qdev,
        interface="torch",
        diff_method="adjoint",
    )
    def circuit(
        inputs,
        weights,
        encoding_scale,
        encoding_bias,
    ):
        for layer_idx in range(N_QLAYERS):
            for task_idx in range(N_TASKS):
                base_wire = task_idx * QUBITS_PER_TASK

                for local_wire in range(QUBITS_PER_TASK):
                    wire = base_wire + local_wire

                    y_idx = _angle_index(
                        task_idx, layer_idx, local_wire, 0
                    )
                    z_idx = _angle_index(
                        task_idx, layer_idx, local_wire, 1
                    )

                    y_angle = (
                        inputs[..., y_idx]
                        * encoding_scale[layer_idx, wire, 0]
                        + encoding_bias[layer_idx, wire, 0]
                    )
                    z_angle = (
                        inputs[..., z_idx]
                        * encoding_scale[layer_idx, wire, 1]
                        + encoding_bias[layer_idx, wire, 1]
                    )

                    qml.RY(y_angle, wires=wire)
                    qml.RZ(z_angle, wires=wire)

            for wire in range(N_QUBITS):
                qml.Rot(
                    weights[layer_idx, wire, 0],
                    weights[layer_idx, wire, 1],
                    weights[layer_idx, wire, 2],
                    wires=wire,
                )

            for task_idx in range(N_TASKS):
                base_wire = task_idx * QUBITS_PER_TASK
                for local_wire in range(QUBITS_PER_TASK):
                    qml.CZ(
                        wires=[
                            base_wire + local_wire,
                            base_wire + ((local_wire + 1) % QUBITS_PER_TASK),
                        ]
                    )

            if cross_task:
                shift = layer_idx % QUBITS_PER_TASK
                for task_idx in range(N_TASKS):
                    next_task = (task_idx + 1) % N_TASKS
                    base_a = task_idx * QUBITS_PER_TASK
                    base_b = next_task * QUBITS_PER_TASK

                    for local_wire in range(QUBITS_PER_TASK):
                        target_local = (local_wire + shift) % QUBITS_PER_TASK
                        qml.CNOT(
                            wires=[
                                base_a + local_wire,
                                base_b + target_local,
                            ]
                        )

        observables = []

        for wire in range(N_QUBITS):
            observables.append(
                qml.expval(qml.PauliZ(wire))
            )

        for task_idx in range(N_TASKS):
            base_wire = task_idx * QUBITS_PER_TASK
            for local_wire in range(QUBITS_PER_TASK):
                observables.append(
                    qml.expval(
                        qml.PauliZ(base_wire + local_wire)
                        @ qml.PauliZ(
                            base_wire + ((local_wire + 1) % QUBITS_PER_TASK)
                        )
                    )
                )

        for task_idx in range(N_TASKS):
            next_task = (task_idx + 1) % N_TASKS
            base_a = task_idx * QUBITS_PER_TASK
            base_b = next_task * QUBITS_PER_TASK

            for local_wire in range(QUBITS_PER_TASK):
                observables.append(
                    qml.expval(
                        qml.PauliZ(base_a + local_wire)
                        @ qml.PauliZ(base_b + local_wire)
                    )
                )

        return observables

    return circuit

class QuantumCoreV9(nn.Module):
    def __init__(self, cross_task=True):
        super().__init__()
        self.cross_task = bool(cross_task)

        weight_shapes = {
            "weights": (N_QLAYERS, N_QUBITS, 3),
            "encoding_scale": (N_QLAYERS, N_QUBITS, 2),
            "encoding_bias": (N_QLAYERS, N_QUBITS, 2),
        }

        qnode = make_v9_quantum_qnode(
            cross_task=self.cross_task
        )

        self.layer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes,
        )

        params = dict(self.layer.named_parameters())

        nn.init.normal_(
            params["weights"],
            mean=0.0,
            std=Q_WEIGHT_STD,
        )

        with torch.no_grad():
            params["encoding_scale"].fill_(1.0)
            params["encoding_scale"].add_(
                torch.randn_like(params["encoding_scale"]) * Q_ENCODING_STD
            )
            params["encoding_bias"].zero_()
            params["encoding_bias"].add_(
                torch.randn_like(params["encoding_bias"]) * Q_ENCODING_STD
            )

    def forward(self, x):
        return self.layer(x)

class ParameterMatchedClassicalCoreV9(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_ENCODING_ANGLES, 2),
            nn.Tanh(),
            nn.Linear(2, N_Q_OBSERVABLES),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ---- cell 16
class QMT_PAT_V9(nn.Module):
    def __init__(
        self,
        anchor,
        core_type="quantum",
    ):
        super().__init__()

        assert core_type in {
            "quantum",
            "quantum_local",
            "classical",
        }

        self.core_type = core_type
        self.anchor = copy.deepcopy(anchor)

        for p in self.anchor.parameters():
            p.requires_grad = False

        joint_dim = 2 * MODEL_DIM

        self.angle_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, ANGLE_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(ANGLE_HIDDEN_DIM, ANGLES_PER_TASK),
                nn.Tanh(),
            )
            for _ in TASK_ORDER
        ])

        self.context_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, FUSION_DIM),
                nn.GELU(),
                nn.LayerNorm(FUSION_DIM),
                nn.Tanh(),
            )
            for _ in TASK_ORDER
        ])

        self.core_norm = nn.LayerNorm(N_Q_OBSERVABLES)

        self.core_readouts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(N_Q_OBSERVABLES, FUSION_DIM),
                nn.GELU(),
                nn.LayerNorm(FUSION_DIM),
                nn.Dropout(DROPOUT),
            )
            for _ in TASK_ORDER
        ])

        self.delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * FUSION_DIM),
                nn.Linear(2 * FUSION_DIM, 64),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(64, len(TASKS[task])),
            )
            for task in TASK_ORDER
        ])

        self.core_aux_heads = nn.ModuleList([
            nn.Linear(FUSION_DIM, len(TASKS[task]))
            for task in TASK_ORDER
        ])

        for head in self.delta_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=0.005)
            nn.init.zeros_(head[-1].bias)

        if core_type == "quantum":
            self.core = QuantumCoreV9(cross_task=True)
        elif core_type == "quantum_local":
            self.core = QuantumCoreV9(cross_task=False)
        else:
            self.core = ParameterMatchedClassicalCoreV9()

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self,
        features,
        padding_mask,
    ):
        self.anchor.eval()

        with torch.no_grad():
            anchor_out = self.anchor(
                features,
                padding_mask,
            )

            task_embeddings = anchor_out[
                "task_embeddings"
            ].detach()

            anchor_logits = [
                x.detach()
                for x in anchor_out["logits"]
            ]

        global_embedding = task_embeddings.mean(
            dim=1
        )

        task_joint = []
        task_context = []
        angle_blocks = []

        for task_idx in range(N_TASKS):
            joint = torch.cat(
                [
                    task_embeddings[:, task_idx, :],
                    global_embedding,
                ],
                dim=1,
            )
            task_joint.append(joint)

            context = self.context_projectors[
                task_idx
            ](joint)
            task_context.append(context)

            task_angles = self.angle_projectors[
                task_idx
            ](joint)
            angle_blocks.append(task_angles)

        angles = torch.cat(
            angle_blocks,
            dim=1,
        ) * math.pi

        if self.core_type.startswith("quantum"):
            core_cpu = self.core(
                angles.float().to("cpu")
            )
            core_features = core_cpu.to(
                device=features.device,
                dtype=features.dtype,
            )
        else:
            core_features = self.core(angles)

        core_features = self.core_norm(
            core_features
        )

        core_task_features = []
        core_aux_logits = []
        deltas = []
        final_logits = []

        for task_idx, task in enumerate(TASK_ORDER):
            q_task = self.core_readouts[
                task_idx
            ](core_features)

            context = task_context[task_idx]

            interaction = q_task * context

            fusion = torch.cat(
                [q_task, interaction],
                dim=1,
            )

            raw_delta = self.delta_heads[
                task_idx
            ](fusion)

            delta = raw_delta - raw_delta.mean(
                dim=1,
                keepdim=True,
            )

            final = anchor_logits[task_idx] + delta

            aux = self.core_aux_heads[
                task_idx
            ](q_task)

            core_task_features.append(q_task)
            core_aux_logits.append(aux)
            deltas.append(delta)
            final_logits.append(final)

        return {
            "anchor_logits": anchor_logits,
            "core_features": core_features,
            "core_task_features": core_task_features,
            "core_aux_logits": core_aux_logits,
            "deltas": deltas,
            "final_logits": final_logits,
            "angles": angles,
        }



# ---- cell 8 (image encoder)
from torchvision import models


class ViTMultiTask(nn.Module):
    """ViT-B/16 with one linear head per task on the class token (Section 3.4).

    Built without downloading ImageNet weights: the fine-tuned checkpoint
    overwrites every parameter."""

    def __init__(self):
        super().__init__()
        backbone = models.vit_b_16(weights=None)
        self.feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            task: nn.Linear(self.feature_dim, len(classes))
            for task, classes in TASKS.items()
        })

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        return {task: head(features) for task, head in self.heads.items()}

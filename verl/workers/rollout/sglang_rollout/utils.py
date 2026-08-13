# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from dataclasses import asdict
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig
from verl.utils.device import get_device_name
from verl.workers.rollout.utils import ensure_async_iterator

SGLANG_LORA_NAME = "verl_actor_lora_name"


def sglang_lora_target_modules(target_modules: Any) -> list[str]:
    """Render verl's ``model.target_modules`` as SGLang's ``lora_target_modules``.

    Following PEFT, verl overloads this field by type: a list names modules matched
    exactly or by suffix, while a bare string is either the ``"all-linear"`` shorthand
    or a *regex* matched against the whole parameter key -- see
    :func:`verl.utils.model.check_target_modules`, which dispatches on exactly that.

    SGLang supports neither form. It normalizes the field with ``set(...)``, so a bare
    string is torn into its characters and the LoRA memory pool later dies on one of
    them::

        NotImplementedError: get_hidden_dim not implemented for i

    ``"all-linear"`` translates to SGLang's own ``"all"`` sentinel, which it expands
    itself. A regex has no SGLang equivalent, and reading it as a literal module name
    would silently adapt a different set of modules than training did, so it is
    rejected with an actionable message instead.
    """
    if target_modules == "all-linear":
        return ["all"]
    if isinstance(target_modules, str):
        raise ValueError(
            f"SGLang cannot serve a regex `target_modules` ({target_modules!r}). PEFT matches a "
            f"string against the full parameter key, which SGLang's LoRA pool has no equivalent "
            f"of, and reading it as a literal module name would adapt different modules than "
            f"training did. Use `all-linear`, or list the module names explicitly, e.g. "
            f"[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]."
        )
    return list(target_modules)


def normalize_peft_config_for_sglang(peft_config: dict) -> dict:
    """Normalize an engine's adapter config into what SGLang's adapter loader accepts.

    ``BaseEngine.get_per_tensor_param`` declares this value ``Optional[dict]``, and the
    FSDP engine honours that with ``LoraConfig.to_dict()`` -- which leaves ``task_type``
    and ``peft_type`` as enum members rather than the strings the wire format needs, so
    unwrap them. The input is not mutated.

    The megatron engine returns a differently shaped dict: ``build_peft_config_for_vllm()``
    carries no ``peft_type`` key at all. Rather than guess a value for a path that has not
    been exercised, reject it with a message that says so. Nothing regresses -- that
    combination cannot reach SGLang today either, since the caller crashes further up.
    """
    normalized = asdict(peft_config) if isinstance(peft_config, LoraConfig) else dict(peft_config)
    for key in ("task_type", "peft_type"):
        if key in normalized:
            normalized[key] = getattr(normalized[key], "value", normalized[key])
    if "peft_type" not in normalized:
        raise ValueError(
            "adapter config has no 'peft_type', which SGLang's adapter loader requires. "
            "The megatron engine's build_peft_config_for_vllm() omits it; that pairing is "
            "not covered by this code path. Keys present: " + ", ".join(sorted(normalized))
        )
    normalized["target_modules"] = sglang_lora_target_modules(normalized["target_modules"])
    return normalized


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


async def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, tensor.clone())]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor.clone()))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket

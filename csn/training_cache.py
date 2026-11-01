"""Bounded caches for immutable cell graphs; no model outputs are cached."""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch


class _ByteLRU:
    def __init__(self, budget):
        self.budget = int(budget)
        self.bytes = 0
        self.items = OrderedDict()

    def get(self, key):
        item = self.items.get(key)
        if item is not None:
            self.items.move_to_end(key)
            return item[0]
        return None

    def put(self, key, value, size):
        if size > self.budget:
            return
        previous = self.items.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        while self.items and self.bytes + size > self.budget:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
        self.items[key] = (value, size)
        self.bytes += size

    def clear(self):
        self.items.clear()
        self.bytes = 0


@dataclass
class PreparedGraph:
    inputs: tuple
    targets: tuple
    relation_layouts: tuple = None
    edge_output_indices: tuple = None

    @property
    def decoder_kwargs(self):
        return dict(relation_layouts=self.relation_layouts,
                    edge_output_indices=self.edge_output_indices)

    @property
    def valid(self):
        return self.targets[0].numel() > 1 and self.targets[1].numel() > 1


@dataclass
class RelationLayout:
    relation_types: torch.Tensor
    type_index: torch.Tensor
    symmetric_index: torch.Tensor
    paired_index: torch.Tensor
    reverse_index: torch.Tensor
    unpaired_index: torch.Tensor


class TrainingGraphCache:
    """Reuse shared tensors and exact edge masks, with bounded CPU/GPU memory.

    Cell graphs must remain unchanged for the lifetime of this cache. Packed
    CPU masks avoid storing full per-cell relation matrices for every epoch.
    """

    def __init__(self, device, budget_mb=256, compact_relations=False):
        self.device = torch.device(device)
        self.compact_relations = compact_relations
        budget = int(budget_mb * 1024 ** 2)
        device_budget = budget
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            device_budget = min(budget, int(free // 16))
        self.masks = _ByteLRU(budget)
        self.shared = _ByteLRU(device_budget)
        self.prepared = _ByteLRU(device_budget)
        self.catalogs = _ByteLRU(device_budget)

    def _relation_catalog(self, data):
        relations = getattr(data, "edge_relation_attr", None)
        symmetric = getattr(data, "edge_symmetric_mask", None)
        count = data.edge_index.shape[1]
        key = (id(data.edge_index), id(relations), id(symmetric), count)
        cached = self.catalogs.get(key)
        if cached is not None:
            return cached
        if relations is None:
            unique = np.ones((1, 1), dtype=np.float32)
            inverse = np.zeros(count, dtype=np.int64)
        else:
            values = relations.detach().cpu().numpy()
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            unique, inverse = np.unique(values, axis=0, return_inverse=True)
        symmetric_cpu = (np.zeros(count, dtype=bool) if symmetric is None
                         else symmetric.detach().cpu().numpy().astype(bool))
        # Match reverse endpoints AND the full relation vector. Missing reverse
        # edges are handled by the ordinary reverse-residual computation.
        edge_cpu = data.edge_index.detach().cpu().numpy()
        keys = np.empty(count, dtype=[("src", np.int64), ("dst", np.int64), ("relation", np.int64)])
        keys["src"], keys["dst"], keys["relation"] = edge_cpu[0], edge_cpu[1], inverse
        order = np.argsort(keys, kind="stable")
        reverse_keys = keys.copy()
        reverse_keys["src"], reverse_keys["dst"] = edge_cpu[1], edge_cpu[0]
        positions = np.searchsorted(keys[order], reverse_keys)
        present = positions < count
        present[present] &= keys[order[positions[present]]] == reverse_keys[present]
        reverse_cpu = np.full(count, -1, dtype=np.int64)
        reverse_cpu[present] = order[positions[present]]
        types = torch.from_numpy(unique).to(self.device)
        type_index = torch.from_numpy(inverse.astype(np.int64)).to(self.device)
        symmetric_index = np.flatnonzero(symmetric_cpu)
        paired = reverse_cpu[symmetric_index] >= 0
        metadata = (symmetric_index, symmetric_index[paired], reverse_cpu[symmetric_index[paired]],
                    symmetric_index[~paired])
        catalog = RelationLayout(types, type_index,
                                 *(torch.from_numpy(array).to(self.device) for array in metadata))
        size = sum(t.numel() * t.element_size() for t in vars(catalog).values())
        self.catalogs.put(key, catalog, size)
        return catalog

    def _on_device(self, tensor):
        if tensor.device == self.device:
            return tensor
        key = id(tensor)
        cached = self.shared.get(key)
        if cached is not None and cached[0] is tensor:
            return cached[1]
        copied = tensor.to(self.device)
        self.shared.put(key, (tensor, copied), copied.numel() * copied.element_size())
        return copied

    def get(self, data):
        key = id(data)
        cached = self.prepared.get(key)
        if cached is not None:
            return cached
        packed = self.masks.get(key)
        if packed is None:
            weights = data.edge_attr.detach().cpu().numpy()
            packed = tuple(np.packbits(weights == value) for value in (0, 1))
            self.masks.put(key, packed, sum(item.nbytes for item in packed))
        cpu_indices = tuple(
            np.flatnonzero(np.unpackbits(item, count=data.edge_attr.numel()))
            for item in packed
        )
        indices = tuple(torch.from_numpy(item).to(self.device) for item in cpu_indices)
        edges = self._on_device(data.edge_index)
        layouts = None
        if self.compact_relations:
            layouts = (self._relation_catalog(data), None)
            # Decode the shared candidate graph once, then select original
            # positive/negative logits for their separate loss reductions.
            edge_inputs = (edges, edges.index_select(1, indices[1]))
            relation_inputs = (None, None, None, None)
        else:
            edge_inputs = tuple(edges.index_select(1, index) for index in indices)
            relations = getattr(data, "edge_relation_attr", None)
            if relations is None:
                relations = torch.ones((edges.shape[1], 1), device=self.device)
            else:
                relations = self._on_device(relations)
                if relations.ndim == 1:
                    relations = relations.unsqueeze(1)
            symmetric = getattr(data, "edge_symmetric_mask", None)
            if symmetric is None:
                symmetric = torch.zeros(edges.shape[1], dtype=torch.bool, device=self.device)
            else:
                symmetric = self._on_device(symmetric)
            relation_inputs = (*(relations.index_select(0, index) for index in indices),
                               *(symmetric.index_select(0, index) for index in indices))
        weights = data.edge_attr.to(self.device)
        graph = PreparedGraph(
            inputs=(
                data.x.to(self.device),
                *edge_inputs,
                *relation_inputs,
            ),
            targets=(*(weights.index_select(0, index) for index in indices), data.y.view(-1).to(self.device)),
            relation_layouts=layouts,
            edge_output_indices=indices if self.compact_relations else None,
        )
        size = sum(t.numel() * t.element_size() for t in graph.inputs + graph.targets if t is not None)
        if layouts is not None:
            # Count shared storage conservatively to keep eviction bounded.
            size += sum(t.numel() * t.element_size() for t in vars(layouts[0]).values())
            size += sum(t.numel() * t.element_size() for t in indices)
        self.prepared.put(key, graph, size)
        return graph

    def clear(self):
        self.prepared.clear()
        self.shared.clear()
        self.masks.clear()
        self.catalogs.clear()

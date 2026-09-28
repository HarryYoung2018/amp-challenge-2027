"""Local, restored model hooks count actual deep-copied native work."""

from __future__ import annotations

import contextlib
import copy
import weakref

from amp_challenge.generators.diffusion.model import NativeDenoiser


class NativeWorkCounter:
    def __init__(self, units):
        self.phase = "preflight"
        self.rows = {}
        self.models = weakref.WeakSet()
        self.handles = []
        self.backward_handles = []
        self.active = False
        self.initial_models = tuple(
            {id(model): model for unit in units for model in (unit.model, unit.reference)}.values()
        )

        def observe(model, inputs, output):
            if type(model) is not NativeDenoiser:
                raise TypeError("native work counter observed a different model type")
            self.models.add(model)
            phase = self.phase
            row = self.rows.setdefault(
                phase,
                {
                    "row_forwards": 0,
                    "forward_calls": 0,
                    "grad_enabled_forwards": 0,
                    "backward_calls": 0,
                },
            )
            row["row_forwards"] += int(inputs[0].shape[0])
            row["forward_calls"] += 1
            if output.requires_grad:
                row["grad_enabled_forwards"] += 1

                def backward(gradient):
                    self.rows[phase]["backward_calls"] += 1
                    return None

                self.backward_handles.append(output.register_hook(backward))
            # Returning None leaves the real forward output unchanged. Python
            # function closures retain this counter when modules are deep-copied.
            return None

        self.hook = observe

    def __enter__(self):
        if self.active:
            raise ValueError("native work counter is already active")
        for model in self.initial_models:
            self.models.add(model)
            self.handles.append(model.register_forward_hook(self.hook))
        self.active = True
        return self

    def __exit__(self, *exc):
        for handle in (*self.handles, *self.backward_handles):
            handle.remove()
        # Copies inherit the hook IDs but not the original removable handles.
        for model in tuple(self.models):
            for key, hook in tuple(model._forward_hooks.items()):
                if hook is self.hook:
                    model._forward_hooks.pop(key)
                    model._forward_hooks_with_kwargs.pop(key, None)
                    model._forward_hooks_always_called.pop(key, None)
        self.active = False

    def require_active(self, units, *, fresh=False):
        """Check real hook coverage; never reset or subtract an opening ledger."""
        if not self.active or any(
            not any(hook is self.hook for hook in model._forward_hooks.values())
            for unit in units
            for model in (unit.model, unit.reference)
        ):
            raise ValueError(
                "native work counter must be active and cover all original/reference models"
            )
        if fresh and self.rows:
            raise ValueError("standalone partial update requires a fresh zero work ledger")

    @contextlib.contextmanager
    def at(self, phase):
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    def document(self):
        totals = {
            name: sum(row[name] for row in self.rows.values())
            for name in ("row_forwards", "forward_calls", "grad_enabled_forwards", "backward_calls")
        }
        return {
            "phases": copy.deepcopy(self.rows),
            "total": totals,
            "shared_inner_subphase_split_measured": False,
            "external_feature_model_work_counted_here": False,
        }

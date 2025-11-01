"""
Constraint helpers for structure-token decoding.

These utilities derive token categories from the dataset metadata and
enforce the simplified sequence grammar while sampling crystal structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def _is_prefix(token: str | None, prefix: str) -> bool:
    return token is not None and token.startswith(prefix)


@dataclass
class _CategorySets:
    version: set[int]
    length_a: set[int]
    length_b_over_a: set[int]
    length_c_over_a: set[int]
    angle_alpha: set[int]
    angle_beta: set[int]
    angle_gamma: set[int]
    elements: set[int]
    frac_x: set[int]
    frac_y: set[int]
    frac_z: set[int]


class CrystalConstraintBuilder:
    """Factory that tags vocabulary entries and spawns constraint states."""

    _LATTICE_SEQUENCE = (
        "length_a",
        "length_b_over_a",
        "length_c_over_a",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
    )
    _SITE_SEQUENCE = ("elements", "frac_x", "frac_y", "frac_z")

    def __init__(
        self,
        stoi: Mapping[str, int],
        itos: Mapping[int, str],
        eos_token: str | None,
        bos_token: str | None,
        pad_token_id: int,
    ) -> None:
        self._stoi = stoi
        self._itos = itos
        self._pad_token_id = pad_token_id
        self._eos_id = stoi.get(eos_token) if eos_token is not None else None
        self._bos_id = stoi.get(bos_token) if bos_token is not None else None

        self._categories = _CategorySets(
            version=set(),
            length_a=set(),
            length_b_over_a=set(),
            length_c_over_a=set(),
            angle_alpha=set(),
            angle_beta=set(),
            angle_gamma=set(),
            elements=set(),
            frac_x=set(),
            frac_y=set(),
            frac_z=set(),
        )
        self._token_category: dict[int, str] = {}
        self._populate_categories()

        self._lattice_sequence = [
            name
            for name in self._LATTICE_SEQUENCE
            if getattr(self._categories, name)
        ]
        self._site_sequence = [
            name
            for name in self._SITE_SEQUENCE
            if getattr(self._categories, name)
        ]

    @classmethod
    def from_meta(cls, meta: Mapping[str, object]) -> CrystalConstraintBuilder:
        stoi = meta.get("stoi")  # type: ignore[assignment]
        if not isinstance(stoi, Mapping):
            raise TypeError("meta['stoi'] must be a mapping.")

        itos_raw = meta.get("itos")
        if isinstance(itos_raw, Mapping):
            itos = dict(itos_raw)
        elif isinstance(itos_raw, Sequence):
            itos = {idx: tok for idx, tok in enumerate(itos_raw)}
        else:
            raise TypeError("meta['itos'] must be a mapping or list.")

        eos_token = meta.get("eos_token")
        bos_token = meta.get("bos_token")
        pad_token_id = int(meta.get("pad_token_id", 0))

        return cls(
            stoi=stoi,  # type: ignore[arg-type]
            itos=itos,
            eos_token=eos_token if isinstance(eos_token, str) else None,
            bos_token=bos_token if isinstance(bos_token, str) else None,
            pad_token_id=pad_token_id,
        )

    @property
    def eos_id(self) -> int | None:
        return self._eos_id

    @property
    def bos_id(self) -> int | None:
        return self._bos_id

    def vocab_size(self) -> int:
        return len(self._itos)

    def _populate_categories(self) -> None:
        for idx, token in self._itos.items():
            if not isinstance(token, str):
                continue
            if token == "[PAD]" or idx == self._pad_token_id:
                self._token_category[idx] = "pad"
            elif token == "[EOS]" or idx == self._eos_id:
                self._token_category[idx] = "eos"
            elif token == "[BOS]" or idx == self._bos_id:
                self._token_category[idx] = "bos"
            elif _is_prefix(token, "[VER:"):
                self._categories.version.add(idx)
                self._token_category[idx] = "version"
            elif _is_prefix(token, "[a:"):
                self._categories.length_a.add(idx)
                self._token_category[idx] = "length_a"
            elif _is_prefix(token, "[b/a:"):
                self._categories.length_b_over_a.add(idx)
                self._token_category[idx] = "length_b_over_a"
            elif _is_prefix(token, "[c/a:"):
                self._categories.length_c_over_a.add(idx)
                self._token_category[idx] = "length_c_over_a"
            elif _is_prefix(token, "[alpha:"):
                self._categories.angle_alpha.add(idx)
                self._token_category[idx] = "angle_alpha"
            elif _is_prefix(token, "[beta:"):
                self._categories.angle_beta.add(idx)
                self._token_category[idx] = "angle_beta"
            elif _is_prefix(token, "[gamma:"):
                self._categories.angle_gamma.add(idx)
                self._token_category[idx] = "angle_gamma"
            elif _is_prefix(token, "[El:"):
                self._categories.elements.add(idx)
                self._token_category[idx] = "elements"
            elif _is_prefix(token, "[fx:"):
                self._categories.frac_x.add(idx)
                self._token_category[idx] = "frac_x"
            elif _is_prefix(token, "[fy:"):
                self._categories.frac_y.add(idx)
                self._token_category[idx] = "frac_y"
            elif _is_prefix(token, "[fz:"):
                self._categories.frac_z.add(idx)
                self._token_category[idx] = "frac_z"
            else:
                self._token_category[idx] = "unknown"

    def new_state(self, prefix: Iterable[int] | None = None) -> CrystalConstraintState:
        state = CrystalConstraintState(self)
        if prefix:
            state.observe_sequence(prefix)
        return state

    def category_for_id(self, token_id: int) -> str:
        return self._token_category.get(token_id, "unknown")

    def category_ids(self, category: str) -> set[int]:
        return getattr(self._categories, category, set())

    def token_text(self, token_id: int) -> str | None:
        return self._itos.get(token_id)

    def lattice_sequence(self) -> Sequence[str]:
        return self._lattice_sequence

    def site_sequence(self) -> Sequence[str]:
        return self._site_sequence if self._site_sequence else ("elements",)


class CrystalConstraintState:
    """Track decoding progress and expose allowed token ids."""

    def __init__(self, builder: CrystalConstraintBuilder) -> None:
        self._builder = builder
        self._disabled = False
        self._finished = False
        self._phase = "version" if builder.category_ids("version") else "lattice"
        self._lattice_sequence = list(builder.lattice_sequence())
        self._site_sequence = list(builder.site_sequence())
        self._lattice_index = 0
        self._site_phase = "elements"
        if self._phase == "lattice":
            self._advance_lattice_if_empty()
        self._refresh_site_phase()

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def is_finished(self) -> bool:
        return self._finished

    def observe_sequence(self, tokens: Iterable[int]) -> None:
        for token_id in tokens:
            self.observe_token(token_id)
            if self._finished:
                break

    def observe_token(self, token_id: int) -> None:
        if self._disabled or self._finished:
            if token_id == self._builder.eos_id:
                self._finished = True
            return

        category = self._builder.category_for_id(token_id)
        if category in {"pad", "bos"}:
            return
        if category == "eos":
            if self._phase != "sites" or self._site_phase != "elements":
                self._disable()
                return
            self._finished = True
            return

        if self._phase == "version":
            if category == "version":
                self._phase = "lattice" if self._lattice_sequence else "sites"
                if self._phase == "lattice":
                    self._advance_lattice_if_empty()
                else:
                    self._set_sites_phase()
                return
            self._phase = "lattice" if self._lattice_sequence else "sites"
            self.observe_token(token_id)
            return

        if self._phase == "lattice":
            if self._lattice_index >= len(self._lattice_sequence):
                self._enter_post_lattice_phase()
                self.observe_token(token_id)
                return
            expected = self._lattice_sequence[self._lattice_index]
            if category != expected:
                self._disable()
                return
            self._lattice_index += 1
            self._advance_lattice_if_empty()
            return

        if self._phase == "sites":
            self._consume_site_token(category)
            return

        self._disable()

    def allowed_token_ids(self) -> list[int]:
        if self._disabled:
            return []
        if self._finished:
            return [self._builder.eos_id] if self._builder.eos_id is not None else []

        if self._phase == "version":
            return sorted(self._builder.category_ids("version"))

        if self._phase == "lattice":
            if self._lattice_index >= len(self._lattice_sequence):
                self._enter_post_lattice_phase()
                return self.allowed_token_ids()
            category = self._lattice_sequence[self._lattice_index]
            return sorted(self._builder.category_ids(category))

        if self._phase == "sites":
            return self._allowed_site_tokens()

        return []

    def _allowed_site_tokens(self) -> list[int]:
        if self._site_phase == "elements":
            allowed = set(self._builder.category_ids("elements"))
            eos_id = self._builder.eos_id
            if eos_id is not None:
                allowed.add(eos_id)
            return sorted(allowed)
        if self._site_phase == "frac_x":
            return sorted(self._builder.category_ids("frac_x"))
        if self._site_phase == "frac_y":
            return sorted(self._builder.category_ids("frac_y"))
        if self._site_phase == "frac_z":
            return sorted(self._builder.category_ids("frac_z"))
        return []

    def _consume_site_token(self, category: str) -> None:
        if self._site_phase == "elements":
            if category == "elements":
                self._advance_site_phase("frac_x")
            else:
                self._disable()
        elif self._site_phase == "frac_x":
            if category == "frac_x":
                self._advance_site_phase("frac_y")
            else:
                self._disable()
        elif self._site_phase == "frac_y":
            if category == "frac_y":
                self._advance_site_phase("frac_z")
            else:
                self._disable()
        elif self._site_phase == "frac_z":
            if category == "frac_z":
                self._advance_site_phase("elements")
            else:
                self._disable()
        else:
            self._disable()

    def _advance_site_phase(self, next_phase: str) -> None:
        self._site_phase = next_phase
        self._refresh_site_phase()

    def _refresh_site_phase(self) -> None:
        order = ("elements", "frac_x", "frac_y", "frac_z")
        while self._site_phase in order:
            ids = self._builder.category_ids(self._site_phase)
            if ids:
                return
            if self._site_phase == "elements":
                self._site_phase = "frac_x"
            elif self._site_phase == "frac_x":
                self._site_phase = "frac_y"
            elif self._site_phase == "frac_y":
                self._site_phase = "frac_z"
            elif self._site_phase == "frac_z":
                self._site_phase = "elements"
        if self._site_phase not in order:
            self._site_phase = "elements"

    def _set_sites_phase(self) -> None:
        self._phase = "sites"
        self._site_phase = "elements"
        self._refresh_site_phase()

    def _enter_post_lattice_phase(self) -> None:
        self._set_sites_phase()

    def _advance_lattice_if_empty(self) -> None:
        while self._phase == "lattice" and self._lattice_index < len(self._lattice_sequence):
            category = self._lattice_sequence[self._lattice_index]
            if self._builder.category_ids(category):
                return
            self._lattice_index += 1
        if self._phase == "lattice" and self._lattice_index >= len(self._lattice_sequence):
            self._enter_post_lattice_phase()

    def _disable(self) -> None:
        self._disabled = True

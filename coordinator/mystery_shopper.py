# Copyright 2026 OpenHydra contributors
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

from __future__ import annotations

from dataclasses import dataclass

from coordinator.chain import ChainResult
from verification.auditor import AuditSampler
from verification.redundant import compare_outputs


@dataclass(frozen=True)
class VerificationResult:
    audited: bool
    match: bool
    primary_text: str
    secondary_text: str | None
    tertiary_text: str | None
    winner: str
    mode: str
    sample_rate: float
    auditor_triggered: bool


class MysteryShopper:
    """Tier 1 verification by random re-execution."""

    def __init__(
        self,
        sample_rate: float = 0.1,
        seed: int | None = None,
        mode: str = "mystery_shopper",
        auditor_sample_rate: float = 0.0,
    ):
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.auditor_sample_rate = max(0.0, min(1.0, auditor_sample_rate))
        self._audit_sampler = AuditSampler(sample_rate=self.sample_rate, seed=seed)
        self._auditor_sampler = AuditSampler(
            sample_rate=self.auditor_sample_rate,
            seed=(None if seed is None else seed + 1),
        )
        self.mode = mode

    def should_audit(self) -> bool:
        return self._audit_sampler.should_sample()

    def should_run_auditor_spotcheck(self) -> bool:
        return self._auditor_sampler.should_sample()

    def build_skip_result(self, primary: ChainResult) -> VerificationResult:
        """Return a no-op verification result for single-peer topologies.

        Skips all secondary/tertiary re-execution — no network calls,
        no GPU compute.  The primary result is trusted unconditionally.
        """
        return VerificationResult(
            audited=False,
            match=True,
            primary_text=primary.text,
            secondary_text=None,
            tertiary_text=None,
            winner="primary",
            mode="skipped_single_peer",
            sample_rate=self.sample_rate,
            auditor_triggered=False,
        )

    @staticmethod
    def _same_output(primary: str, candidate: str) -> bool:
        return compare_outputs(primary, candidate).match

    def verify(
        self,
        primary: ChainResult,
        run_secondary: callable,
        run_tertiary: callable | None = None,
    ) -> VerificationResult:
        if not self.should_audit():
            return VerificationResult(
                audited=False,
                match=True,
                primary_text=primary.text,
                secondary_text=None,
                tertiary_text=None,
                winner="primary",
                mode=self.mode,
                sample_rate=self.sample_rate,
                auditor_triggered=False,
            )

        secondary = run_secondary()
        primary_secondary_match = self._same_output(primary.text, secondary.text)

        run_tertiary_now = False
        auditor_triggered = False
        if run_tertiary is not None:
            if primary_secondary_match:
                auditor_triggered = self.should_run_auditor_spotcheck()
                run_tertiary_now = auditor_triggered
            else:
                run_tertiary_now = True

        tertiary = run_tertiary() if run_tertiary_now and run_tertiary is not None else None

        if tertiary is None and primary_secondary_match:
            return VerificationResult(
                audited=True,
                match=True,
                primary_text=primary.text,
                secondary_text=secondary.text,
                tertiary_text=None,
                winner="primary",
                mode=self.mode,
                sample_rate=self.sample_rate,
                auditor_triggered=auditor_triggered,
            )

        if tertiary is None:
            winner = "secondary"
            match = False
            tertiary_text = None
        else:
            primary_text_norm = primary.text.strip()
            secondary_text_norm = secondary.text.strip()
            tertiary_text_norm = tertiary.text.strip()
            labels_by_text: dict[str, list[str]] = {}
            for label, text in (
                ("primary", primary_text_norm),
                ("secondary", secondary_text_norm),
                ("tertiary", tertiary_text_norm),
            ):
                labels_by_text.setdefault(text, []).append(label)

            label_preference = {"primary": 0, "secondary": 1, "tertiary": 2}
            best_count = max(len(labels) for labels in labels_by_text.values())
            winning_texts = [text for text, labels in labels_by_text.items() if len(labels) == best_count]
            winner_text = sorted(
                winning_texts,
                key=lambda text: min(label_preference[label] for label in labels_by_text[text]),
            )[0]
            winner = sorted(labels_by_text[winner_text], key=lambda label: label_preference[label])[0]
            match = len(labels_by_text) == 1
            tertiary_text = tertiary.text

        return VerificationResult(
            audited=True,
            match=match,
            primary_text=primary.text,
            secondary_text=secondary.text,
            tertiary_text=tertiary_text,
            winner=winner,
            mode=self.mode,
            sample_rate=self.sample_rate,
            auditor_triggered=auditor_triggered,
        )

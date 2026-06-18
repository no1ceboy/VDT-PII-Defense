"""
Document Injector — injects prompt injection payloads into clean documents.

Supports multiple injection positions (beginning, middle, end, random)
and returns metadata about where/how the injection was placed.
"""

import random
from dataclasses import dataclass, asdict
from typing import Optional

from .attack_templates import AttackTemplate
from .data_loader import Document


@dataclass
class InjectionResult:
    """Result of injecting an attack template into a document."""
    original_document: str
    poisoned_document: str
    attack_template_id: str
    attack_category: str
    attack_difficulty: str
    injection_position: str          # beginning, middle, end
    injection_char_offset: int       # character position where injection starts
    injection_length: int            # length of the injected payload

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentInjector:
    """Injects attack templates into documents at various positions."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject(
        self,
        document: str,
        template: AttackTemplate,
        position: str = "middle",
    ) -> InjectionResult:
        """
        Inject an attack template into a document.

        Args:
            document: The clean document text.
            template: The attack template to inject.
            position: Where to inject — "beginning", "middle", "end", or "random".

        Returns:
            InjectionResult with the poisoned document and metadata.
        """
        payload = template.template

        if position == "random":
            position = self.rng.choice(["beginning", "middle", "end"])

        if position == "beginning":
            offset = 0
            poisoned = payload + document

        elif position == "end":
            offset = len(document)
            poisoned = document + payload

        elif position == "middle":
            # Find a natural break point (paragraph boundary) near the middle
            offset = self._find_middle_break(document)
            poisoned = document[:offset] + payload + document[offset:]

        else:
            raise ValueError(f"Unknown injection position: {position}")

        return InjectionResult(
            original_document=document,
            poisoned_document=poisoned,
            attack_template_id=template.id,
            attack_category=template.category,
            attack_difficulty=template.difficulty,
            injection_position=position,
            injection_char_offset=offset,
            injection_length=len(payload),
        )

    def _find_middle_break(self, text: str) -> int:
        """Find a paragraph boundary near the middle of the text."""
        mid = len(text) // 2

        # Search for a newline near the midpoint
        search_range = min(500, len(text) // 4)

        best_pos = mid
        best_dist = search_range + 1

        for i in range(max(0, mid - search_range), min(len(text), mid + search_range)):
            if text[i] == "\n":
                dist = abs(i - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = i + 1  # inject after the newline

        return best_pos

    def inject_all_positions(
        self,
        document: str,
        template: AttackTemplate,
    ) -> list[InjectionResult]:
        """Inject a template at all three positions (beginning, middle, end)."""
        results = []
        for pos in ["beginning", "middle", "end"]:
            results.append(self.inject(document, template, pos))
        return results


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from .attack_templates import ALL_TEMPLATES

    sample_doc = (
        "Hải Phòng: Bắt 2 đối tượng dùng chiêu 've sầu thoát xác'.\n"
        "\n"
        "Sau khi gây án, bị công an truy nã, Mái và Quân đã thay tên đổi họ.\n"
        "\n"
        "Họ xuống Hải Phòng làm công nhân và tỏ ra chăm chỉ, hiền lành.\n"
        "\n"
        "Cuối cùng, công an đã bắt gọn 2 đối tượng.\n"
    )

    injector = DocumentInjector(seed=42)
    template = ALL_TEMPLATES[0]

    result = injector.inject(sample_doc, template, position="middle")
    print(f"Template: {template.name}")
    print(f"Position: {result.injection_position}")
    print(f"Offset: {result.injection_char_offset}")
    print(f"\n--- Poisoned Document ---")
    print(result.poisoned_document)

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import yaml

REVISION = "d967c30b053fa40b06c5a0dddf0be493f2dfae46"
VOCABULARY = Path(__file__).with_name("lexicons") / "Vocabulary"
LEXICONS = (
    ("lexicon_covid19", "COVID-19词库"),
    ("lexicon_gfw", "GFW补充词库"),
    ("lexicon_other", "其他词库"),
    ("lexicon_reactionary", "反动词库"),
    ("lexicon_advertising", "广告类型"),
    ("lexicon_politics", "政治类型"),
    ("lexicon_new_thought", "新思想启蒙"),
    ("lexicon_terrorism", "暴恐词库"),
    ("lexicon_livelihood", "民生词库"),
    ("lexicon_weapons", "涉枪涉爆"),
    ("lexicon_netease", "网易前端过滤敏感词库"),
    ("lexicon_sexual_types", "色情类型"),
    ("lexicon_sexual", "色情词库"),
    ("lexicon_supplement", "补充词库"),
    ("lexicon_corruption", "贪腐词库"),
    ("lexicon_tencent", "零时-Tencent"),
    ("lexicon_urls", "非法网址"),
)


class LexiconMatch(NamedTuple):
    start: int
    end: int
    keyword: str


@lru_cache(maxsize=1)
def excluded_terms() -> frozenset[str]:
    return frozenset(
        line.strip().casefold()
        for line in (VOCABULARY.parent / "excluded_terms.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@lru_cache(maxsize=17)
def lexicon_terms(name: str) -> tuple[str, ...]:
    label = next((label for key, label in LEXICONS if key == name), None)
    if label is None:
        raise ValueError(f"Unknown keyword lexicon: {name}")
    return tuple(
        dict.fromkeys(
            term
            for line in (VOCABULARY / f"{label}.txt").read_text(encoding="utf-8-sig").splitlines()
            if (term := line.strip()) and exclusion_reason(term) is None
        )
    )


def exclusion_reason(term: str) -> str | None:
    if term.casefold() in excluded_terms():
        return "reviewed_entry"
    if re.fullmatch(r"[0-9０-９]+", term):
        return "bare_number"
    if re.fullmatch(r"\+?[0-9０-９][0-9０-９ ()（）-]{5,}[0-9０-９]", term):
        return "contact_number"
    if re.search(r"\{(?:MOD|BANNED)\}|\{\d+(?:[,}]|$)|\.\{\d|^\d+\}", term):
        return "legacy_rule_fragment"
    if "=>" in term:
        return "legacy_replacement_rule"
    return None


@lru_cache(maxsize=17)
def lexicon_pattern(name: str) -> Mapping[str, re.Pattern[str]]:
    return MappingProxyType(
        {
            first: re.compile("|".join(re.escape(term) for term in sorted(group, key=len, reverse=True)), re.IGNORECASE)
            for first, group in groupby(
                sorted(lexicon_terms(name), key=lambda term: term[0].lower()), key=lambda term: term[0].lower()
            )
        }
    )


def lexicon_matches(name: str, text: str) -> tuple[LexiconMatch, ...]:
    patterns = lexicon_pattern(name)
    return tuple(
        LexiconMatch(match.start(), match.end(), match[0])
        for position, char in enumerate(text)
        if (pattern := patterns.get(char.lower())) is not None and (match := pattern.match(text, position)) is not None
    )


def available_lexicons() -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "display_name": label,
            "description": "Sensitive-lexicon · " + label,
            "default_action": "BLOCK",
            "match_mode": "literal",
        }
        for name, label in LEXICONS
        if lexicon_terms(name)
    ]


def lexicon_preview(name: str) -> str:
    terms = lexicon_terms(name)
    return yaml.safe_dump(
        {
            "category_name": name,
            "display_name": dict(LEXICONS)[name],
            "match_mode": "literal",
            "source": "https://github.com/konsheng/Sensitive-lexicon/tree/" + REVISION + "/Vocabulary",
            "revision": REVISION,
            "curation_revision": "2026-10-27-r2",
            "keywords": [{"keyword": term} for term in terms],
        },
        allow_unicode=True,
        sort_keys=False,
    )

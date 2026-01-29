import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from dgp.entities import HOUSEHOLD_ITEMS, LETTERS


@dataclass
class Query:
    """Represents a single question-answer template."""

    question: str
    answer_category: str


@dataclass
class Templates:
    """Holds all the template strings for generating task text."""

    definitions: Dict[str, str]
    queries: Dict[str, Query]
    prefix: Optional[str] = None
    capitalize_first_clause: bool = True


@dataclass
class Schema:
    """Represents the blueprint for a binding task using a dataclass for structure."""

    name: str
    items: Dict[str, List[str]]
    unused_items: Dict[str, List[str]]
    templates: Templates
    categories: List[str] = field(init=False)
    max_new_tokens: int = 1
    checker: Callable[[str, str], bool] = (
        lambda neural, causal: causal.lower().strip() in neural.lower().strip()
    )
    matchers: Optional[list] = None

    def __post_init__(self):
        """Derive categories from the items dictionary after initialization."""
        self.categories = list(self.items.keys())


SCHEMA_BOXES = Schema(
    name="boxes",
    items={"Object": HOUSEHOLD_ITEMS, "Box": [x.upper() for x in LETTERS]},
    unused_items={"Object": [], "Box": []},
    templates=Templates(
        prefix="",
        definitions={
            "default": "the {Object} is in Box {Box}",
        },
        queries={
            "Q:Box A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What does Box {Box} contain? Answer:",
                answer_category="Object",
            ),
            "Q:Object A:Box": Query(
                question="Respond in one word, only the answer and nothing else: Which box is the {Object} in? Box",
                answer_category="Box",
            ),
        },
        capitalize_first_clause=True,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal
    in re.search("(Box )?([A-Z])", neural.strip())
    .group(2)
    .strip(),  # Checker for when querying the letters
    # checker=lambda neural, causal: causal.strip().lower() in neural.strip().lower(), # Checker for when querying the items
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_ITEMS)})$", s) is not None,
        lambda s: re.match("^ [A-Z]$", s) is not None,
    ],
)

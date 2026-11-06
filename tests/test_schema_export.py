"""Public JSON output contracts, compared with independently authored references."""

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel
from pydantic.json_schema import JsonSchemaValue

from pydantic_ajv import Rule, RuleModel, rule


def reference(filename: str, case: str | None = None) -> JsonSchemaValue:
    document: object = json.loads(
        (Path(__file__).parent / "fixtures" / "schemas" / filename).read_text()
    )
    assert isinstance(document, dict)
    selected = document if case is None else document[case]
    assert isinstance(selected, dict)
    return selected


@pytest.mark.parametrize(
    ("case", "expression"),
    [
        ("eq", lambda x: x == 2),
        ("ne", lambda x: x != 2),
        ("lt", lambda x: x < 2),
        ("le", lambda x: x <= 2),
        ("gt", lambda x: x > 2),
        ("ge", lambda x: x >= 2),
        ("and", lambda x: (x >= 0) & (x <= 10)),
        ("or", lambda x: (x == 1) | (x == 3)),
        ("membership", lambda x: x.in_([1, 1, 3])),
        ("empty_membership", lambda x: x.in_([])),
        ("large_integer", lambda x: x < 9007199254740993),
    ],
)
def test_root_rules(case: str, expression: Callable[[RuleModel], Rule]) -> None:
    class Value(RootModel[int]):
        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return expression(x.root)

    assert Value.model_json_schema() == reference("root_rules.json", case)


@pytest.mark.parametrize(
    ("case", "expression"),
    [
        ("mixed_membership", lambda x: x.in_([1, True, None, "yes"])),
        ("guard", lambda x: x.when(int | float) > 0),
    ],
)
def test_primitive_union_rules(
    case: str, expression: Callable[[RuleModel], Rule]
) -> None:
    class Value(RootModel[int | float | str | bool | None]):
        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return expression(x.root)

    assert Value.model_json_schema() == reference("root_rules.json", case)


def test_default_bounds() -> None:
    class Ordered(BaseModel):
        low: int = 2
        high: int = 5

        @rule(error="low must precede high")
        @classmethod
        def ordered(cls, x: RuleModel) -> Rule:
            """Bounds must be ordered."""
            return x.low < x.high

    assert Ordered.model_json_schema() == reference("default_bounds.json")


@pytest.mark.parametrize("validated", [False, True])
def test_nested_default_schema(validated: bool) -> None:
    class Child(BaseModel):
        value: int = Field(alias="v", default=3)

    class Parent(BaseModel):
        child: Child = Field(default={}, validate_default=validated)

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.child.value > 3

    # An omitted child is skipped either way: its default is an unvalidated dict
    # or, when validated, isn't evaluated. An omitted value uses its default, 3.
    assert Parent.model_json_schema()["allOf"] == [
        {
            "title": "check",
            "allOf": [
                {
                    "if": {
                        "type": "object",
                        "properties": {
                            "child": {
                                "type": "object",
                                "properties": {"v": True},
                                "required": ["v"],
                            }
                        },
                        "required": ["child"],
                    },
                    "then": {
                        "type": "object",
                        "properties": {
                            "child": {
                                "type": "object",
                                "properties": {
                                    "v": {"type": "number", "exclusiveMinimum": 3}
                                },
                            }
                        },
                    },
                },
                {
                    "if": {
                        "type": "object",
                        "properties": {
                            "child": {"type": "object", "not": {"required": ["v"]}}
                        },
                        "required": ["child"],
                    },
                    "then": False,
                },
            ],
        }
    ]


@pytest.mark.parametrize(("by_alias", "case"), [(True, "by_alias"), (False, "by_name")])
def test_aliases_and_pointer_escaping(by_alias: bool, case: str) -> None:
    class Aliased(BaseModel):
        low: int = Field(alias="a/b~", title="Lower")
        high: int = Field(validation_alias=AliasChoices("upper", "high"), title="Upper")

        @rule
        @classmethod
        def bounds(cls, x: RuleModel) -> Rule:
            return x.high > x.low

    assert Aliased.model_json_schema(by_alias=by_alias) == reference(
        "aliases.json", case
    )


def test_dynamic_membership() -> None:
    class Member(BaseModel):
        value: int
        allowed: list[int]

        @rule
        @classmethod
        def contains(cls, x: RuleModel) -> Rule:
            return x.value.in_(x.allowed)

    assert Member.model_json_schema() == reference("dynamic_membership.json")


def test_fixed_array_indexes() -> None:
    class Increasing(RootModel[list[int]]):
        @rule
        @classmethod
        def order(cls, x: RuleModel) -> Rule:
            return x.root[0] < x.root[1]

    assert Increasing.model_json_schema() == reference("array_indexes.json")


def test_unique_properties_with_aliases() -> None:
    class Item(BaseModel):
        id: str = Field(alias="itemId")

    class Items(RootModel[list[Item]]):
        @rule(error="IDs must be unique")
        @classmethod
        def ids(cls, x: RuleModel) -> Rule:
            return x.root.unique_by("id")

    assert Items.model_json_schema() == reference("unique_properties.json")


def test_nested_model_in_unruled_parent() -> None:
    class Amount(RootModel[int]):
        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.root > 0

    class Parent(BaseModel):
        amount: Amount

    assert Parent.model_json_schema() == reference("nested.json")


def test_recursive_model() -> None:
    class Node(BaseModel):
        value: int
        child: Node | None = None

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    assert Node.model_json_schema() == reference("recursive.json")


def test_empty_key_pointer() -> None:
    class EmptyKey(RootModel[dict[str, int]]):
        @rule
        @classmethod
        def same(cls, x: RuleModel) -> Rule:
            return x.root["left"] == x.root[""]

    assert EmptyKey.model_json_schema() == reference("empty_key.json")


@pytest.mark.parametrize(("case", "expression"), [("ne", lambda x: x["a"] != 1)])
def test_missing_operand_inequality(
    case: str, expression: Callable[[RuleModel], Rule]
) -> None:
    class Missing(RootModel[dict[str, int]]):
        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return expression(x.root)

    assert Missing.model_json_schema() == reference("missing.json", case)


@pytest.mark.parametrize(
    ("case", "options"),
    [
        ("docstring", {}),
        ("explicit", {"description": "Explicit description"}),
        ("empty", {"description": "", "error": ""}),
        ("escaped", {"error": "literal ${value}"}),
    ],
)
def test_annotations(case: str, options: dict[str, str]) -> None:
    class Value(RootModel[int]):
        @rule(description=options.get("description"), error=options.get("error"))
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            """Must be positive.

            Documented separately from the error.
            """
            return x.root > 0

    assert Value.model_json_schema() == reference("annotations.json", case)


def test_serialization_schema_omits_rules() -> None:
    class Value(RootModel[int]):
        @rule(error="positive only")
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.root > 0

    assert Value.model_json_schema(mode="serialization") == {
        "title": "Value",
        "type": "integer",
    }


def test_discriminator_guard_preserves_base_union() -> None:
    class Number(BaseModel):
        kind: Literal["number"]
        value: int = -1

    class Text(BaseModel):
        kind: Literal["text"]
        text: str

    class Choice(BaseModel):
        child: Annotated[Number | Text, Field(discriminator="kind")]

        @rule
        @classmethod
        def positive_number(cls, x: RuleModel) -> Rule:
            return x.child.when(Number).value > 0

    assert Choice.model_json_schema() == reference("discriminator_guard.json")


def test_rules_are_independently_required() -> None:
    class Value(RootModel[int]):
        @rule(error="Must be positive")
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.root > 0

        @rule(error="Must be below ten")
        @classmethod
        def bounded(cls, x: RuleModel) -> Rule:
            return x.root < 10

    assert Value.model_json_schema() == reference("independent_rules.json")


def default_branches(model: type[BaseModel]) -> list[JsonSchemaValue]:
    """The branches a model's only rule adds for omitted fields."""
    fragment = model.model_json_schema()["allOf"][0]
    branches = fragment.get("allOf", [fragment])
    assert all("if" in branch for branch in branches)
    return branches[1:]


def omitted(*names: str) -> JsonSchemaValue:
    required = [{"required": [name]} for name in names]
    return {
        "type": "object",
        "not": required[0] if len(required) == 1 else {"anyOf": required},
    }


def under(key: str, leaf: JsonSchemaValue) -> JsonSchemaValue:
    return {"type": "object", "properties": {key: leaf}, "required": [key]}


def test_defaults_without_a_known_value_are_skipped() -> None:
    class Defaults(BaseModel):
        model_config = ConfigDict(validate_default=True)
        validated: int = 9
        produced: int = Field(default_factory=lambda: 9)
        nested: int = Field(default=9, validation_alias=AliasPath("a", "b"))
        limit: int

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return (
                (x.validated < x.limit) & (x.produced < x.limit) & (x.nested < x.limit)
            )

    # Each omission still skips its check, so Ajv stays more lenient.
    assert "not" not in json.dumps(Defaults.model_json_schema()["allOf"])


def test_omitted_parents_use_their_defaults() -> None:
    class Leaf(BaseModel):
        value: int = Field(alias="v", default=4)

    class Middle(BaseModel):
        leaf: Leaf = Leaf(v=1)

    class Root(BaseModel):
        middle: Middle = Middle()

        @rule
        @classmethod
        def large(cls, x: RuleModel) -> Rule:
            return x.middle.leaf.value > 2

    # middle and leaf default to a value of 1, which fails; v defaults to 4, which
    # passes and needs no branch.
    assert default_branches(Root) == [
        {"if": omitted("middle"), "then": False},
        {"if": under("middle", omitted("leaf")), "then": False},
    ]


def test_static_guards_and_json_values() -> None:
    class Kind(StrEnum):
        A = "a"
        B = "b"

    class Child(BaseModel):
        value: int = 4

    class Example(BaseModel):
        child: Child | None = None
        kind: Kind = Kind.B
        expected: str

        @rule
        @classmethod
        def large(cls, x: RuleModel) -> Rule:
            return x.child.when(Child).value > 5

    class Matching(Example):
        @rule
        @classmethod
        def matching(cls, x: RuleModel) -> Rule:
            return x.kind == x.expected

    # A null child fails the guard, so only an omitted value needs a branch.
    guard = under("child", {"type": "object"})
    assert default_branches(Example) == [
        {"if": {"allOf": [under("child", omitted("value")), guard]}, "then": False}
    ]
    matching = Matching.model_json_schema()["allOf"][1]["allOf"][1]
    assert matching == {
        "if": {
            "allOf": [
                omitted("kind"),
                {
                    "type": "object",
                    "properties": {"expected": True},
                    "required": ["expected"],
                },
            ]
        },
        "then": {"type": "object", "properties": {"expected": {"const": "b"}}},
    }


def present(key: str) -> JsonSchemaValue:
    return {"type": "object", "properties": {key: True}, "required": [key]}


def test_defaults_cover_every_input_name() -> None:
    class Bounds(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        low: int = Field(alias="lo", default=7)
        high: int = Field(
            validation_alias=AliasChoices("upper", AliasPath("high")), default=3
        )

        @rule
        @classmethod
        def ordered(cls, x: RuleModel) -> Rule:
            return x.low < x.high

    # Pydantic reads each field from any of its names, so all must be absent.
    assert default_branches(Bounds) == [
        {
            "if": {"allOf": [omitted("high", "upper"), present("lo")]},
            "then": {
                "type": "object",
                "properties": {"lo": {"type": "number", "exclusiveMaximum": 3}},
            },
        },
        {
            "if": {"allOf": [omitted("lo", "low"), present("upper")]},
            "then": {
                "type": "object",
                "properties": {"upper": {"type": "number", "exclusiveMinimum": 7}},
            },
        },
        {
            "if": {"allOf": [omitted("lo", "low"), omitted("high", "upper")]},
            "then": False,
        },
    ]


def test_default_collections() -> None:
    class Item(BaseModel):
        sku: str

    class Cart(BaseModel):
        items: list[Item] = [Item(sku="a"), Item(sku="a")]

        @rule
        @classmethod
        def unique(cls, x: RuleModel) -> Rule:
            return x.items.unique_by("sku")

    class Tagged(BaseModel):
        tags: list[str] = ["x"]
        primary: str = "y"

        @rule
        @classmethod
        def tagged(cls, x: RuleModel) -> Rule:
            return x.primary.in_(x.tags)

    assert default_branches(Cart) == [{"if": omitted("items"), "then": False}]
    assert default_branches(Tagged) == [
        {
            "if": {"allOf": [omitted("tags"), present("primary")]},
            "then": {"type": "object", "properties": {"primary": {"enum": ["x"]}}},
        },
        {
            "if": {"allOf": [omitted("primary"), present("tags")]},
            "then": {
                "type": "object",
                "properties": {"tags": {"type": "array", "contains": {"const": "y"}}},
            },
        },
        {"if": {"allOf": [omitted("primary"), omitted("tags")]}, "then": False},
    ]

from collections.abc import Callable
from copy import deepcopy
from typing import Annotated, Generic, Literal, TypeVar

import pytest
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    RootModel,
    ValidationError,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from pydantic_ajv import Rule, RuleDefinitionError, RuleModel, rule


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (lambda x: x.values == x.value, "Comparison operands must be JSON primitive"),
        (lambda x: x.value == x.values, "Comparison operands must be JSON primitive"),
        (lambda x: x.value.in_(x.value), "Membership requires an array"),
        (
            lambda x: x.value.in_([[1]]),
            "Membership requires an array of primitive elements",
        ),
        (
            lambda x: x.value.in_(x.nested),
            "Membership requires primitive array elements",
        ),
        (
            lambda x: x.value.in_(x.untyped),
            "Array operations require a declared item type",
        ),
    ],
    ids=[
        "array_subject",
        "array_comparison",
        "scalar_membership",
        "nested_literal",
        "nested_reference",
        "untyped_reference",
    ],
)
def test_invalid_operand_shapes(
    expression: Callable[[RuleModel], Rule], message: str
) -> None:
    with pytest.raises(RuleDefinitionError, match=message):

        class Invalid(BaseModel):
            value: int
            values: list[int]
            nested: list[list[int]]
            untyped: list

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                return expression(x)


def assert_native(
    model: type[BaseModel], payloads: list[object], expected: list[bool]
) -> None:
    original = deepcopy(payloads)
    actual = []
    for payload in payloads:
        try:
            model.model_validate(payload)
            actual.append(True)
        except ValidationError:
            actual.append(False)
    assert actual == expected
    assert payloads == original, "Native validation mutated the caller's input"


def rule_annotations(model: type[BaseModel], rule_id: str) -> JsonSchemaValue:
    """Read public annotations without prescribing the compiler's schema layout."""
    pending: list[object] = [model.model_json_schema()]
    matches = []
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get("title") == rule_id:
                matches.append(node)
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    assert len(matches) == 1, f"Expected one exported annotation for {rule_id!r}"
    return matches[0]


class Ordered(BaseModel):
    low: int = 2
    high: int = 5

    @rule(error="low must precede high")
    @classmethod
    def ordered(cls, x: RuleModel) -> Rule:
        """Bounds must be ordered."""
        return x.low < x.high


def test_defaults_and_errors() -> None:
    assert_native(
        Ordered,
        [{}, {"high": 1}, {"low": 8}, {"low": 3, "high": 4}],
        [True, False, False, True],
    )
    fragment = rule_annotations(Ordered, "ordered")
    assert fragment["description"] == "Bounds must be ordered."
    with pytest.raises(ValidationError) as exc:
        Ordered(high=1)
    assert exc.value.errors()[0]["loc"] == ()
    assert exc.value.errors()[0]["ctx"]["rule_id"] == "ordered"


def test_nested_and_recursive() -> None:
    class Parent(BaseModel):
        child: Ordered

    assert_native(Parent, [{"child": {}}, {"child": {"high": 0}}], [True, False])
    with pytest.raises(ValidationError) as exc:
        Parent(child={"high": 0})
    assert exc.value.errors()[0]["loc"] == ("child",)

    class Node(Ordered):
        child: Node | None = None

    assert_native(Node, [{"child": {}}, {"child": {"high": 0}}], [True, False])


def test_aliases() -> None:
    class Aliased(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        low: int = Field(alias="a/b~", default=2)
        high: int = Field(validation_alias=AliasChoices("upper", "high"))

        @rule
        @classmethod
        def bounds(cls, x: RuleModel) -> Rule:
            return x.low < x.high

    assert_native(Aliased, [{"a/b~": 2, "upper": 3}, {"upper": 0}], [True, False])
    assert_native(Aliased, [{"low": 2, "high": 3}, {"high": 0}], [True, False])


def test_missing_boolean_semantics() -> None:
    class Missing(BaseModel):
        values: dict[str, int]

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.values["a"] != 1

    assert_native(Missing, [{"values": {}}, {"values": {"a": 1}}], [True, False])


def test_membership_types_and_sequences() -> None:
    class Values(BaseModel):
        value: int | str | bool | None
        allowed: list[int | str | bool | None] = [1, "yes", None]
        numbers: list[int]

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.value.in_(x.allowed) & (x.numbers[0] < x.numbers[1])

    assert_native(
        Values,
        [
            {"value": 1, "numbers": [1, 2]},
            {"value": True, "numbers": []},
            {"value": None, "numbers": [2]},
            {"value": "no", "numbers": []},
        ],
        [True, True, True, False],
    )


def test_union_guard() -> None:
    class Number(BaseModel):
        kind: Literal["number"]
        low: int = 3
        high: int = 7

    class Text(BaseModel):
        kind: Literal["text"]
        text: str

    class UnionModel(BaseModel):
        item: Annotated[Number | Text, Field(discriminator="kind")]

        @rule
        @classmethod
        def bounds(cls, x: RuleModel) -> Rule:
            return x.item.when(Number).low < x.item.when(Number).high

    assert_native(
        UnionModel,
        [
            {"item": {"kind": "number"}},
            {"item": {"kind": "number", "high": 1}},
            {"item": {"kind": "text", "text": "x"}},
        ],
        [True, False, True],
    )


def test_primitive_guard() -> None:
    class Guarded(BaseModel):
        value: int | str | None

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value.when(int) > 0

    assert_native(
        Guarded,
        [{"value": "x"}, {"value": None}, {"value": 1}, {"value": 0}],
        [True, True, True, False],
    )


def test_unique_by() -> None:
    class Item(BaseModel):
        id: int = Field(alias="itemId")

    class Items(BaseModel):
        items: list[Item]

        @rule
        @classmethod
        def ids(cls, x: RuleModel) -> Rule:
            return x.items.unique_by("id")

    assert_native(
        Items,
        [
            {"items": []},
            {"items": [{"itemId": 1}, {"itemId": 2}]},
            {"items": [{"itemId": 1}, {"itemId": 1}]},
        ],
        [True, True, False],
    )


def test_root_model() -> None:
    class Increasing(RootModel[list[int]]):
        @rule
        @classmethod
        def order(cls, x: RuleModel) -> Rule:
            return x.root[0] < x.root[1]

    assert_native(Increasing, [[], [1], [1, 2], [2, 1]], [True, True, True, False])


def test_inherited_rule_override_does_not_change_base_behavior() -> None:
    class Base(BaseModel):
        value: int

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    class Sub(Base):
        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 10

    payloads = [{"value": 0}, {"value": 1}, {"value": 10}, {"value": 11}]
    assert_native(Base, payloads, [False, True, True, True])
    assert_native(Sub, payloads, [False, False, False, True])
    assert_native(Base, payloads, [False, True, True, True])


def test_json_validation_enforces_rules() -> None:
    assert Ordered.model_validate_json('{"low": 3, "high": 4}').low == 3
    with pytest.raises(ValidationError, match="low must precede high"):
        Ordered.model_validate_json('{"low": 4, "high": 3}')


def test_docstring_and_literal_error() -> None:
    class Described(BaseModel):
        a: int

        @rule(error="literal ${value}", description="")
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            """Not used."""
            return x.a > 0

    fragment = rule_annotations(Described, "positive")
    assert fragment["description"] == ""
    assert fragment["errorMessage"] == "literal \\$\\{value}"
    with pytest.raises(ValidationError, match=r"literal \$\{value\}"):
        Described(a=0)


def test_definition_errors() -> None:
    with pytest.raises(RuleDefinitionError, match="Numeric"):

        class Unsafe(BaseModel):
            value: int | str

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                return x.value > 0

    with pytest.raises(RuleDefinitionError, match="truth value"):
        bool(RuleModel.for_model(Ordered).low > 0)


def test_parent_defaults_and_null() -> None:
    class Child(BaseModel):
        value: int = 3

    class Parent(BaseModel):
        child: Child = {"value": 4}
        limit: int = 5
        model_config = ConfigDict(validate_default=True)

        @rule
        @classmethod
        def bounds(cls, x: RuleModel) -> Rule:
            return x.child.value < x.limit

    assert_native(
        Parent,
        [{}, {"limit": 4}, {"child": {}, "limit": 4}, {"child": {}, "limit": 3}],
        [True, False, True, False],
    )

    class NullValue(BaseModel):
        value: int | None = 2

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.value == None

    assert_native(NullValue, [{}, {"value": None}], [False, True])


def test_generic_completion() -> None:
    T = TypeVar("T")

    class Box(BaseModel, Generic[T]):
        value: T

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    IntBox = Box[int]
    assert_native(IntBox, [{"value": 2}, {"value": -1}], [True, False])
    assert_native(
        Box[float],
        [{"value": 0.5}, {"value": 0.0}, {"value": -0.5}],
        [True, False, False],
    )
    with pytest.raises(RuleDefinitionError, match="Numeric"):
        Box[str]


def test_cooperative_hooks_and_validator_order() -> None:
    events = []

    class Hooked(BaseModel):
        value: int

        @classmethod
        def __pydantic_on_complete__(cls) -> None:
            super().__pydantic_on_complete__()

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core: CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            schema = super().__get_pydantic_json_schema__(core, handler)
            schema["description"] = "custom hook"
            return schema

        @rule(error="first")
        @classmethod
        def first(cls, x: RuleModel) -> Rule:
            return x.value > 0

        @model_validator(mode="after")
        def record(self) -> Hooked:
            events.append("after")
            return self

        @rule(error="second")
        @classmethod
        def second(cls, x: RuleModel) -> Rule:
            return x.value > 1

    with pytest.raises(ValidationError) as exc:
        Hooked(value=0)
    assert [error["msg"] for error in exc.value.errors()] == ["first"]
    assert events == []
    with pytest.raises(ValidationError) as exc:
        Hooked(value=1)
    assert [error["msg"] for error in exc.value.errors()] == ["second"]
    assert events == ["after"]
    assert Hooked.model_json_schema()["description"] == "custom hook"


def test_composite_errors_are_model_level() -> None:
    class Pair(BaseModel):
        values: dict[str, int]

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return (x.values["a"] > 0) & (x.values["b"] > 0)

    with pytest.raises(ValidationError) as exc:
        Pair(values={"a": -1, "b": -1})
    assert exc.value.errors()[0]["loc"] == ()
    assert len(exc.value.errors()) == 1
    with pytest.raises(ValidationError) as exc:
        Pair(values={"a": -1, "b": 1})
    assert exc.value.errors()[0]["loc"] == ()


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("eq", [False, False, False, True, False]),
        ("ne", [True, True, True, False, True]),
        ("lt", [True, False, False, False, True]),
        ("le", [True, False, False, True, True]),
        ("gt", [False, True, True, False, False]),
        ("ge", [False, True, True, True, False]),
    ],
)
def test_comparison_default_cross_product(op: str, expected: list[bool]) -> None:
    class Comparison(BaseModel):
        a: int = 2
        b: int = 3

        @rule
        @classmethod
        def compare(cls, x: RuleModel) -> Rule:
            return getattr(x.a, f"__{op}__")(x.b)

    payloads = [{}, {"a": 4}, {"b": 1}, {"a": 3, "b": 3}, {"a": -1, "b": 0}]
    assert_native(Comparison, payloads, expected)


@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        ([], [False, False, False, False, False, False]),
        ([1, 1, True, None], [False, True, True, False, True, False]),
        (["a", "b"], [False, False, False, False, False, True]),
    ],
)
def test_literal_membership(choices: list[object], expected: list[bool]) -> None:
    class Member(BaseModel):
        value: int | str | bool | None

        @rule
        @classmethod
        def contains(cls, x: RuleModel) -> Rule:
            return x.value.in_(choices)

    values = [0, 1, True, False, None, "a"]
    assert_native(Member, [{"value": v} for v in values], expected)


def test_duplicate_ids_and_unguarded_unions() -> None:
    with pytest.raises(RuleDefinitionError, match="Duplicate"):

        class Duplicate(Ordered):
            @rule(id="ordered")
            @classmethod
            def another(cls, x: RuleModel) -> Rule:
                return x.low > 0

    with pytest.raises(RuleDefinitionError, match="Union traversal"):

        class Unguarded(BaseModel):
            child: Ordered | None

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                return x.child.low > 0


def test_docstring_omission_and_empty_error() -> None:
    class Empty(BaseModel):
        value: int

        @rule(error="")
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.value > 0

    fragment = rule_annotations(Empty, "check")
    assert "description" not in fragment
    assert fragment["errorMessage"] == ""
    with pytest.raises(ValidationError) as exc:
        Empty(value=0)
    assert exc.value.errors()[0]["msg"] == ""


def test_assignment_and_bypass() -> None:
    class Assignment(Ordered):
        model_config = ConfigDict(validate_assignment=True)

    instance = Assignment()
    with pytest.raises(ValidationError):
        instance.low = 10
    invalid = Assignment.model_construct(low=10, high=1)
    assert invalid.low == 10
    # Pydantic still executes model after-validators on existing instances.
    with pytest.raises(ValidationError):
        Assignment.model_validate(invalid)


def test_initial_forward_completion() -> None:
    class Deferred(BaseModel):
        child: Later
        limit: int

        @rule
        @classmethod
        def bounds(cls, x: RuleModel) -> Rule:
            return x.child.value < x.limit

    class Later(BaseModel):
        value: int

    Deferred.model_rebuild()
    assert_native(
        Deferred,
        [{"child": {"value": 1}, "limit": 2}, {"child": {"value": 2}, "limit": 1}],
        [True, False],
    )


def test_conflicting_union_defaults_and_aliases() -> None:
    class Small(BaseModel):
        kind: Literal["small"]
        value: int = Field(default=2, alias="a/b~")

    class Large(BaseModel):
        kind: Literal["large"]
        value: int = Field(default=10, alias="other")

    class Choice(BaseModel):
        child: Annotated[Small | Large, Field(discriminator="kind")]
        limit: int = 5

        @rule
        @classmethod
        def small(cls, x: RuleModel) -> Rule:
            return x.limit > x.child.when(Small).value

        @rule
        @classmethod
        def large(cls, x: RuleModel) -> Rule:
            return x.limit > x.child.when(Large).value

    assert_native(
        Choice,
        [
            {"child": {"kind": "small"}},
            {"child": {"kind": "large"}},
            {"child": {"kind": "small", "a/b~": 6}},
            {"child": {"kind": "large", "other": 4}},
        ],
        [True, False, False, True],
    )


def test_data_factory_runs_only_during_model_validation() -> None:
    calls = []

    def make_default(data: dict[str, object]) -> int:
        calls.append(1)
        return 3

    class Factory(BaseModel):
        value: int = Field(default_factory=make_default)

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    # A factory reading validated data has no value at export.
    assert calls == []
    Factory.model_json_schema()
    assert calls == []
    assert Factory().value == 3
    assert calls == [1]


def test_export_rejects_omitted_or_colliding_fields() -> None:
    class Omitted(Ordered):
        @classmethod
        def __get_pydantic_json_schema__(
            cls, core: CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            schema = handler(core)
            schema["properties"].pop("low")
            return schema

        # A local rule installs its hook around the custom hook.
        @rule
        @classmethod
        def extra(cls, x: RuleModel) -> Rule:
            return x.low > 0

    with pytest.raises(RuleDefinitionError, match="omitted"):
        Omitted.model_json_schema()

    class Colliding(Ordered):
        low: int = Field(alias="same")
        high: int = Field(alias="same")

    with pytest.raises(RuleDefinitionError, match="colliding"):
        Colliding.model_json_schema()


def test_static_model_and_array_defaults() -> None:
    class Child(BaseModel):
        value: int = Field(alias="v", default=4)

    class Parent(BaseModel):
        child: Child = Child(v=4)
        values: list[int] = [1, 4]

        @rule
        @classmethod
        def allowed(cls, x: RuleModel) -> Rule:
            return x.child.value.in_(x.values)

    assert_native(
        Parent,
        [{}, {"values": [1]}, {"child": {"v": 1}}, {"child": {"v": 3}}],
        [True, False, True, False],
    )


def test_dictionary_unique_and_original_string_ids() -> None:
    class Items(BaseModel):
        fields: list[dict[str, str]]

        @rule
        @classmethod
        def unique(cls, x: RuleModel) -> Rule:
            return x.fields.unique_by("field_id")

    assert_native(
        Items,
        [
            {"fields": [{"field_id": "a"}, {"field_id": "b"}]},
            {"fields": [{"field_id": "a"}, {"field_id": "a"}]},
            {"fields": [{}, {}]},
            {"fields": [{}, {"field_id": "a"}]},
        ],
        [True, False, False, True],
    )


def test_disjunction_suppresses_failed_alternatives() -> None:
    class Either(BaseModel):
        a: int
        b: int

        @rule(error="one must be positive")
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return (x.a > 0) | (x.b > 0)

    assert_native(
        Either,
        [{"a": 1, "b": -1}, {"a": -1, "b": 1}, {"a": -1, "b": -1}],
        [True, True, False],
    )
    with pytest.raises(ValidationError) as exc:
        Either(a=-1, b=-1)
    assert exc.value.errors()[0]["loc"] == ()


def test_empty_key_reference() -> None:
    class EmptyKey(RootModel[dict[str, int]]):
        @rule
        @classmethod
        def same(cls, x: RuleModel) -> Rule:
            return x.root["left"] == x.root[""]

    assert_native(EmptyKey, [{"left": 1, "": 1}, {"left": 1, "": 2}], [True, False])


def test_exact_large_integer_comparison() -> None:
    assert_native(
        Ordered,
        [
            {"low": 9007199254740992, "high": 9007199254740993},
            {"low": 9007199254740993, "high": 9007199254740992},
        ],
        [True, False],
    )


def test_required_operand_with_a_defaulted_field() -> None:
    class Bounds(BaseModel):
        low: int = 10
        high: int

        @rule
        @classmethod
        def ordered(cls, x: RuleModel) -> Rule:
            return x.low <= x.high

    assert_native(
        Bounds,
        [{"high": 5}, {"high": 10}, {"low": 1, "high": 5}, {"low": 9, "high": 5}],
        [False, True, True, False],
    )


def test_missing_dictionary_keys_skip_the_check() -> None:
    class Budget(BaseModel):
        limits: dict[str, int]

        @rule
        @classmethod
        def ordered(cls, x: RuleModel) -> Rule:
            return x.limits["low"] <= x.limits["high"]

    assert_native(
        Budget,
        [
            {"limits": {"low": 1, "high": 5}},
            {"limits": {"low": 9, "high": 5}},
            {"limits": {"high": 5}},
            {"limits": {"low": 9}},
            {"limits": {}},
        ],
        [True, False, True, True, True],
    )

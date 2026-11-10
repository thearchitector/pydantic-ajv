"""Independent acceptance cases for the RuleModel design agreed with the user."""

from collections.abc import Callable, Iterator
from inspect import getattr_static
from types import UnionType
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, GetJsonSchemaHandler, RootModel, ValidationError
from pydantic.json_schema import JsonSchemaValue

import pydantic_ajv
from pydantic_ajv import Node, Rule, RuleDefinitionError, RuleModel, rule


def objects(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


@pytest.mark.parametrize(
    "expression",
    [
        lambda x: x.value > 0,
        lambda x: x.unknown > 0,
        lambda x: x.values["bad"] > 0,
        lambda x: x.value.when(bytes) == b"bad",
        lambda x: x.mapping.when(dict[str, int])["key"] > 0,
    ],
    ids=[
        "mixed_order",
        "unknown_field",
        "invalid_index",
        "impossible_type",
        "generic_target",
    ],
)
def test_bad_expressions_raise_before_the_factory_returns(
    expression: Callable[[RuleModel], Rule],
) -> None:
    events: list[str] = []

    with pytest.raises(RuleDefinitionError):

        class Invalid(BaseModel):
            value: int | str
            values: list[int]
            mapping: dict[str, int]

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                events.append("entered")
                result = expression(x)
                events.append("constructed")
                return result

    assert events == ["entered"]


def test_compatible_numeric_union_needs_no_narrowing() -> None:
    class Numbers(BaseModel):
        value: int | float

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    assert Numbers(value=1)
    assert Numbers(value=0.5)
    for value in [0, -0.5]:
        with pytest.raises(ValidationError):
            Numbers(value=value)


@pytest.mark.parametrize(
    ("target", "value", "valid"),
    [
        (int, 0, False),
        (int, 0.0, True),
        (int, False, True),
        (int, "bad", True),
        (float, 0, True),
        (float, 0.0, False),
        (float, False, True),
        (int | float, 0, False),
        (int | float, 0.0, False),
        (int | float, False, True),
        (int | float, "bad", True),
    ],
)
def test_python_narrowing_uses_exact_types(
    target: type | UnionType, value: object, valid: bool
) -> None:
    class Example(BaseModel):
        value: int | float | str | bool

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value.when(target) > 0

    if valid:
        assert Example(value=value)
    else:
        with pytest.raises(ValidationError) as error:
            Example(value=value)
        detail = error.value.errors()[0]
        assert detail["loc"] == ()
        assert detail["ctx"]["rule_id"] == "positive"
        assert "paths" not in detail["ctx"]


def test_both_operand_guards_must_match_before_comparison() -> None:
    class Example(BaseModel):
        low: int | str
        high: int | str

        @rule
        @classmethod
        def ordered(cls, x: RuleModel) -> Rule:
            return x.low.when(int) < x.high.when(int)

    for low, high in [("bad", 0), (0, "bad"), ("bad", "bad"), (0, 1)]:
        assert Example(low=low, high=high)
    with pytest.raises(ValidationError):
        Example(low=2, high=1)


@pytest.mark.parametrize("conjunction", [True, False])
def test_guard_skip_remains_local_under_boolean_composition(conjunction: bool) -> None:
    class Example(BaseModel):
        value: int | str
        limit: int

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            guarded = x.value.when(int) > 0
            independent = x.limit > 0
            return guarded & independent if conjunction else guarded | independent

    if conjunction:
        with pytest.raises(ValidationError):
            Example(value="skip", limit=-1)
    else:
        assert Example(value="skip", limit=-1)
    with pytest.raises(ValidationError):
        Example(value=-1, limit=-1)


def test_even_shared_model_fields_require_narrowing_before_access() -> None:
    class Foo(BaseModel):
        kind: Literal["foo"]
        count: int

    class Bar(BaseModel):
        kind: Literal["bar"]
        count: int

    reached: list[str] = []
    with pytest.raises(RuleDefinitionError):

        class Invalid(BaseModel):
            model: Annotated[Foo | Bar, Field(discriminator="kind")]

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                reached.append("before")
                count = x.model.count
                reached.append("after")
                return count > 0

    assert reached == ["before"]


def test_unused_untagged_union_is_allowed_but_narrowing_it_is_not() -> None:
    class Foo(BaseModel):
        count: int

    class Bar(BaseModel):
        count: str

    class Allowed(BaseModel):
        model: Foo | Bar
        value: int

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.value > 0

    assert Allowed(model=Foo(count=0), value=1)
    assert Allowed.model_json_schema()
    with pytest.raises(RuleDefinitionError):

        class Invalid(BaseModel):
            model: Foo | Bar

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                return x.model.when(Foo).count > 0


def test_exact_model_guard_skips_subclasses_and_other_branches() -> None:
    class Foo(BaseModel):
        kind: Literal["foo"]
        count: int

    class SubFoo(Foo):
        pass

    class Bar(BaseModel):
        kind: Literal["bar"]
        text: str

    class Example(BaseModel):
        model: Annotated[Foo | Bar, Field(discriminator="kind")]

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.model.when(Foo).count > 0

    with pytest.raises(ValidationError):
        Example(model=Foo(kind="foo", count=0))
    assert Example(model=SubFoo(kind="foo", count=0))
    assert Example(model=Bar(kind="bar", text="no count attribute"))


def test_nullable_model_narrowing_does_not_require_discriminator() -> None:
    class Child(BaseModel):
        value: int

    class Parent(BaseModel):
        child: Child | None = None

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.child.when(Child).value > 0

    assert Parent()
    assert Parent(child=Child(value=1))
    with pytest.raises(ValidationError):
        Parent(child=Child(value=0))
    assert Parent.model_json_schema()


def test_nullable_root_model_guard_matches_its_json_root_representation() -> None:
    class Amount(RootModel[int]):
        pass

    class Parent(BaseModel):
        amount: Amount | None = None

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.amount.when(Amount).root > 0

    assert Parent()
    assert Parent(amount=Amount(1))
    with pytest.raises(ValidationError):
        Parent(amount=Amount(0))
    fragment = Parent.model_json_schema()["allOf"]
    conditions = [obj["if"] for obj in objects(fragment) if "if" in obj]
    assert any(
        obj.get("type") == "integer"
        for condition in conditions
        for obj in objects(condition)
    )


def test_model_fields_named_like_reference_helpers_remain_usable() -> None:
    class Child(BaseModel):
        path: int
        data: int
        resolve: int

    class Parent(BaseModel):
        child: Child

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            selected = x.child.when(Child)
            return (selected.path > 0) & (selected.data > 0) & (selected.resolve > 0)

    assert Parent(child=Child(path=1, data=1, resolve=1))
    with pytest.raises(ValidationError):
        Parent(child=Child(path=0, data=1, resolve=1))
    assert Parent.model_json_schema()


def test_tag_guards_use_exported_aliases_and_preserve_original_union() -> None:
    class Foo(BaseModel):
        kind: Literal["foo"] = Field(alias="type")
        count: int = Field(alias="fooCount")

    class Bar(BaseModel):
        kind: Literal["bar"] = Field(alias="type")
        count: str = Field(alias="barCount")

    class Plain(BaseModel):
        model: Annotated[Foo | Bar, Field(discriminator="kind", alias="branch")]

    class Checked(Plain):
        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return x.model.when(Foo).count > 0

    for by_alias in [True, False, True]:
        baseline = Plain.model_json_schema(by_alias=by_alias)
        schema = Checked.model_json_schema(by_alias=by_alias)
        assert schema["properties"] == baseline["properties"]
        assert schema["$defs"] == baseline["$defs"]
        tag = "type" if by_alias else "kind"
        conditions = [obj["if"] for obj in objects(schema["allOf"]) if "if" in obj]
        constraints = [
            obj["properties"][tag]
            for condition in conditions
            for obj in objects(condition)
            if isinstance(obj.get("properties"), dict) and tag in obj["properties"]
        ]
        assert any(
            c.get("const") == "foo" or c.get("enum") == ["foo"] for c in constraints
        )
        assert Plain.model_json_schema(by_alias=by_alias) == baseline


def test_container_union_indexing_requires_narrowing_even_for_shared_index() -> None:
    with pytest.raises(RuleDefinitionError):

        class Invalid(BaseModel):
            values: list[int] | tuple[int, ...]

            @rule
            @classmethod
            def check(cls, x: RuleModel) -> Rule:
                return x.values[0] > 0

    class Allowed(BaseModel):
        values: list[int] | dict[str, int]

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return (x.values.when(list)[0] > 0) & (x.values.when(dict)["price"] > 0)

    for values in [[], {}, [1], {"price": 1}]:
        assert Allowed(values=values)
    for values in [[0], {"price": 0}]:
        with pytest.raises(ValidationError):
            Allowed(values=values)


def test_python_operators_keep_native_boolean_number_equality() -> None:
    class Example(BaseModel):
        value: bool | int

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return (x.value == 1) & x.value.in_([1])

    assert Example(value=True)
    with pytest.raises(ValidationError):
        Example(value=False)


def test_default_factory_is_called_by_dumping_not_construction() -> None:
    calls: list[str] = []

    def default() -> int:
        calls.append("called")
        return 0

    class Example(BaseModel):
        value: int = Field(default_factory=default)

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.value > 0

    assert calls == []
    # The omitted value is checked against what the factory produces, 0.
    branches = Example.model_json_schema()["allOf"][0]["allOf"]
    assert branches[1] == {
        "if": {"type": "object", "not": {"required": ["value"]}},
        "then": False,
    }
    assert calls == ["called"]
    with pytest.raises(ValidationError):
        Example()
    assert calls == ["called", "called"]


def test_each_model_dumps_its_own_defaults() -> None:
    class Example(BaseModel):
        low: int = 2
        high: int = 5

        @rule
        @classmethod
        def check(cls, x: RuleModel) -> Rule:
            return x.low < x.high

    class DifferentDefaults(Example):
        low: int = 9
        high: int = 0

    def omitted_branch(model: type[BaseModel]) -> JsonSchemaValue:
        branches = model.model_json_schema()["allOf"][0]["allOf"]
        both_omitted = (
            {"type": "object", "not": {"required": ["high"]}},
            {"type": "object", "not": {"required": ["low"]}},
        )
        matches = [
            branch
            for branch in branches
            if all(part in branch["if"].get("allOf", []) for part in both_omitted)
        ]
        assert len(matches) <= 1
        return matches[0] if matches else {}

    assert Example()
    with pytest.raises(ValidationError):
        DifferentDefaults()
    # 2 < 5 needs no branch; 9 < 0 rejects a document that omits both.
    assert omitted_branch(Example) == {}
    assert omitted_branch(DifferentDefaults)["then"] is False


def test_rule_factory_receives_a_model_specific_context_and_resolve_returns_bool() -> (
    None
):
    contexts: list[RuleModel] = []
    expressions: list[Rule] = []

    class Example(RootModel[int]):
        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            contexts.append(x)
            expression = x.root > 0
            expressions.append(expression)
            return expression

    assert len(contexts) == 1
    assert isinstance(contexts[0], RuleModel)
    assert type(contexts[0]) is not RuleModel
    assert expressions[0].resolve(Example.model_construct(root=1)) is True
    assert expressions[0].resolve(Example.model_construct(root=0)) is False
    assert Example.model_json_schema()["type"] == "integer"
    for removed in ["compile_python", "compile_json", "bind", "implies", "__invert__"]:
        assert not hasattr(expressions[0], removed)
    for removed in [
        "is_type",
        "is_number",
        "is_integer",
        "is_string",
        "is_boolean",
        "is_null",
    ]:
        assert getattr_static(contexts[0].root, removed, None) is None
    assert not hasattr(pydantic_ajv, "RuleContext")


def test_tagged_unions_inside_collections_keep_their_discriminator_metadata() -> None:
    class Foo(BaseModel):
        kind: Literal["foo"]
        count: int

    class Bar(BaseModel):
        kind: Literal["bar"]
        text: str

    Item = Annotated[Foo | Bar, Field(discriminator="kind")]

    class Example(BaseModel):
        items: list[Item]
        records: dict[str, Item]

        @rule
        @classmethod
        def positive(cls, x: RuleModel) -> Rule:
            return (x.items[0].when(Foo).count > 0) & (
                x.records["primary"].when(Foo).count > 0
            )

    assert Example(items=[], records={})
    assert Example(items=[Bar(kind="bar", text="skip")], records={})
    with pytest.raises(ValidationError):
        Example(items=[Foo(kind="foo", count=0)], records={})
    with pytest.raises(ValidationError):
        Example(items=[], records={"primary": Foo(kind="foo", count=0)})
    assert Example.model_json_schema()


def test_a_new_node_needs_only_resolve_and_dump() -> None:
    class Reject(Node):
        def resolve(self, instance: BaseModel) -> bool:
            return False

        def dump(
            self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
        ) -> JsonSchemaValue:
            return {"not": {}}

    class Example(BaseModel):
        @rule(error="rejected")
        @classmethod
        def reject(cls, x: RuleModel) -> Rule:
            return Rule(Reject())

    with pytest.raises(ValidationError) as error:
        Example()
    assert error.value.errors()[0]["loc"] == ()
    assert error.value.errors()[0]["msg"] == "rejected"
    assert Example.model_json_schema()["allOf"] == [
        {"not": {}, "title": "reject", "errorMessage": "rejected"}
    ]

"""Direct construction establishes the same contracts as symbolic authoring."""

from collections.abc import Callable
from decimal import Decimal
from enum import IntEnum
from inspect import signature

import pytest
from pydantic import BaseModel, Field, GetJsonSchemaHandler, ValidationError
from pydantic.json_schema import JsonSchemaValue

from pydantic_ajv import RuleDefinitionError, RuleModel, nodes, references
from pydantic_ajv.nodes import (
    Equality,
    EqualityOperator,
    Logical,
    LogicalOperator,
    Membership,
    Node,
    Ordering,
    OrderingOperator,
    Uniqueness,
)
from pydantic_ajv.references import (
    ContainerStep,
    Data,
    LiteralValue,
    ModelStep,
    Narrowing,
    Projection,
    Reference,
    TypeShape,
)
from pydantic_ajv.validation import definition


class Example(BaseModel):
    value: int
    other: int
    choices: list[int]
    items: list[dict[str, int]]


def reference(annotation: object, name: str = "value") -> Reference[object]:
    return Reference(TypeShape(annotation), (ContainerStep(name),))


class Tag(IntEnum):
    ONE = 1


@pytest.mark.parametrize(
    "value",
    [
        b"x",
        Decimal(1),
        Tag.ONE,
        float("inf"),
        float("nan"),
        [float("inf")],
        ([1],),
        {"key": 1},
    ],
)
def test_direct_literals_reject_non_json_values(value: object) -> None:
    with pytest.raises(RuleDefinitionError, match="Unsupported rule literal"):
        LiteralValue(value)


@pytest.mark.parametrize(
    "construct",
    [
        lambda: Equality(EqualityOperator.EQ, reference(list[int]), LiteralValue(1)),
        lambda: Ordering(OrderingOperator.LT, reference(int | str), LiteralValue(1)),
        lambda: Ordering(OrderingOperator.LT, reference(int), LiteralValue(True)),
        lambda: Ordering(OrderingOperator.LT, reference(int), reference(str)),
        lambda: Ordering(EqualityOperator.EQ, reference(int), LiteralValue(1)),
        lambda: Equality(EqualityOperator.EQ, LiteralValue(1), LiteralValue(2)),
        lambda: Membership(reference(list[int]), LiteralValue((1,))),
        lambda: Membership(reference(int), LiteralValue(1)),
        lambda: Membership(reference(int), reference(list[list[int]])),
        lambda: Membership(reference(int), reference(list)),
        lambda: Uniqueness(reference(int), Projection(ContainerStep("id"))),
        lambda: Uniqueness(
            reference(list[dict[str, list[int]]]), Projection(ContainerStep("id"))
        ),
        lambda: Projection(ContainerStep(0)),
        lambda: Projection("id"),
        lambda: Reference("int"),
        lambda: Reference(TypeShape(int), ("value",)),
        lambda: Reference(TypeShape(int), (ContainerStep(-1),)),
        lambda: Reference(TypeShape(int), (ContainerStep(True),)),
        lambda: Reference(TypeShape(int), (), (object(),)),
        lambda: Reference(TypeShape(int), (ModelStep("value", Example, Field()),)),
    ],
    ids=[
        "array_equality",
        "numeric_union",
        "numeric_boolean",
        "numeric_reference",
        "operator_enum",
        "literal_comparison",
        "array_subject",
        "scalar_collection",
        "nested_items",
        "untyped_items",
        "scalar_unique",
        "object_property",
        "numeric_projection",
        "invalid_projection",
        "invalid_shape",
        "invalid_step",
        "negative_index",
        "boolean_index",
        "invalid_guard",
        "foreign_field",
    ],
)
def test_direct_construction_rejects_invalid_operands(
    construct: Callable[[], object],
) -> None:
    with pytest.raises(RuleDefinitionError) as error:
        construct()
    assert not isinstance(error.value, ValidationError)


def test_direct_literal_collection_is_a_snapshot() -> None:
    source = [9007199254740993, True, 1.5, None, "1"]
    literal = LiteralValue(source)
    member = Membership(reference(int), literal)
    source.append(99)
    assert literal.value == [9007199254740993, True, 1.5, None, "1"]
    assert [type(value) for value in literal.value] == [
        int,
        bool,
        float,
        type(None),
        str,
    ]
    assert not member.resolve(Example(value=99, other=0, choices=[], items=[]))


def test_authoring_preserves_symbolic_identity() -> None:
    x = RuleModel.for_model(Example)
    left, right, choices = x.value, x.other, x.choices
    comparison = (left < right).node
    membership = left.in_(choices).node
    assert isinstance(comparison, Ordering)
    assert comparison.left is left
    assert comparison.right is right
    assert isinstance(membership, Membership)
    assert membership.subject is left
    assert membership.collection is choices


class Handler(GetJsonSchemaHandler):
    mode = "validation"

    def __call__(self, schema: object) -> JsonSchemaValue:
        raise AssertionError("These references use container keys")

    def resolve_ref_schema(self, schema: JsonSchemaValue) -> JsonSchemaValue:
        return schema


@pytest.mark.parametrize(
    ("operator", "keyword", "expected"),
    [
        (OrderingOperator.LT, "exclusiveMinimum", True),
        (OrderingOperator.LE, "minimum", True),
        (OrderingOperator.GT, "exclusiveMaximum", False),
        (OrderingOperator.GE, "maximum", False),
    ],
)
def test_direct_reversed_ordering(
    operator: OrderingOperator, keyword: str, expected: bool
) -> None:
    expression = Ordering(operator, LiteralValue(1), reference(int))
    instance = Example(value=2, other=3, choices=[], items=[])
    assert expression.resolve(instance) is expected
    left, right = expression.values(Handler(), Example.model_json_schema())
    assert left == Data(("value",))
    assert right == 1
    assert expression.constraint(left, right) == {
        "type": "object",
        "properties": {"value": {"type": "number", keyword: 1}},
    }


def test_direct_membership_export_variants() -> None:
    handler, schema = Handler(), Example.model_json_schema()
    dynamic = Membership(reference(int), reference(list[int], "choices"))
    contains = Membership(LiteralValue(2), reference(list[int], "choices"))
    empty = Membership(reference(int), LiteralValue(()))
    assert dynamic.dump(handler, schema)["then"]["properties"]["value"] == {
        "enum": {"$data": "1/choices"}
    }
    assert contains.dump(handler, schema)["then"]["properties"]["choices"] == {
        "type": "array",
        "contains": {"const": 2},
    }
    assert empty.dump(handler, schema)["then"] is False


def test_uniqueness_rejects_another_models_projection() -> None:
    class Item(BaseModel):
        id: int

    class Other(BaseModel):
        id: int

    collection = reference(list[Item], "items")
    projection = Projection.for_collection(reference(list[Other]), "id")
    with pytest.raises(RuleDefinitionError, match="belong to its collection"):
        Uniqueness(collection, projection)


def test_reference_shape_must_match_field_metadata() -> None:
    step = ModelStep("choices", Example, Example.model_fields["choices"])
    with pytest.raises(RuleDefinitionError, match="shape does not match"):
        Reference(TypeShape(int), (step,))


@pytest.mark.parametrize("targets", [(), (str,), (list[int],)])
def test_direct_narrowing_validates_targets(targets: tuple[object, ...]) -> None:
    with pytest.raises(RuleDefinitionError):
        Narrowing(reference(int), targets)


@pytest.mark.parametrize("operator", [LogicalOperator.AND, LogicalOperator.OR])
def test_direct_logical_nodes_preserve_short_circuiting(
    operator: LogicalOperator,
) -> None:
    class Constant(Node):
        def resolve(self, instance: BaseModel) -> bool:
            return operator is LogicalOperator.OR

        def dump(
            self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
        ) -> JsonSchemaValue:
            return {}

    class Unreachable(Constant):
        def resolve(self, instance: BaseModel) -> bool:
            raise AssertionError("The right node must be skipped")

    left, right = Constant(), Unreachable()
    expression = Logical(operator, left, right)
    assert expression.left is left
    assert expression.right is right
    assert expression.resolve(Example(value=1, other=1, choices=[], items=[])) is (
        operator is LogicalOperator.OR
    )
    with pytest.raises(RuleDefinitionError):
        Logical(operator, left, LiteralValue(True))


@pytest.mark.parametrize("operator", [EqualityOperator.EQ, EqualityOperator.NE])
def test_direct_reversed_equality(operator: EqualityOperator) -> None:
    expression = Equality(operator, LiteralValue(1), reference(int))
    assert expression.resolve(Example(value=1, other=1, choices=[], items=[])) is (
        operator is EqualityOperator.EQ
    )
    leaf = {"const": 1} if operator is EqualityOperator.EQ else {"not": {"const": 1}}
    assert expression.dump(Handler(), Example.model_json_schema())["then"] == {
        "type": "object",
        "properties": {"value": leaf},
    }


def test_resolution_and_predicates_do_not_validate_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = RuleModel.for_model(Example)
    expressions = [
        x.value < x.other,
        x.value == x.other,
        x.value.in_(x.choices),
        x.items.unique_by("id"),
    ]
    instance = Example(value=1, other=1, choices=[1], items=[{"id": 1}, {"id": 2}])

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Construction validation ran during evaluation")

    for module, names in [
        (
            references,
            [
                "literal_contents",
                "reference_metadata",
                "uniqueness_inputs",
                "projection_metadata",
            ],
        ),
        (
            nodes,
            [
                "equality_inputs",
                "ordering_inputs",
                "membership_inputs",
                "uniqueness_inputs",
            ],
        ),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, fail)
    for _ in range(2):
        assert [expression.resolve(instance) for expression in expressions] == [
            False,
            True,
            True,
            True,
        ]
    del instance.value
    assert all(expression.resolve(instance) for expression in expressions)


def test_definition_decorator_preserves_signatures_and_body_errors() -> None:
    def accept(value: int, *, other: int = 1) -> int:
        return value + other

    checked = definition("invalid input")(accept)
    assert signature(checked) == signature(accept)
    assert checked(2, other=3) == 5
    with pytest.raises(RuleDefinitionError, match="invalid input"):
        checked("2")

    @definition("argument error")
    def invalid_body(value: int) -> None:
        Example.model_validate({})

    with pytest.raises(ValidationError):
        invalid_body(1)


def test_uniqueness_rebuilds_only_over_a_reference() -> None:
    items = reference(list[dict[str, int]], "items")
    unique = Uniqueness(items, Projection.for_collection(items, "id"))
    assert unique.rebuild(items, LiteralValue("id")) == unique
    with pytest.raises(RuleDefinitionError, match="model reference"):
        unique.rebuild(LiteralValue([]), LiteralValue("id"))

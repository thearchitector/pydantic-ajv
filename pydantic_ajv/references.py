"""Typed model references, structural traversal, and local narrowing guards."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import reduce
from operator import or_
from types import UnionType
from typing import Annotated, Any, Literal, TypeGuard, Union, cast, get_args, get_origin

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    GetJsonSchemaHandler,
    InstanceOf,
    ValidateAs,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticSerializationError,
    PydanticUndefined,
    PydanticUndefinedType,
    SchemaValidator,
    ValidationError,
    core_schema,
    to_jsonable_python,
)

from .validation import (
    JsonLiteral,
    Number,
    PathKey,
    Primitive,
    PropertyName,
    RuleDefinitionError,
    definition,
)

__all__ = [
    "MISSING",
    "NUMERIC",
    "SCALARS",
    "ArrayOperand",
    "ContainerStep",
    "Data",
    "Default",
    "JsonKind",
    "LiteralCollection",
    "LiteralValue",
    "ModelStep",
    "Narrowing",
    "Number",
    "NumericOperand",
    "ObjectCollection",
    "Operand",
    "Path",
    "Primitive",
    "PrimitiveArray",
    "PrimitiveCollection",
    "Projection",
    "Reference",
    "RuleDefinitionError",
    "ScalarOperand",
    "Schema",
    "SchemaScope",
    "Step",
    "TypeShape",
    "absent",
    "at",
    "conjunction",
    "is_model",
    "lookup",
    "require_shape",
    "uniqueness_inputs",
]

type Path = tuple[str | int, ...]
type PrimitiveCollection = tuple[Primitive, ...]
type LiteralCollection = list[Primitive] | PrimitiveCollection
type ObjectCollection = Collection[BaseModel | Mapping[str, Primitive]]
type PrimitiveArray = Collection[Primitive]
type Schema = JsonSchemaValue | bool

MISSING = PydanticUndefined


class JsonKind(StrEnum):
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    NULL = "null"
    ARRAY = "array"
    OBJECT = "object"


SCALARS = frozenset({
    JsonKind.INTEGER,
    JsonKind.NUMBER,
    JsonKind.STRING,
    JsonKind.BOOLEAN,
    JsonKind.NULL,
})
NUMERIC = frozenset({JsonKind.INTEGER, JsonKind.NUMBER})


def is_model(annotation: object) -> TypeGuard[type[BaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


@dataclass(frozen=True, slots=True)
class TypeShape:
    """The declared alternatives of one reference, known before operators run."""

    annotation: object
    discriminator: str | None = None

    def __post_init__(self) -> None:
        annotation = self.annotation
        while get_origin(annotation) is Annotated:
            annotation, *metadata = get_args(annotation)
            for item in metadata:
                if isinstance(item, FieldInfo) and isinstance(item.discriminator, str):
                    object.__setattr__(self, "discriminator", item.discriminator)
        object.__setattr__(self, "annotation", annotation)

    @property
    def alternatives(self) -> tuple[object, ...]:
        if get_origin(self.annotation) in (Union, UnionType):
            return get_args(self.annotation)
        return (self.annotation,)

    @property
    def kinds(self) -> frozenset[JsonKind]:
        kinds: set[JsonKind] = set()
        for alternative in self.alternatives:
            if get_origin(alternative) is Annotated:
                kinds.update(TypeShape(alternative).kinds)
            elif get_origin(alternative) is Literal:
                for value in get_args(alternative):
                    kinds.update(TypeShape(type(value)).kinds)
            elif alternative is Any:
                kinds.update(JsonKind)
            else:
                typ = get_origin(alternative) or alternative
                if typ is type(None):
                    kinds.add(JsonKind.NULL)
                elif typ is bool:
                    kinds.add(JsonKind.BOOLEAN)
                elif isinstance(typ, type) and issubclass(typ, int):
                    kinds.add(JsonKind.INTEGER)
                elif isinstance(typ, type) and issubclass(typ, float):
                    kinds.add(JsonKind.NUMBER)
                elif isinstance(typ, type) and issubclass(typ, str):
                    kinds.add(JsonKind.STRING)
                elif typ in (list, tuple, set, frozenset, Sequence):
                    kinds.add(JsonKind.ARRAY)
                elif is_model(typ) or typ in (dict, Mapping):
                    kinds.add(JsonKind.OBJECT)
                else:
                    raise RuleDefinitionError(
                        f"Unsupported referenced type: {alternative!r}"
                    )
        return frozenset(kinds)

    def concrete(self) -> object:
        if len(self.alternatives) != 1:
            raise RuleDefinitionError("Union traversal requires narrowing with .when()")
        return self.annotation

    def select(self, target: object) -> tuple[TypeShape, tuple[type, ...]]:
        targets = TypeShape(target).alternatives
        if not targets:
            raise RuleDefinitionError(
                ".when() requires runtime types, not parameterized targets"
            )
        selected: list[object] = []
        checked: list[type] = []
        for target_type in targets:
            if not isinstance(target_type, type) or target_type is Any:
                raise RuleDefinitionError(
                    ".when() requires runtime types, not parameterized targets"
                )
            matching: list[object] = []
            for alternative in self.alternatives:
                alternative = TypeShape(alternative).annotation
                if alternative is Any:
                    matching.append(target_type)
                elif get_origin(alternative) is Literal:
                    values = tuple(
                        v for v in get_args(alternative) if type(v) is target_type
                    )
                    if values:
                        matching.append(Literal[values])
                elif (get_origin(alternative) or alternative) is target_type:
                    matching.append(alternative)
            if not matching:
                raise RuleDefinitionError(
                    f"Narrowing target {target_type!r} cannot match {self.annotation!r}"
                )
            selected.extend(t for t in matching if t not in selected)
            checked.append(target_type)
        annotation = reduce(or_, selected)
        return TypeShape(annotation, self.discriminator), tuple(checked)

    def array_items(self) -> tuple[TypeShape, ...]:
        require_shape(
            self, frozenset({JsonKind.ARRAY}), "Membership requires an array operand"
        )
        items: list[TypeShape] = []
        for alternative in self.alternatives:
            args = get_args(TypeShape(alternative).annotation)
            if not args:
                raise RuleDefinitionError(
                    "Array operations require a declared item type"
                )
            items.extend(TypeShape(arg) for arg in args if arg is not Ellipsis)
        return tuple(items)


@dataclass(frozen=True, slots=True)
class ContainerStep:
    key: str | int

    def exported_key(self, handler: GetJsonSchemaHandler) -> str | int:
        return self.key


@dataclass(frozen=True, slots=True)
class ModelStep:
    key: str
    model: type[BaseModel]
    field: FieldInfo

    def exported_key(self, handler: GetJsonSchemaHandler) -> str | None:
        if self.model.__pydantic_root_model__ and self.key == "root":
            return None
        names: dict[str, str] = {}
        for name, info in self.model.model_fields.items():
            alias = info.validation_alias
            validation_alias = (
                alias.convert_to_aliases()
                if alias is not None and not isinstance(alias, str)
                else alias
            )
            one = handler(
                core_schema.typed_dict_schema({
                    name: core_schema.typed_dict_field(
                        core_schema.any_schema(), validation_alias=validation_alias
                    )
                })
            )
            names[name] = next(iter(one["properties"]))
        key = names[self.key]
        if list(names.values()).count(key) != 1:
            raise RuleDefinitionError(
                "Referenced model field has colliding exported aliases"
            )
        return key

    def default(self) -> object:
        """The value an instance holds when the field is omitted, or ``MISSING``.

        As in Pydantic, a factory is called and a validated default is passed
        through the field's own validator. A factory that reads validated data,
        a validated default whose validators take ``ValidationInfo`` (its data
        holds other inputs), or a default that fails validation has no value
        known at export, so an omitted field with any of them still skips the
        check.
        """
        info = self.field
        if info.is_required() or info.default_factory_takes_validated_data:
            return MISSING
        value = info.get_default(call_default_factory=True)
        validated = (
            info.validate_default
            if info.validate_default is not None
            else self.model.model_config.get("validate_default", False)
        )
        if not validated:
            return value
        validator = field_validator(self.model, self.key)
        if validator is None:
            return MISSING
        try:
            return validator.validate_python(value)
        except ValidationError:
            return MISSING

    def input_paths(self) -> tuple[Path, ...] | None:
        """Every location Pydantic may read this field from, or None."""
        # An empty string is a valid alias, so only None means there is none.
        alias = self.field.validation_alias
        if alias is None:
            alias = self.field.alias
        if alias is None:
            return ((self.key,),)
        config = self.model.model_config
        paths: set[Path] = (
            {(self.key,)}
            if config.get("validate_by_name") or config.get("populate_by_name")
            else set()
        )
        if config.get("validate_by_alias", True):
            for choice in alias.choices if isinstance(alias, AliasChoices) else [alias]:
                path = (choice,) if isinstance(choice, str) else tuple(choice.path)
                # A negative index counts from the end, which JSON Schema can't address.
                if any(type(key) is int and key < 0 for key in path):
                    return None
                paths.add(path)
        return tuple(
            sorted(paths, key=lambda path: [(type(k) is int, str(k)) for k in path])
        )


def field_validator(model: type[BaseModel], key: str) -> SchemaValidator | None:
    """A validator for one model field alone, without the model's own validators.

    None when the field can't be validated alone, including when one of its
    validators takes ``ValidationInfo``, whose data only a model provides.
    """
    core: Mapping[str, Any] = model.__pydantic_core_schema__
    definitions: list[CoreSchema] = []
    if core["type"] == "definitions":
        definitions = core["definitions"]
        core = core["schema"]
    refs = {definition.get("ref"): definition for definition in definitions}
    while core["type"] != "model":
        if core["type"] == "definition-ref" and core["schema_ref"] in refs:
            core = refs[core["schema_ref"]]
        elif "schema" in core:
            core = core["schema"]
        else:
            return None
    if core["cls"] is not model:
        return None
    fields = core["schema"]
    while fields["type"] != "model-fields" and "schema" in fields:
        fields = fields["schema"]
    if fields["type"] != "model-fields" or key not in fields["fields"]:
        return None
    inner = fields["fields"][key]["schema"]
    if inner["type"] == "default":
        inner = inner["schema"]
    if reads_validation_info(inner, refs, set()):
        return None
    if definitions:
        inner = core_schema.definitions_schema(inner, definitions)
    return SchemaValidator(inner, core.get("config"))


def reads_validation_info(
    schema: object, refs: Mapping[object, CoreSchema], seen: set[str]
) -> bool:
    """Whether a field schema runs a validator that takes ``ValidationInfo``.

    A nested model validates with its own data, so the search stops there.
    """
    if isinstance(schema, list):
        return any(reads_validation_info(item, refs, seen) for item in schema)
    if not isinstance(schema, Mapping):
        return False
    kind = schema.get("type")
    if kind == "model":
        return False
    if kind == "with-info":
        return True
    if kind == "definition-ref":
        ref = schema["schema_ref"]
        if ref in seen or ref not in refs:
            return False
        seen.add(ref)
        return reads_validation_info(refs[ref], refs, seen)
    return any(reads_validation_info(value, refs, seen) for value in schema.values())


type Step = ContainerStep | ModelStep


def lookup(value: object, path: Path) -> object:
    for key in path:
        if isinstance(value, BaseModel) and isinstance(key, str):
            value = getattr(value, key, MISSING)
        elif isinstance(value, Mapping):
            value = value.get(key, MISSING)
        elif (
            isinstance(value, (list, tuple))
            and type(key) is int
            and 0 <= key < len(value)
        ):
            value = value[key]
        else:
            return MISSING
    return value


def absent(paths: tuple[Path, ...]) -> Schema:
    """No path resolves in an object, as when Pydantic finds no input for a field."""
    found: list[Schema] = []
    for first, *rest in paths:
        resolved: JsonSchemaValue = {"required": [first]}
        if rest:
            resolved["properties"] = {first: at(tuple(rest), True, required=True)}
        found.append(resolved)
    return {"type": "object", "not": found[0] if len(found) == 1 else {"anyOf": found}}


def conjunction(parts: list[Schema]) -> Schema:
    filtered = [part for part in parts if part is not True]
    return (
        True
        if not filtered
        else filtered[0]
        if len(filtered) == 1
        else {"allOf": filtered}
    )


def at(path: Path, leaf: Schema, *, required: bool = False) -> Schema:
    for key in reversed(path):
        if isinstance(key, str):
            leaf = {
                "type": "object",
                "properties": {key: leaf},
                **({"required": [key]} if required else {}),
            }
        else:
            leaf = {
                "type": "array",
                "prefixItems": [True] * key + [leaf],
                **({"minItems": key + 1} if required else {}),
            }
    return leaf


@dataclass(frozen=True, slots=True)
class Data:
    """One emitted JSON location, with relative $data addressing."""

    path: Path

    def constrain(self, leaf: Schema) -> Schema:
        return at(self.path, leaf)

    def present(self) -> Schema:
        return at(self.path, True, required=True)

    def relative_to(self, subject: Data) -> JsonSchemaValue:
        pointer = "/".join(
            str(key).replace("~", "~0").replace("/", "~1") for key in self.path
        )
        return {"$data": str(len(subject.path)) + ("/" + pointer if self.path else "")}


class SchemaScope:
    """Checks emitted field availability without mutating the base schema."""

    def __init__(self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue) -> None:
        self.handler = handler
        self.schema = schema

    def alternatives(self, schema: Schema) -> list[JsonSchemaValue]:
        if isinstance(schema, bool):
            return []
        schema = self.handler.resolve_ref_schema(schema)
        result = [schema]
        for keyword in ("anyOf", "oneOf", "allOf"):
            for alternative in schema.get(keyword, []):
                result.extend(self.alternatives(alternative))
        return result

    def verify(self, steps: tuple[Step, ...], path: Path) -> None:
        nodes: list[Schema] = [self.schema]
        for step, key in zip(steps, path, strict=True):
            children: list[Schema] = []
            for node in nodes:
                for alternative in self.alternatives(node):
                    if isinstance(key, int):
                        prefix = alternative.get("prefixItems", [])
                        children.append(
                            prefix[key]
                            if key < len(prefix)
                            else alternative.get("items", True)
                        )
                    elif key in alternative.get("properties", {}):
                        children.append(alternative["properties"][key])
                    elif isinstance(step, ContainerStep):
                        children.append(alternative.get("additionalProperties", True))
            if not children:
                raise RuleDefinitionError(
                    f"Referenced field {step.key!r} is omitted from the emitted schema"
                )
            nodes = children


@dataclass(frozen=True, slots=True)
class Default[T]:
    """An operand's value when one model field on its path is omitted."""

    condition: Schema
    value: T


@dataclass(frozen=True, slots=True)
class Narrowing:
    """An explicit local guard attached before its predicate is constructed."""

    reference: Reference[object]
    targets: tuple[type, ...]

    def __post_init__(self) -> None:
        narrowing_inputs(self.reference, self.targets)

    def matches(self, instance: BaseModel) -> bool:
        return type(lookup(instance, self.reference._path)) in self.targets

    def dump(self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue) -> Schema:
        location = self.reference._data(handler, schema)
        alternatives: list[Schema] = []
        for target in self.targets:
            discriminator = self.reference._shape.discriminator
            if is_model(target) and discriminator is not None:
                info = target.model_fields[discriminator]
                key = ModelStep(discriminator, target, info).exported_key(handler)
                assert key is not None
                values = get_args(TypeShape(info.annotation).annotation)
                leaf = (
                    {"const": values[0]} if len(values) == 1 else {"enum": list(values)}
                )
                alternatives.append(at((*location.path, key), leaf, required=True))
            else:
                guard_shape = TypeShape(target)
                while (
                    is_model(guard_shape.annotation)
                    and guard_shape.annotation.__pydantic_root_model__
                ):
                    guard_shape = TypeShape(
                        guard_shape.annotation.model_fields["root"].annotation
                    )
                kinds = guard_shape.kinds
                json_type = next(iter(kinds)) if len(kinds) == 1 else sorted(kinds)
                alternatives.append(
                    at(location.path, {"type": json_type}, required=True)
                )
        return alternatives[0] if len(alternatives) == 1 else {"anyOf": alternatives}


@dataclass(frozen=True, slots=True, eq=False)
class Reference[T]:
    """A reference already carrying its model fields, types, and local guards."""

    _shape: TypeShape
    _steps: tuple[Step, ...] = ()
    _guards: tuple[Narrowing, ...] = ()

    def __post_init__(self) -> None:
        reference_metadata(self._shape, self._steps, self._guards)

    @property
    def _path(self) -> Path:
        return tuple(step.key for step in self._steps)

    def _child(self, key: str | int) -> tuple[TypeShape, tuple[Step, ...]]:
        annotation = self._shape.concrete()
        if is_model(annotation):
            if not isinstance(key, str) or key not in annotation.model_fields:
                raise RuleDefinitionError(f"Unknown model field {key!r}")
            field = annotation.model_fields[key]
            step: Step = ModelStep(key, annotation, field)
            shape = TypeShape(
                field.annotation,
                field.discriminator if isinstance(field.discriminator, str) else None,
            )
        else:
            origin, args = get_origin(annotation), get_args(annotation)
            if (origin or annotation) in (dict, Mapping):
                if not isinstance(key, str) or not args or args[0] is not str:
                    raise RuleDefinitionError(
                        "Dictionary paths require typed string keys"
                    )
                shape = TypeShape(args[1])
            elif (origin or annotation) in (list, tuple, Sequence):
                if type(key) is not int or key < 0 or not args:
                    raise RuleDefinitionError(
                        "Sequence paths require a nonnegative fixed index and an element type"
                    )
                if origin is tuple and args[-1] is not Ellipsis:
                    if key >= len(args):
                        raise RuleDefinitionError(
                            "Tuple index is outside its declared shape"
                        )
                    shape = TypeShape(args[key])
                else:
                    shape = TypeShape(args[0])
            else:
                raise RuleDefinitionError(f"Cannot traverse {annotation!r}")
            step = ContainerStep(key)
        return shape, (*self._steps, step)

    def _resolve(self, instance: BaseModel) -> T | PydanticUndefinedType:
        if not all(guard.matches(instance) for guard in self._guards):
            return MISSING
        # Model lookup is dynamic; construction has checked the declared shape.
        return cast(T | PydanticUndefinedType, lookup(instance, self._path))

    def _data(self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue) -> Data:
        keys = [(step, step.exported_key(handler)) for step in self._steps]
        emitted_steps = tuple(step for step, key in keys if key is not None)
        path = tuple(key for _, key in keys if key is not None)
        SchemaScope(handler, schema).verify(emitted_steps, path)
        return Data(path)

    def _condition(
        self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
    ) -> Schema:
        return conjunction([
            self._data(handler, schema).present(),
            *(guard.dump(handler, schema) for guard in self._guards),
        ])

    def _defaults(
        self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
    ) -> tuple[Default[T], ...]:
        """One case per omitted model field on the path whose default Pydantic uses."""
        path = self._data(handler, schema).path
        defaults: list[Default[T]] = []
        emitted = 0
        for index, step in enumerate(self._steps):
            if step.exported_key(handler) is None:
                continue
            if isinstance(step, ModelStep):
                default = self._default(handler, schema, index, path[:emitted], step)
                if default is not None:
                    defaults.append(default)
            emitted += 1
        return tuple(defaults)

    def _default(
        self,
        handler: GetJsonSchemaHandler,
        schema: JsonSchemaValue,
        index: int,
        parent: Path,
        step: ModelStep,
    ) -> Default[T] | None:
        paths = step.input_paths()
        if paths is None:
            return None
        value = step.default()
        if value is MISSING:
            return None
        conditions = [at(parent, absent(paths), required=True)]
        for guard in self._guards:
            guarded = guard.reference._path
            if len(guarded) <= index:
                conditions.append(guard.dump(handler, schema))
            elif type(lookup(value, guarded[index + 1 :])) not in guard.targets:
                return None
        value = lookup(value, self._path[index + 1 :])
        if value is MISSING:
            return None
        # A default Ajv can't compare against, such as a sentinel object(), has
        # no JSON value, so like Pydantic's own export it is left out.
        try:
            to_jsonable_python(value)
        except PydanticSerializationError:
            return None
        # As in _resolve, the default's type is dynamic but matches the declared shape.
        return Default(conjunction(conditions), cast(T, value))


@dataclass(frozen=True, slots=True)
class LiteralValue[T]:
    value: T

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", literal_contents(self.value))

    def _resolve(self, instance: BaseModel) -> T:
        return self.value

    def _data(self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue) -> T:
        return self.value

    def _condition(
        self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
    ) -> Schema:
        return True

    def _defaults(
        self, handler: GetJsonSchemaHandler, schema: JsonSchemaValue
    ) -> tuple[Default[T], ...]:
        return ()


type Operand[T] = LiteralValue[T] | Reference[T]


@dataclass(frozen=True, slots=True)
class Projection:
    step: Step
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", projection_metadata(self.step))

    @classmethod
    def for_collection(cls, collection: Reference[object], key: str) -> Projection:
        items = collection._shape.array_items()
        if len(items) != 1:
            raise RuleDefinitionError(
                "unique_by requires one concrete object item type"
            )
        item = Reference[object](items[0])
        shape, steps = item._child(key)
        require_shape(shape, SCALARS, "unique_by requires primitive property values")
        return cls(steps[0])

    def dump(self, handler: GetJsonSchemaHandler) -> str:
        key = self.step.exported_key(handler)
        if not isinstance(key, str):
            raise RuleDefinitionError("unique_by requires a direct object property")
        return key


def require_shape(shape: TypeShape, allowed: frozenset[JsonKind], message: str) -> None:
    if not shape.kinds <= allowed:
        raise RuleDefinitionError(message)


@definition(".when() requires runtime types, not parameterized targets")
def narrowing_inputs(
    reference: InstanceOf[Reference[object]], targets: tuple[type, ...]
) -> None:
    if not targets:
        raise RuleDefinitionError(
            ".when() requires runtime types, not parameterized targets"
        )
    reference._shape.select(reduce(or_, targets))
    models = [
        TypeShape(t).annotation
        for t in reference._shape.alternatives
        if is_model(TypeShape(t).annotation)
    ]
    if (
        len(models) > 1
        and any(is_model(t) for t in targets)
        and reference._shape.discriminator is None
    ):
        raise RuleDefinitionError(
            "Narrowing multiple model alternatives requires an explicit discriminator"
        )


@definition("Invalid reference metadata")
def reference_metadata(
    shape: InstanceOf[TypeShape],
    steps: tuple[InstanceOf[ContainerStep] | InstanceOf[ModelStep], ...],
    guards: tuple[InstanceOf[Narrowing], ...],
) -> None:
    for step in steps:
        path_key(step.key)
        if isinstance(step, ModelStep):
            if not is_model(step.model) or step.key not in step.model.model_fields:
                raise RuleDefinitionError("Invalid model field metadata")
            if step.field is not step.model.model_fields[step.key]:
                raise RuleDefinitionError("Invalid model field metadata")
    if steps and isinstance(steps[-1], ModelStep):
        terminal = steps[-1]
        expected = TypeShape(
            terminal.field.annotation,
            terminal.field.discriminator
            if isinstance(terminal.field.discriminator, str)
            else None,
        )
        path = tuple(step.key for step in steps)
        for guard in guards:
            if guard.reference._path == path:
                expected, _ = expected.select(reduce(or_, guard.targets))
        if shape != expected:
            raise RuleDefinitionError("Reference shape does not match its model field")


@definition("Paths require string keys or nonnegative indexes")
def path_key(key: PathKey) -> None:
    pass


@definition("Unsupported rule literal: {value!r}")
def literal_contents(
    value: Annotated[
        object,
        ValidateAs(
            JsonLiteral | list[JsonLiteral] | tuple[JsonLiteral, ...], lambda v: v
        ),
    ],
) -> object:
    return value


@dataclass(frozen=True, slots=True)
class OperandShape:
    allowed: frozenset[JsonKind]
    message: str

    def __call__[T](self, operand: Operand[T]) -> Operand[T]:
        shape = (
            operand._shape
            if isinstance(operand, Reference)
            else TypeShape(type(operand.value))
        )
        require_shape(shape, self.allowed, self.message)
        return operand


def primitive_array_operand[T](operand: Operand[T]) -> Operand[T]:
    if isinstance(operand, Reference):
        for item in operand._shape.array_items():
            require_shape(item, SCALARS, "Membership requires primitive array elements")
    elif not isinstance(operand.value, (list, tuple)):
        raise RuleDefinitionError("Membership requires an array operand")
    return operand


type ScalarOperand = Annotated[
    InstanceOf[Reference[Primitive]] | InstanceOf[LiteralValue[Primitive]],
    AfterValidator(
        OperandShape(SCALARS, "Comparison operands must be JSON primitives")
    ),
]
type MembershipSubject = Annotated[
    InstanceOf[Reference[Primitive]] | InstanceOf[LiteralValue[Primitive]],
    AfterValidator(
        OperandShape(SCALARS, "Membership subjects must be JSON primitives")
    ),
]
type NumericOperand = Annotated[
    InstanceOf[Reference[Number]] | InstanceOf[LiteralValue[Number]],
    AfterValidator(
        OperandShape(
            NUMERIC,
            "Numeric comparisons require numeric operands; narrow unions with .when()",
        )
    ),
]
type ArrayOperand = Annotated[
    InstanceOf[Reference[PrimitiveArray]] | InstanceOf[LiteralValue[LiteralCollection]],
    AfterValidator(primitive_array_operand),
]


@definition("unique_by requires a string property name")
def uniqueness_inputs(
    collection: InstanceOf[Reference[ObjectCollection]],
    projection: InstanceOf[Projection],
) -> None:
    expected = Projection.for_collection(collection, projection.key)
    if projection != expected:
        raise RuleDefinitionError("unique_by projection must belong to its collection")


@definition("unique_by requires a string property name")
def projection_key(key: PropertyName) -> None:
    pass


@definition("unique_by requires a direct object property")
def projection_metadata(step: InstanceOf[ContainerStep] | InstanceOf[ModelStep]) -> str:
    key = step.key
    if not isinstance(key, str):
        raise RuleDefinitionError("unique_by requires a direct object property")
    projection_key(key)
    return key

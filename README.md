# pydantic-ajv

![PyPI Downloads](https://img.shields.io/pypi/dm/pydantic-ajv?style=flat)
![Made with AI](https://img.shields.io/badge/%E2%9C%A8-Made_with_AI-8A2BE2?style=flat)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/thearchitector/pydantic-ajv/ci.yaml?style=flat)

Add composable validation rules to Pydantic models and export them as Ajv-compatible JSON Schema.

It supports:

- comparing fields with other fields or literal values
- checking membership in a fixed set or another field's values
- combining rules with AND and OR at any depth
- accessing nested models, dictionary keys, and array items
- applying rules to selected types and discriminated union variants
- requiring unique properties across a collection of objects
- custom validation messages, rule IDs, and descriptions
- Pydantic aliases, defaults, inherited rules, and `RootModel`

Requires Python 3.14+ and Pydantic `>=2.12,<3`.

## Quick start

Place `@rule` above `@classmethod`, return a rule expression, and validate with
Pydantic as usual:

```python
from pydantic import BaseModel, ValidationError
from pydantic_ajv import Rule, RuleModel, rule


class Article(BaseModel):
    tags: list[str]
    primary_tag: str

    @rule(error="primary_tag must occur in tags")
    @classmethod
    def primary_tag_exists(cls, x: RuleModel) -> Rule:
        return x.primary_tag.in_(x.tags)


article = Article(tags=["python", "validation"], primary_tag="python")
schema = Article.model_json_schema()

try:
    Article(tags=["python"], primary_tag="rust")
except ValidationError as exc:
    error = exc.errors()[0]
    assert error["msg"] == "primary_tag must occur in tags"
    assert error["ctx"]["rule_id"] == "primary_tag_exists"
```

### Available operations

In a rule method, `x` refers to the model's fields:

| Operation        | Syntax                 | Example                                   |
| ---------------- | ---------------------- | ----------------------------------------- |
| Equality         | `==`, `!=`             | `x.password == x.confirmation`            |
| Numeric ordering | `<`, `<=`, `>`, `>=`   | `x.low <= x.high`                         |
| Fixed membership | `.in_(list_or_tuple)`  | `x.status.in_(["draft", "published"])`    |
| Field membership | `.in_(field)`          | `x.primary_tag.in_(x.tags)`               |
| AND              | `&`                    | `(x.low >= 0) & (x.low <= x.high)`        |
| OR               | `\|`                   | `(x.status == "draft") \| (x.score > 0)`  |
| Nested field     | `.field`               | `x.address.country == "US"`               |
| Dictionary key   | `["key"]`              | `x.prices["sale"] <= x.prices["regular"]` |
| Array item       | `[index]`              | `x.values[0] < x.values[1]`               |
| Type condition   | `.when(type_or_union)` | `x.value.when(int \| float) > 0`          |
| Unique property  | `.unique_by("field")`  | `x.items.unique_by("sku")`                |

Equality and membership accept JSON primitives: strings, numbers, booleans, and
`None`. Ordering accepts numbers. Parenthesize comparisons when combining rules;
use `&` and `|`, not Python `and`, `or`, or chained comparisons.

## Compare fields and compose rules

Combine field comparisons, literal comparisons, and membership checks:

```python
from pydantic import BaseModel
from pydantic_ajv import Rule, RuleModel, rule


class Publication(BaseModel):
    status: str
    score: int
    minimum_score: int = 10

    @rule(error="Use a valid status and meet the publishing score")
    @classmethod
    def publishable(cls, x: RuleModel) -> Rule:
        return x.status.in_(["draft", "published"]) & (
            (x.status == "draft") | (x.score >= x.minimum_score)
        )


Publication(status="draft", score=0)
Publication(status="published", score=12)
```

Add multiple decorated methods to require multiple rules. Subclasses inherit
rules; a method with the same name replaces the inherited rule.

## Validate nested values

Access model fields, literal dictionary keys, and fixed nonnegative array indexes
in the same expression:

```python
from pydantic import BaseModel, Field
from pydantic_ajv import Rule, RuleModel, rule


class Budget(BaseModel):
    limit: int


class Project(BaseModel):
    budget: Budget
    costs: dict[str, int]
    milestones: list[int] = Field(min_length=2)

    @rule(error="Costs and milestones must fit the project limits")
    @classmethod
    def within_limits(cls, x: RuleModel) -> Rule:
        return (x.costs["total"] <= x.budget.limit) & (
            x.milestones[0] < x.milestones[1]
        )


Project(budget=Budget(limit=100), costs={"total": 80}, milestones=[1, 3])
```

A missing dictionary key or array index skips that individual check. Use
Pydantic fields and collection constraints to require values.

An omitted model field with a default is not missing: Pydantic checks the
default, and the exported schema does too, so `{}` fails both for a model with
`low: int = 9`, `high: int = 0` and the rule `x.low < x.high`. Defaults the
schema can't know at export time still skip the check when omitted: a
`default_factory`, a default Pydantic validates (`validate_default=True`), and
a field read through a nested `AliasPath`. For model fields
whose names collide with helpers, use brackets, such as `x["when"]`.

## Apply rules conditionally

Use `.when()` to check only matching types. Values of other types pass that
individual check:

```python
from pydantic import BaseModel
from pydantic_ajv import Rule, RuleModel, rule


class Measurement(BaseModel):
    value: int | float | str

    @rule(error="Numeric measurements must be positive")
    @classmethod
    def positive_number(cls, x: RuleModel) -> Rule:
        return x.value.when(int | float) > 0


Measurement(value=2.5)
Measurement(value="pending")
```

### Discriminated unions

Select a model variant before accessing its fields. Declare a discriminator for
unions containing multiple model types:

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ajv import Rule, RuleModel, rule


class Delivery(BaseModel):
    kind: Literal["delivery"]
    distance: int


class Pickup(BaseModel):
    kind: Literal["pickup"]
    store: str


class Order(BaseModel):
    fulfillment: Delivery | Pickup = Field(discriminator="kind")
    delivery_radius: int = 20

    @rule(error="Delivery must be within the service area")
    @classmethod
    def in_service_area(cls, x: RuleModel) -> Rule:
        return x.fulfillment.when(Delivery).distance <= x.delivery_radius


Order(fulfillment=Delivery(kind="delivery", distance=12))
Order(fulfillment=Pickup(kind="pickup", store="downtown"))
```

For nullable models or containers, select the non-null type before traversing:
`x.budget.when(Budget).limit > 0` or `x.costs.when(dict)["total"] > 0`.
Use runtime types such as `dict` as targets.

## Require unique object properties

Use `.unique_by()` with a direct primitive property of each model or dictionary
in a collection:

```python
from pydantic import BaseModel
from pydantic_ajv import Rule, RuleModel, rule


class Item(BaseModel):
    sku: str
    quantity: int


class Cart(BaseModel):
    items: list[Item]

    @rule(error="Each SKU must appear only once")
    @classmethod
    def unique_skus(cls, x: RuleModel) -> Rule:
        return x.items.unique_by("sku")


Cart(items=[Item(sku="book", quantity=2), Item(sku="pen", quantity=5)])
```

Ajv consumers must enable
[`uniqueItemProperties`](https://ajv.js.org/packages/ajv-keywords.html#uniqueitemproperties)
from `ajv-keywords`; see [Validate with Ajv](#validate-with-ajv).

## Customize errors and descriptions

Set an explicit rule ID, description, and validation message:

```python
from pydantic import BaseModel, ValidationError
from pydantic_ajv import Rule, RuleModel, rule


class Bounds(BaseModel):
    low: int
    high: int

    @rule(
        id="ordered_bounds",
        description="The lower bound must not exceed the upper bound.",
        error="low must be less than or equal to high",
    )
    @classmethod
    def ordered(cls, x: RuleModel) -> Rule:
        return x.low <= x.high


try:
    Bounds(low=10, high=5)
except ValidationError as exc:
    error = exc.errors()[0]
    assert error["loc"] == ()
    assert error["ctx"]["rule_id"] == "ordered_bounds"
    assert error["msg"] == "low must be less than or equal to high"
```

All three options are optional. The ID defaults to the method name, the
description to its docstring, and the error to `Rule '<id>' failed`. IDs must be
unique within a model. Custom messages are also available in Ajv with
[`ajv-errors`](https://ajv.js.org/packages/ajv-errors.html).

## Use field aliases

Write rules with Python field names. Export with aliases or Python names using
`model_json_schema(by_alias=...)`:

```python
from pydantic import BaseModel, Field
from pydantic_ajv import Rule, RuleModel, rule


class Window(BaseModel):
    start: int = Field(alias="startTime")
    end: int = Field(alias="endTime")

    @rule
    @classmethod
    def ordered(cls, x: RuleModel) -> Rule:
        return x.start <= x.end


Window(startTime=1, endTime=3)
schema = Window.model_json_schema(by_alias=True)
assert "startTime" in schema["properties"]
```

## Validate root models

Use `x.root` for `RootModel` values:

```python
from pydantic import RootModel
from pydantic_ajv import Rule, RuleModel, rule


class Interval(RootModel[tuple[int, int]]):
    @rule(error="The start must precede the end")
    @classmethod
    def ordered(cls, x: RuleModel) -> Rule:
        return x.root[0] < x.root[1]


Interval.model_validate([1, 5])
schema = Interval.model_json_schema()
```

## Evaluate a rule directly

Create a rule for an existing model and check instances with `.resolve()`:

```python
from pydantic import BaseModel
from pydantic_ajv import RuleModel


class Range(BaseModel):
    low: int
    high: int


x = RuleModel.for_model(Range)
ordered = x.low <= x.high

assert ordered.resolve(Range(low=1, high=3))
assert not ordered.resolve(Range(low=3, high=1))
```

## Validate with Ajv

Use Ajv, with support for custom errors and unique object properties:

```bash
pnpm add ajv@8 ajv-errors ajv-keywords
```

Validate the `Article` schema exported by `Article.model_json_schema()` in the
[quick start](#quick-start):

```js
const Ajv2020 = require("ajv/dist/2020");
const addErrors = require("ajv-errors");
const addKeywords = require("ajv-keywords");

const schema = {
    allOf: [
        {
            errorMessage: "primary_tag must occur in tags",
            if: {
                allOf: [
                    {
                        properties: { primary_tag: true },
                        required: ["primary_tag"],
                        type: "object",
                    },
                    {
                        properties: { tags: true },
                        required: ["tags"],
                        type: "object",
                    },
                ],
            },
            then: {
                properties: { primary_tag: { enum: { $data: "1/tags" } } },
                type: "object",
            },
            title: "primary_tag_exists",
        },
    ],
    properties: {
        tags: { items: { type: "string" }, title: "Tags", type: "array" },
        primary_tag: { title: "Primary Tag", type: "string" },
    },
    required: ["tags", "primary_tag"],
    title: "Article",
    type: "object",
};

const ajv = new Ajv2020({
    $data: true,
    allErrors: true,
    strict: false,
});
addErrors(ajv);
addKeywords(ajv, ["uniqueItemProperties"]);

const validate = ajv.compile(schema);

console.log(validate({ tags: ["python"], primary_tag: "python" })); // true
console.log(validate({ tags: ["python"], primary_tag: "rust" })); // false
console.log(validate.errors);
```

Use the 2020-12 entry point with
[`$data: true`](https://ajv.js.org/guide/combining-schemas.html#data-reference)
and [`strict: false`](https://ajv.js.org/strict-mode.html).

## License

[BSD 3-Clause Clear](LICENSE).

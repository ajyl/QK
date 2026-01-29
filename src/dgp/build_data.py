import random
from dgp.schemas import Schema, SCHEMA_BOXES


def fill_schema(
    schema: Schema,
    num_instances: int,
    query_cat_id: int,
    answer_cat_id: int,
    query_same_item: bool = False,
    query_index_vals: list[int] | None = None,
    query_from_unused: bool = False,
):
    """
    A template (meant to be used with partial) for sampling an answerable question from a schema. It samples a random instance for each category, and then sets the query to a random instance.

    Args:
        schema: The schema to sample an answerable question from.
        num_instances: The number of instances (or tuples) to sample.
        query_cat_id: The index of the categories used for querying.
        answer_cat_id: The index of the answer category, which we query.
    """
    input_obj = {}
    vals_per_cat = {}
    for cat_id in range(len(schema.categories)):
        items = schema.items[schema.categories[cat_id]]
        if query_same_item and cat_id == query_cat_id:
            query_object = items[0]

        vals_per_cat[cat_id] = random.sample(items, num_instances)
        for i, val in enumerate(vals_per_cat[cat_id]):
            input_obj[f"Object.{cat_id}.{i}"] = val

    for i in range(num_instances):
        input_obj[f"Object.0.Ordinal.{i}"] = i

    if query_index_vals is None:
        query_index = random.randint(0, num_instances - 1)
    else:
        query_index = random.choice(query_index_vals)

    if query_same_item:
        orig_index = vals_per_cat[query_cat_id].index(query_object)
        # swap items
        input_obj[f"Object.{query_cat_id}.{orig_index}"] = input_obj[
            f"Object.{query_cat_id}.{query_index}"
        ]
        input_obj[f"Object.{query_cat_id}.{query_index}"] = query_object

    for cat_id in range(len(schema.categories)):
        if cat_id == query_cat_id:
            input_obj[f"Object.{cat_id}.Query"] = input_obj[
                f"Object.{cat_id}.{query_index}"
            ]
        else:
            input_obj[f"Object.{cat_id}.Query"] = None

    input_obj["queryIndex"] = query_index
    input_obj["queryCategoryIndex"] = query_cat_id
    input_obj["queryCategory"] = schema.categories[query_cat_id]
    input_obj["answerCategoryIndex"] = answer_cat_id
    input_obj["answerCategory"] = schema.categories[answer_cat_id]
    input_obj["answer"] = input_obj[f"Object.{answer_cat_id}.{query_index}"]
    input_obj["numInstances"] = num_instances
    input_obj["unused_items"] = {
        schema.categories[cat_id]: schema.unused_items[schema.categories[cat_id]]
        for cat_id in range(len(schema.categories))
    }
    if query_from_unused:
        # Swap out query item:
        if query_same_item:
            replacement_object = input_obj["unused_items"][schema.categories[query_cat_id]][0]
        else:
            replacement_object = random.choice(
                input_obj["unused_items"][schema.categories[query_cat_id]]
            )
        input_obj[f"Object.{query_cat_id}.{query_index}"] = replacement_object
        input_obj[f"Object.{query_cat_id}.Query"] = replacement_object

    input_obj["schemaName"] = schema.name

    return input_obj


def _format_list(items: list[str]) -> str:
    """Format a list of clauses."""
    if len(items) < 2:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def fill_prompt(sample, schema: Schema, template_key: str = "default"):
    prefix = ""
    if schema.templates.prefix:
        breakpoint()

    # Build context.
    template = schema.templates.definitions.get(template_key, "default")
    num_instances = sample["numInstances"]
    categories = schema.categories
    clauses = []
    for i in range(num_instances):
        mapping = {
            categories[cat_id]: sample[f"Object.{cat_id}.{i}"]
            for cat_id in range(len(categories))
        }
        clauses.append(template.format(**mapping))

    if schema.templates.capitalize_first_clause:
        clauses[0] = clauses[0][0].upper() + clauses[0][1:]

    context = _format_list(clauses) + "."

    # Build question.
    query_category = sample["queryCategory"]
    answer_category = sample["answerCategory"]
    query = schema.templates.queries[f"Q:{query_category} A:{answer_category}"]
    question = query.question.format(
        **{
            sample["queryCategory"]: sample[
                f"Object.{sample['queryCategoryIndex']}.Query"
            ]
        }
    )

    return f"{prefix}{context} {question}"


def build_prompt(
    schema: Schema,
    num_instances: int,
    query_cat_idx: int,
    answer_cat_idx: int,
    query_same_item: bool = False,
    query_idx_vals: list[int] | None = None,
    query_from_unused: bool = False,
):
    sample = fill_schema(
        schema,
        num_instances,
        query_cat_idx,
        answer_cat_idx,
        query_same_item=query_same_item,
        query_index_vals=query_idx_vals,
        query_from_unused=query_from_unused,
    )
    sample["raw_input"] = fill_prompt(sample, schema)
    return sample


def build_counterfactual_lexical(sample, schema):

    counterfactual = sample.copy()

    query_index = sample["queryIndex"]
    num_instances = sample["numInstances"]
    # cf_query_index = random.choice(list(set(range(num_instances)) - {query_index}))

    query_cat_id = sample["queryCategoryIndex"]
    replacement_object = random.choice(
        sample["unused_items"][schema.categories[query_cat_id]]
    )
    counterfactual[f"Object.{query_cat_id}.{query_index}"] = replacement_object
    counterfactual[f"Object.{query_cat_id}.Query"] = replacement_object
    counterfactual["raw_input"] = fill_prompt(counterfactual, schema)
    return counterfactual


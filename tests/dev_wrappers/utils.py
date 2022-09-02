def is_same_types(value, expected_type):
    if isinstance(value, expected_type):
        return True
    elif isinstance(value, dict):
        return all(is_same_types(sub_val, expected_type) for sub_val in value.values())
    elif isinstance(value, (tuple, list)):
        return all(is_same_types(sub_val, expected_type) for sub_val in value)
    else:
        raise Exception(
            f"Unexpected type: {type(value)}, expected types: {expected_type}, dict, tuple and list"
        )

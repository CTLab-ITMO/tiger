import copy
from itertools import product


class Params:

    def __init__(self, config, params):
        self._initial_config = copy.deepcopy(config)
        self._initial_params = copy.deepcopy(params)

    def __iter__(self):
        keys = []
        values = []

        all_keys = set(self._initial_config.keys()).union(set(self._initial_params.keys()))

        for field_name in all_keys:
            keys.append(field_name)

            initial_field_value = self._initial_config.get(field_name)
            params_fields_value = self._initial_params.get(field_name)

            if initial_field_value:
                if params_fields_value is None:  # We don't want to iterate through this field
                    values.append([initial_field_value])
                elif isinstance(initial_field_value, list) and isinstance(initial_field_value, list):
                    assert len(initial_field_value) == len(params_fields_value)
                    list_values = []
                    for i in range(len(initial_field_value)):
                        field_variations = list(Params(initial_field_value[i], params_fields_value[i]))
                        list_values.append(field_variations)
                    list_values = [p for p in product(*list_values)]
                    values.append(list_values)
                elif isinstance(initial_field_value, dict):  # It is composite param, need to go inside
                    field_variations = list(Params(initial_field_value, params_fields_value))
                    values.append(field_variations)
                else:  # Simple param, can take as it is
                    values.append([initial_field_value] + params_fields_value)
            else:
                values.append(self._initial_params[field_name])

        yield from [dict(zip(keys, p)) for p in product(*values)]

        return StopIteration

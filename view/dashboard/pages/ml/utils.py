def get_model_feat_params(string) -> list[int | str]:
    params = string.split("]_(")[0].split("[")[1]
    indicator_params = params.split(", ")
    indicator_params = [int(x) if x.isdigit() else x for x in indicator_params]
    indicator_params[-1] = indicator_params[2].replace("'", "")
    return indicator_params

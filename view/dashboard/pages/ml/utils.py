def get_model_feat_params(string) -> list[int | str]:
    params = string.split("]_(")[0].split("[")[1]
    indicator_params = params.split(", ")
    indicator_params = [int(x) if x.isdigit() else x for x in indicator_params]
    indicator_params[-1] = indicator_params[2].replace("'", "")
    return indicator_params

eval_metric = ['logloss', 'error']

scorings = [
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "average_precision",
    "neg_brier_score",
    "neg_log_loss",
    "f1",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "f1_samples",
    "precision",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "precision_samples",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "recall_samples",
    "jaccard_micro",
    "jaccard_macro",
    "jaccard_weighted",
    "jaccard_samples",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted",
]

def match_dimensions(task_flag: str, pred):
    if task_flag == "classification":
        pred = pred.squeeze(1)
    elif task_flag == "forecasting":
        pred = pred.squeeze(1)
    elif task_flag == "anomaly_detection":
        pred = pred
    else:
        raise ValueError("Task flag not recognized")
    return pred

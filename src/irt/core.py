import jsonlines
import numpy as np

from irt.math_utils import item_curve, estimate_ability_parameters


def create_irt_dataset(responses, dataset_name, question_ids=None):
    """
    Creates a dataset suitable for IRT analysis from a given set of responses and saves it in a JSON lines format.
    
    Parameters:
    - responses: A numpy array where each row represents a subject and each column a question.
    - dataset_name: The name of the file where the dataset will be saved.
    - question_ids: Optional list of question IDs (will use indices if not provided)
    
    Returns:
    - question_id_to_irt_id: Mapping from original question IDs to IRT item IDs (q0, q1, ...)
    """

    dataset = []
    question_id_to_irt_id = {}
    
    # Build mapping from original question IDs to IRT IDs
    if question_ids is not None:
        for j, qid in enumerate(question_ids):
            question_id_to_irt_id[str(qid)] = f'q{j}'
    
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}

        # Iterate over each question to create a response dict
        for j in range(responses.shape[1]):
            val = responses[i, j]
            # Skip missing responses (NaN) to match py-irt JSONL expectations
            if not np.isnan(val):
                aux_q['q' + str(j)] = int(val)
        # Only include subjects with at least one observed response
        if len(aux_q) > 0:
            aux['subject_id'] = str(i)
            aux['responses'] = aux_q
            dataset.append(aux)

    # Save the dataset in JSON lines format
    with jsonlines.open(dataset_name, mode='w') as writer:
        writer.write_all([dataset[i] for i in range(len(dataset))])
    
    return question_id_to_irt_id


def train_irt_model_python_api(dataset_name, D, lr, epochs, device, anchor_items: list[dict] | None = None, question_id_mapping: dict[str, str] | None = None,
                               lr_decay: float = 0.9999, deterministic: bool = True):
    """
    Trains an IRT model using the py-irt Python API.

    Parameters:
    - dataset_name: The name of the dataset file.
    - D: The number of dimensions for the IRT model.
    - lr: Learning rate for the model training.
    - epochs: The number of epochs to train the model.
    - device: The computing device ('cpu' or 'gpu') to use for training.
    - anchor_items: List of anchor item dicts with 'item_id', 'difficulty', 'discrimination', etc.
    - question_id_mapping: Mapping from original question IDs to IRT item IDs (q0, q1, ...)
    - lr_decay: Learning rate decay factor.
    - deterministic: Whether to use deterministic training (default: True).

    Returns:
    - trainer: The trained IRT model trainer object.
    """
    from py_irt.training import IrtConfig, IrtModelTrainer
    # Create IRT config
    config = IrtConfig(
        model_type='multidim_2pl',
        epochs=epochs,
        priors='hierarchical',
        dims=D,
        lr=lr,
        lr_decay=lr_decay,
        seed=42,
        deterministic=deterministic,
        log_every=max(epochs // 10, 1)  # Log every 10% of epochs
    )

    trainer_kwargs = {"data_path": dataset_name}

    if anchor_items:
        try:
            from py_irt.dataset import Dataset
        except ImportError as exc:
            raise RuntimeError(
                "Anchor-based calibration requires py-irt with AnchorItem support"
            ) from exc

        dataset = Dataset.from_jsonlines(dataset_name)
        if not hasattr(dataset, "add_anchor_items"):
            raise RuntimeError("Installed py-irt does not support add_anchor_items()")
        
        # Convert anchor item IDs to IRT format (q0, q1, ...) if mapping provided
        mapped_anchor_items = []
        if question_id_mapping:
            for item in anchor_items:
                orig_id = str(item["item_id"])
                if orig_id in question_id_mapping:
                    mapped_item = item.copy()
                    mapped_item["item_id"] = question_id_mapping[orig_id]
                    mapped_anchor_items.append(mapped_item)
            print(f"   🔗 Mapped {len(mapped_anchor_items)}/{len(anchor_items)} anchor items to IRT dataset")
        else:
            mapped_anchor_items = anchor_items
        
        if mapped_anchor_items:
            dataset.add_anchor_items(mapped_anchor_items)
            trainer_kwargs = {"dataset": dataset, "data_path": dataset_name}

            existing_initializers = list(getattr(config, "initializers", []) or [])
            if "anchor_items" not in existing_initializers:
                existing_initializers.append("anchor_items")
            config.initializers = existing_initializers

    trainer = IrtModelTrainer(config=config, **trainer_kwargs)
    trainer.train(device=device)
    return trainer


def load_irt_parameters_from_trainer(trainer):
    """
    Loads the parameters directly from a trained IRT model trainer.
    
    Parameters:
    - trainer: The trained IRT model trainer object.
    
    Returns: 
    - A, B, and Theta: The discrimination, difficulty, and ability parameters, respectively, from the IRT model.
    """
    result_params = trainer.best_params if getattr(trainer, "best_params", None) is not None else trainer.last_params
    a_list = result_params["disc"]
    b_list = result_params["diff"]
    theta_list = result_params["ability"]

    A = np.array(a_list).T[None, :, :]
    B = np.array(b_list).T[None, :, :]
    Theta = np.array(theta_list)[:, :, None]
    return A, B, Theta


import random
import numpy as np
from scipy.stats import entropy


# 1. Create instances_dict to keep track of instance information:
# labels: array of labels, []
# train_or_test: string, 'train' or 'test'
# instance_score: float, adjusted sum of label scores
def create_instances_dict(X, y, target_test_size):
    instances_dict = {}
    instance_id = 0
    for _ in X:
        train_or_test = "train"
        if random.uniform(0, 1) <= target_test_size:
            train_or_test = "test"
        instances_dict[instance_id] = {
            "labels": y[instance_id],
            "train_or_test": train_or_test,
            "instance_score": 0,
        }
        instance_id += 1
    return instances_dict


# 2. Create labels_dict to keep track of label information:
# train: int, number of times label appears in train set
# test: int, number of times label appears in test set
# label_score: float, label score
def create_labels_dict(instances_dict):
    labels_dict = {}
    for _, instance_dict in instances_dict.items():
        train_or_test = instance_dict["train_or_test"]
        for label in instance_dict["labels"]:
            try:
                if train_or_test == "train":
                    labels_dict[label]["train"] += 1
                else:
                    labels_dict[label]["test"] += 1
            except:
                if train_or_test == "train":
                    labels_dict[label] = {"train": 1, "test": 0, "label_score": 0}
                else:
                    labels_dict[label] = {"train": 0, "test": 1, "label_score": 0}
    return labels_dict


# 3. Calculate the label score for each label in labels_dict
# Positive score if too much of the label is in the test set
# Negative score if too much of the label is in the train set
def score_labels(labels_dict, target_test_size, average_labels_per_instance):
    for label, label_dict in labels_dict.items():
        label_score = 0
        label_count = label_dict["train"] + label_dict["test"]
        if label_count > 1:
            actual_test_proportion = label_dict["test"] / label_count
            if (
                actual_test_proportion >= target_test_size
            ):  # Too much of the label is in the test set
                label_score = (actual_test_proportion - target_test_size) / (
                    1 - target_test_size
                )
                if actual_test_proportion > 0.999:
                    label_score += average_labels_per_instance
            else:  # Too much of the label is in the train set
                label_score = (
                    actual_test_proportion - target_test_size
                ) / target_test_size
                if actual_test_proportion < 0.001:
                    label_score -= average_labels_per_instance
        labels_dict[label]["label_score"] = label_score


# 4. Calculate the instance score for each instance in instances_dict
# A high score means the instance is a good candidate for swapping
def score_instances(instances_dict, labels_dict):
    for instance_id, instance_dict in instances_dict.items():
        instance_score = 0
        train_or_test = instance_dict["train_or_test"]
        for label in instance_dict["labels"]:
            label_score = labels_dict[label]["label_score"]
            if label_score > 0:  # If too much of the label is in the test set
                if train_or_test == "test":
                    instance_score += label_score  # If instance in test, increase score
                elif train_or_test == "train":
                    instance_score -= (
                        label_score  # If instance in train, decrease score
                    )
                else:
                    print(f"Something went wrong: {instance_id}")
            elif label_score < 0:  # If too much of the label is in the train set
                if train_or_test == "train":
                    instance_score -= (
                        label_score  # If instance in train, increase score
                    )
                elif train_or_test == "test":
                    instance_score += label_score  # If instance in test, decrease score
                else:
                    print(f"Something went wrong: {instance_id}")
        instances_dict[instance_id]["instance_score"] = instance_score


# 5. Calculate the total score
# The higher the score, the more 'imbalanced' the distribution of labels between train and test sets
def calculate_total_score(instances_dict):
    total_score = 0
    for _, instance_dict in instances_dict.items():
        total_score += instance_dict["instance_score"]
    return total_score


# 6. Calculate the threshold score for swapping
def calculte_threshold_score(
    instances_dict, average_labels_per_instance, epoch, threshold_proportion, decay
):
    instance_scores = []
    for _, instance_dict in instances_dict.items():
        if instance_dict["instance_score"] < average_labels_per_instance:
            instance_scores.append(instance_dict["instance_score"])
    threshold_score = np.quantile(
        instance_scores, (1 - (threshold_proportion / ((1 + decay) ** epoch)))
    )
    if threshold_score < 0:
        threshold_score = 0
    return threshold_score


# 7. Swap the instances with instance_score that is greater than the threshold score
# Probability of swapping an instance is swap_probability
def swap_instances(
    instances_dict,
    threshold_score,
    swap_counter,
    average_labels_per_instance,
    epoch,
    swap_probability,
    decay,
):
    for instance_id, instance_dict in instances_dict.items():
        instance_score = instance_dict["instance_score"]
        if instance_score >= average_labels_per_instance:
            if random.uniform(0, 1) <= 0.25 / (1.05**epoch):
                current_group = instance_dict["train_or_test"]
                if current_group == "train":
                    instances_dict[instance_id]["train_or_test"] = "test"
                    swap_counter["to_test"] += 1
                elif current_group == "test":
                    instances_dict[instance_id]["train_or_test"] = "train"
                    swap_counter["to_train"] += 1
        elif instance_score > threshold_score and random.uniform(
            0, 1
        ) <= swap_probability / ((1 + decay) ** epoch):
            current_group = instance_dict["train_or_test"]
            if current_group == "train":
                instances_dict[instance_id]["train_or_test"] = "test"
                swap_counter["to_test"] += 1
            elif current_group == "test":
                instances_dict[instance_id]["train_or_test"] = "train"
                swap_counter["to_train"] += 1


def labels_not_in_split(
    all_codes: list[list[str]], split_codes: list[list[str]]
) -> float:
    """Find percentage of labels that are not in the split. Used to validate the splits"""
    all_codes_unique = {code for codes in all_codes for code in codes}
    split_codes_unique = {code for codes in split_codes for code in codes}
    labels_not_in_split = all_codes_unique - split_codes_unique
    return len(labels_not_in_split) * 100 / len(all_codes_unique)


def kl_divergence(all_codes: list[list[str]], split_codes: list[list[str]]) -> float:
    """Find KL divergence between the all and split set."""
    all_codes_unique = {code for codes in all_codes for code in codes}
    code2index = {code: i for i, code in enumerate(all_codes_unique)}
    all_counts = np.zeros(len(code2index))
    split_counts = np.zeros(len(code2index))
    for codes in all_codes:
        for code in codes:
            all_counts[code2index[code]] += 1
    for codes in split_codes:
        for code in codes:
            split_counts[code2index[code]] += 1

    all_counts = all_counts / np.sum(all_counts)
    split_counts = split_counts / np.sum(split_counts)

    return entropy(split_counts, qk=all_counts)

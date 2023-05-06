from datasource import *
from query import *
from sampler import *
from selector import *

from tqdm import tqdm


def run(dataset: int, rp: int, importance: int):
    """
    dataset:
        1:imagenet
        2:nightstreet
        3:ontonotes
        4:tacred
        5:beta(0.01,1.0) 
        6:beta (0.01,2.0)
    rp:
        1:recall
        2:precision
        3:joint
    imporant:
        0:uniform
        1:importance 
        2:naive   

    only jackson need estimator
    """
    dataset_names = {
        1: "imagenet",
        2: "nightstreet",
        3: "ontonotes",
        4: "tacred",
        5: "beta(0.01,1.0)",
        6: "beta(0.01,2.0)",
    }
    rp_names = {1: "recall", 2: "precision", 3: "joint"}
    importance_names = {0: "uniform", 1: "importance", 2: "naive"}

    experiment_name = f"{dataset_names[dataset]}_{rp_names[rp]}_{importance_names[importance]}"
    print(f"Running experiment: {experiment_name}")

    datasource = get_datasource(dataset=dataset)

    query = get_query(dataset, rp)
    sampler = get_sampler(importance)
    selector = get_selector(datasource, query, sampler, rp, importance)
    trial = config.trial
    verbose = config.verbose

    sorted_id = datasource.get_sorted_id()
    print(sorted_id)

    true_labels = datasource.get_label(sorted_id)
    print(true_labels)

    true_number = np.sum(true_labels)
    print(true_number)

    results = []

    if verbose:
        itr = tqdm(range(trial))
    else:
        itr = range(trial)

    for i in itr:
        selected_indexes = selector.select()
        selected_number = np.sum(datasource.get_label(selected_indexes))
        precision = selected_number / len(selected_indexes)
        recall = selected_number / true_number

        if query.query_type == 'jt':
            sampled_number = selector.total_sampled  # JointSelector
        else:
            sampled_number = query.budget
        results.append({

            "query_type": query.query_type,
            "precision": precision,
            "recall": recall,
            'size': len(selected_indexes),
            'true_number': true_number,
            'sampled_number': sampled_number,
            "trial_id": i

        })
        sampler.reset()

    results_df = pd.DataFrame(results)

    check_covered(results_df=results_df, query=query)

    final_results = results_df.aggregate({

        "precision": ["mean", "sem"],
        "recall": ["mean", "sem"],
        "covered": ["mean", "sem"],
        'size': ['mean', 'sem'],
        'true_number': ['mean', 'sem'],
        'sampled_number': ['mean', 'sem'],
        "trial_id": ["count"],
    })
    print(final_results)


def check_covered(results_df: pd.DataFrame, query):
    if query.query_type == "pt":
        # print(query.min_precision)
        results_df["covered"] = results_df["precision"] > query.min_precision
    elif query.query_type == "rt":
        results_df["covered"] = results_df["recall"] > query.min_recall
    elif query.query_type == "jt":
        results_df["covered"] = (
                (results_df["recall"] > query.min_recall)
                & (results_df["precision"] > query.min_precision)
        )
    else:
        results_df["covered"] = False


def get_datasource(dataset: int) -> DataSource:
    if dataset == 1:
        return get_imagenet()
    if dataset == 2:
        return get_nightstreet()
    if dataset == 3:
        return get_ontonotes()
    if dataset == 4:
        return get_tacred()
    if dataset == 5:
        return get_beta(0.01, 1, 0.2)  # alpha beta noise
    if dataset == 6:
        return get_beta(0.01, 2, 0.2)


def get_query(dataset: int, rp: int) -> ApproximateQuery:
    if dataset == 1 or dataset == 3 or dataset == 4:
        if rp == 1:
            return ApproximateQuery("rt", 0.95, 0.95, 1000, 0.05)
        if rp == 2:
            return ApproximateQuery("pt", 0.9, 0.95, 1000, 0.05)
        else:
            return ApproximateQuery("jt", 0.9, 0.9, 1000, 0.05)
    if dataset == 2:
        if rp == 1:
            return ApproximateQuery("rt", 0.95, 0.95, 10000, 0.05)
        if rp == 2:
            return ApproximateQuery("pt", 0.9, 0.95, 10000, 0.05)
        else:
            return ApproximateQuery("jt", 0.9, 0.9, 10000, 0.05)
    else:  # dataset 5 or 6
        if rp == 1:
            return ApproximateQuery("rt", 0.95, 0.95, 10000, 0.05)
        if rp == 2:
            return ApproximateQuery("pt", 0.9, 0.95, 10000, 0.05)
        else:
            return ApproximateQuery("jt", 0.9, 0.9, 10000, 0.05)


def get_sampler(importance) -> Sampler:
    if importance == 0 or importance == 2:
        return Sampler()
    else:
        return ImportanceSampler()


def get_selector(dataset, query, sampler, rp, importance) -> Selector:
    if rp == 3:
        if importance == 1:
            return JointSelector(query, dataset, sampler, "sqrt")
        if importance == 2:
            return JointSelector(query, dataset, sampler, "prop")
        else:
            return JointSelector(query, dataset, sampler, "uniform")
    if rp == 2:  # precision
        if importance == 0:
            return UniformPrecisionSelector(query, dataset, sampler)
        if importance == 1:
            return ImportancePrecisionSelector(query, dataset, sampler)
        else:
            return PrecisionSelector(query, dataset, sampler)
    if rp == 1:
        if importance == 0:
            return RecallModeSelector(query, dataset, sampler, "uniform")
        if importance == 1:
            return RecallModeSelector(query, dataset, sampler, "sqrt")
        else:
            return RecallSelector(query, dataset, sampler)


def main():
    for i in range(1, 6):  # 1         1:imagenet 2:nightstreet 3:onto notes 4:tacred 5:beta(0.01,1.0)
        if i == 2:  # imagenet dataset are lack of labels
            continue
        for j in range(1, 3):  # 1 recall 2 precision
            run(i, j, 0)
            run(i, j, 1)
    run(5, 3, 0)  # beta 1.0 joint uniform
    run(5, 3, 1)  # beta 1.0 joint importance
    run(6, 2, 1)  # beta 2.0 precision importance


if __name__ == '__main__':
    main()

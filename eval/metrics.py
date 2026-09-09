from numpy import mean


def recall_at_k(retrieved : list[str], relevant : list[str], k: int):
    intersection = len(set(retrieved[:k]) & set(relevant))
    total_relevant = len(relevant)

    recall = intersection / total_relevant

    return recall


def precision_at_k(retrieved : list[str], relevant : list[str], k : int):
    top_k = retrieved[:k]
    intersection = len(set(top_k) & set(relevant))

    precision = 0

    try:
        precision = intersection / len(top_k)
    except ZeroDivisionError:
        return precision

    return precision


def reciprocal_rank(retrieved : list[str], relevant : list[str]):
    rank = 0
    for index, element in enumerate(retrieved, start=1):
        if element in relevant:
            rank = index
            break


    if rank == 0:
        return 0
    

    reciprocal_rank = 1 / rank

    return reciprocal_rank


def mrr(list_of_retrieved: list[list[str]], list_of_relevant: list[list[str]]):

    results = []

    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        result = reciprocal_rank(retrieved, relevant)
        results.append(result)

    mrr = mean(results)

    return mrr
def find_eps_sample(points):
    pass


def find_stripe(eps_sample, scoring_fn, k):
    pass


def query(index, stripe):
    pass


def preprocess(points, index_fn, **kwargs):
    # Build index
    index = index_fn(points, kwargs)

    # Build eps sample
    eps_sample = find_eps_sample(points)

    return (index, eps_sample)


def find_kth(index, eps_sample, scoring_fn, k):
    stripe = find_stripe(eps_sample, scoring_fn, k)
    return query(index, stripe)

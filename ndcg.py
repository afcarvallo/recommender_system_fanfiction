import numpy as np

def dcg(r, i):
    r = np.asarray(r)[:i]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0

# ndcg at i 
def ndcg(r, i):
    # dcg ideal ordena articulos por relevancia y les calcula el dcg 
    ideal_dcg = dcg(sorted(r, reverse=True), i)
    if not ideal_dcg:
        return 0
    return dcg(r, i) / ideal_dcg




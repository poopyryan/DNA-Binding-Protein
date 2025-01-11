from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold

# takes seq (string) and k (int), returns list of enumerated kmers (list)
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k)]

# takes seq (string) and k (int), returns dictionary of kmers(string), count(int)
def get_kmers_count(seq, k):
    kmers_lst = get_kmers(seq, k)
    kmers_count = dict()
    for kmer in kmers_lst:
        kmers_count[kmer] = kmers_count.get(kmer, 0) + 1
    return kmers_count

def get_counts(train_set, test_set, k, ngram_tuple):
    train_set['words'] = train_set.apply(lambda x: get_kmers(x['Sequences'], k), axis = 1)
    train_set = train_set.drop('Sequences', axis = 1)

    kmer = list(train_set['words'])
    for i in range(len(kmer)):
        kmer[i] = ' '.join(kmer[i])
    y_train = train_set.iloc[:, 0].values

    cv = CountVectorizer(ngram_range = ngram_tuple)
    X_train = cv.fit_transform(kmer)

    test_set['words'] = test_set.apply(lambda x: get_kmers(x['Sequences'], k), axis = 1)
    test_set = test_set.drop('Sequences', axis = 1)

    kmer_test = list(test_set['words'])
    for i in range(len(kmer_test)):
        kmer_test[i] = ' '.join(kmer_test[i])
    y_test = test_set.iloc[:, 0].values

    X_test = cv.transform(kmer_test)

    return y_train, X_train, y_test, X_test

def selecting_high_variance_features(X_train, X_test, var):
    selector = VarianceThreshold(threshold = var)

    X_train = selector.fit_transform(X_train)
    X_test = selector.transform(X_test)

    return X_train, X_test



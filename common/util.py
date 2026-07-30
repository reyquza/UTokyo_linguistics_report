from .np import *


def preprocess(text: str ='You say goodbye and I say hello.'):
    """ 
    'You say goodbye and I say hello.'
    -> corpus = [0 1 2 3 4 1 5 6] 
    """
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")
    
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
           new_id = len(word_to_id)
           word_to_id[word] = new_id
           id_to_word[new_id] = word
           
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


def create_comatrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        left = max(idx - window_size, 0)
        right = min(idx + window_size + 1, corpus_size)

        # 左側
        for context_id in corpus[left:idx]:
            co_matrix[word_id, context_id] += 1

        # 右側
        for context_id in corpus[idx+1:right]:
            co_matrix[word_id, context_id] += 1

    return co_matrix


def ppmi(C, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0) # そのidに対応する単語が何回出てきたか？
    total = C.shape[0] * C.shape[1]
    
    for i in range(C.shpae[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N /(S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)
    
    return M
    

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.linalg.norm(x) + eps)
    ny = y / (np.linalg.norm(y) + eps)
    return np.dot(nx, ny)


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
        
    return np.array(contexts), np.array(target)


def convert_one_hot(id_list, vocab_size):
    
    N = id_list.shape[0]
    if id_list.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(id_list):
            one_hot[idx, word_id] = 1
    
    elif id_list.ndim == 2:
        C = id_list.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        
        for idx_0, word_ids in enumerate(id_list):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
            

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''類似単語の検索

    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
        
        
def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s is not found' % word)
            return

    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
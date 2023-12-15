import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')

# 多个参考翻译
references = [
    ['The cat is on the mat.', 'The cat is lying on the rug.'],
    ['The quick brown fox jumps over the lazy dog.', 'A fast brown fox leaps over the sleepy canine.'],
    ['A beautiful sunset over the mountains.', 'The mountains are bathed in the warm hues of the setting sun.']
]

# 生成翻译
hypotheses = [
    'The cat is on the rug.',
    'A fast brown fox jumps over the lazy dog.',
    'The mountains are bathed in the warm hues of the setting sun.'
]

# 分词处理参考翻译
tokenized_references = [[nltk.word_tokenize(ref) for ref in group] for group in references]

# 分词处理生成翻译
tokenized_hypotheses = [nltk.word_tokenize(hyp) for hyp in hypotheses]

# 计算BLEU分数
bleu1 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

print("BLEU-1: ", bleu1)
print("BLEU-2: ", bleu2)
print("BLEU-3: ", bleu3)
print("BLEU-4: ", bleu4)

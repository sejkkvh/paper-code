from rouge import Rouge

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

rouge = Rouge()

# 对于每个生成翻译，计算与所有参考翻译的ROUGE-L分数，然后取最高值
scores = []
for hyp, refs in zip(hypotheses, references):
    hyp_scores = [rouge.get_scores(hyp, ref)[0]['rouge-l'] for ref in refs]
    best_score = max(hyp_scores, key=lambda x: x['f'])
    scores.append(best_score)

# 输出每个句子的最佳ROUGE-L分数
for i, score in enumerate(scores):
    print(f"句子 {i+1} 的最佳ROUGE-L分数: {score}")

# 如果需要整体的ROUGE-L分数
total_score = {
    'f': sum([score['f'] for score in scores]) / len(scores),
    'p': sum([score['p'] for score in scores]) / len(scores),
    'r': sum([score['r'] for score in scores]) / len(scores)
}
print(f"整体ROUGE-L分数: {total_score}")


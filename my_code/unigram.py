import numpy as np
import utils
import collections


def train(papers, reviews, training_model, bernoulli):
    total_accepted = 0
    total_papers = 0
    paper_words = utils.get_words_from_papers(papers)

    for id in papers.keys():
        rtitle, rabstract, accepted, per_reviews = reviews[id]
        seen_words = set()
        total_papers += 1
        if accepted:
            total_accepted += 1

        for word in paper_words[id]:
            if bernoulli:
                if word in seen_words:
                    continue
                else:
                    seen_words.add(word)

            if word not in training_model:
                training_model[word] = (0, 0)

            accepted_count, total = training_model[word]
            if accepted:
                training_model[word] = (accepted_count + 1, total + 1)
            else:
                training_model[word] = (accepted_count, total + 1)

    prob_accepted = total_accepted / total_papers

    prob_model = dict()
    for word in training_model:
        accepted, total = training_model[word]
        if accepted > 0 and accepted / total < 1.0:
            prob_model[word] = accepted / total

    best = collections.Counter(prob_model)
    for word, prob in best.most_common(10):
        print(word, prob, training_model[word])

    return training_model, prob_accepted


def add_unseen_vocab(data, papers):
    new_data = data.copy()

    paper_data = utils.get_words_from_papers(papers)
    for id in paper_data.keys():
        for word in paper_data[id]:
            if word not in new_data:
                new_data[word] = (0, 0)

    return new_data


def smooth_words(data, smoothing_alpha=0.01):
    new_data = data.copy()

    for word in new_data.keys():
        apt, tot = new_data[word]
        new_data[word] = (apt + smoothing_alpha, tot + smoothing_alpha * 2)

    return new_data


def evaluate_data(data, papers, reviews, prob_accepted, bernoulli):
    actual_matched = 0
    matched_total = 0
    total_papers = 0
    guess_matched = 0
    false_positive = 0
    false_negative = 0

    paper_words = utils.get_words_from_papers(papers)
    for id in paper_words.keys():
        acceptance = 0
        rejection = 0
        rtitle, rabstract, accepted, per_reviews = reviews[id]
        seen_words = set()

        for word in paper_words[id]:
            if bernoulli:
                if word in seen_words:
                    continue
                else:
                    seen_words.add(word)
            accepted_count, total = data[word]
            acceptance += np.log2(accepted_count / total)
            rejection += np.log2((total - accepted_count) / total)

        acceptance += np.log2(prob_accepted)
        rejection += np.log2(1 - prob_accepted)
        guess_accept = acceptance > rejection

        total_papers += 1
        if accepted:
            actual_matched += 1
        if guess_accept:
            guess_matched += 1
        if guess_accept == accepted:
            matched_total += 1
        else:
            if accepted:
                false_negative += 1
            else:
                false_positive += 1

    print("Correctly matched:", matched_total, "; papers accepted:", actual_matched, "; papers guessed accepted:", guess_matched, "; total papers:", total_papers,
          "; accuracy:", matched_total / total_papers, "; false negatives:", false_negative, "; false positives:", false_positive)
    print('Accuracy:', matched_total / total_papers, 'majority:', (total_papers - actual_matched) / total_papers)


def test(*directory, background_model=None, bernoulli, include_dev=False):
    data = dict()
    all_papers = {}
    all_reviews = {}
    for x in directory:
        paper_train_dir = x + '/train/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))
        review_train_dir = x + '/train/reviews'
        all_reviews.update(utils.parse_reviews(review_train_dir))

    data, prob_accepted = train(all_papers, all_reviews, data, bernoulli)

    all_papers = {}
    all_reviews = {}
    for x in directory:
        if include_dev:
            paper_dev_dir = x + '/dev/parsed_pdfs'
            all_papers.update(utils.parse_pdfs(paper_dev_dir))
            review_dev_dir = x + '/dev/reviews'
            all_reviews.update(utils.parse_reviews(review_dev_dir))
        paper_dev_dir = x + '/test/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_dev_dir))
        review_dev_dir = x + '/test/reviews'
        all_reviews.update(utils.parse_reviews(review_dev_dir))

    data = add_unseen_vocab(data, all_papers)
    data = smooth_words(data)

    print("Unigram test for ", directory)
    evaluate_data(data, all_papers, all_reviews, prob_accepted, bernoulli)


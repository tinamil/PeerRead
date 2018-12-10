import numpy as np
import utils


def train(papers, reviews, training_model):
    for id in papers.keys():
        title, sections, references, abstract, year, emails = papers[id]
        rtitle, rabstract, accepted, per_reviews = reviews[id]
        if sections is not None:
            for x in sections:
                header = x['heading']
                text = x['text']
                for word in text.split():
                    if word not in training_model:
                        training_model[word] = (0, 0)

                    accepted_count, total = training_model[word]
                    if accepted:
                        training_model[word] = (accepted_count + 1, total + 1)
                    else:
                        training_model[word] = (accepted_count, total + 1)
    return training_model


def add_unseen_vocab(data, papers):
    new_data = data.copy()

    paper_data = utils.get_words_from_papers(papers)
    for id in paper_data.keys():
        for word in paper_data[id]:
            if word not in new_data:
                new_data[word] = (0, 0)

    return new_data


def smooth_words(data, smoothing_alpha=0.001):
    new_data = data.copy()

    for word in new_data.keys():
        apt, tot = new_data[word]
        new_data[word] = (apt + smoothing_alpha, tot + smoothing_alpha * 2)

    return new_data


def evaluate_data(data, papers, reviews):
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

        for word in paper_words[id]:
            accepted_count, total = data[word]
            if accepted_count == 0:
                rejection += 1

            acceptance += np.log2(accepted_count / total)
            rejection += np.log2((total - accepted_count) / total)

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


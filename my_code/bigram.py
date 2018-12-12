import numpy as np
import utils

START_WORD = '[START]'
STOP_WORD = '[STOP]'


def train(papers, reviews, training_model):
    total_accepted = 0
    total_papers = 0
    paper_words = utils.get_words_from_papers(papers)
    for id in papers.keys():
        #title, sections, references, abstract, year, emails = papers[id]
        rtitle, rabstract, accepted, per_reviews = reviews[id]

        total_papers += 1
        if accepted:
            total_accepted += 1

        last_word = START_WORD
        for word in paper_words[id]:
            if last_word not in training_model:
                training_model[last_word] = dict()

            if word not in training_model[last_word]:
                training_model[last_word][word] = (0, 0)

            accepted_count, total = training_model[last_word][word]
            if accepted:
                training_model[last_word][word] = (accepted_count + 1, total + 1)
            else:
                training_model[last_word][word] = (accepted_count, total + 1)
            last_word = word

        word = STOP_WORD
        if last_word not in training_model:
            training_model[last_word] = dict()
        if word not in training_model[last_word]:
            training_model[last_word][word] = (0, 0)
        accepted_count, total = training_model[last_word][word]
        if accepted:
            training_model[last_word][word] = (accepted_count + 1, total + 1)
        else:
            training_model[last_word][word] = (accepted_count, total + 1)
    return training_model, total_accepted, total_papers


def add_unseen_vocab(data, papers):
    new_data = data.copy()

    paper_data = utils.get_words_from_papers(papers)
    for id in paper_data.keys():
        last_word = START_WORD
        for word in paper_data[id]:
            if last_word not in new_data:
                new_data[last_word] = dict()
            if word not in new_data[last_word]:
                new_data[last_word][word] = (0, 0)
            last_word = word
        word = STOP_WORD
        if last_word not in new_data:
            new_data[last_word] = dict()
        if word not in new_data[last_word]:
            new_data[last_word][word] = (0, 0)

    return new_data


def smooth_words(data, smoothing_alpha=0.01):
    new_data = data.copy()

    for last_word in new_data.keys():
        for word in new_data[last_word]:
            apt, tot = new_data[last_word][word]
            new_data[last_word][word] = (apt + smoothing_alpha, tot + smoothing_alpha * 2)

    return new_data


def evaluate_data(data, papers, reviews, total_accepted_papers, total_test_papers):
    actual_matched = 0
    matched_total = 0
    total_papers = 0
    guess_matched = 0
    false_positive = 0
    false_negative = 0

    accepted_probability = total_accepted_papers / total_test_papers
    rejected_probability = (total_test_papers - total_accepted_papers) / total_test_papers

    paper_words = utils.get_words_from_papers(papers)
    for id in paper_words.keys():
        acceptance = 0
        rejection = 0
        rtitle, rabstract, accepted, per_reviews = reviews[id]
        last_word = START_WORD
        for word in paper_words[id]:
            accepted_count, total = data[last_word][word]
            if accepted_count == 0:
                rejection += 1

            acceptance += np.log2(accepted_count / total)
            rejection += np.log2((total - accepted_count) / total)
            last_word = word

        acceptance += np.log2(accepted_probability)
        rejection += np.log2(rejected_probability)

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


def test(*directory):
    data = dict()
    all_papers = {}
    all_reviews = {}
    for x in directory:
        paper_train_dir = x + '/train/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))
        review_train_dir = x + '/train/reviews'
        all_reviews.update(utils.parse_reviews(review_train_dir))

    data, total_accepted, total_papers = train(all_papers, all_reviews, data)

    all_papers = {}
    all_reviews = {}
    for x in directory:
        paper_dev_dir = x + '/dev/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_dev_dir))
        review_dev_dir = x + '/dev/reviews'
        all_reviews.update(utils.parse_reviews(review_dev_dir))

    data = add_unseen_vocab(data, all_papers)
    data = smooth_words(data)

    print("Bigram test for ", directory)
    evaluate_data(data, all_papers, all_reviews, total_accepted, total_papers)


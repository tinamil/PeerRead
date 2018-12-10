import utils
import unigram
import os


def test_unigram(*directory):
    data = dict()
    all_papers = {}
    all_reviews = {}
    for x in directory:
        paper_train_dir = x + '/train/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))
        review_train_dir = x + '/train/reviews'
        all_reviews.update(utils.parse_reviews(review_train_dir))

    data = unigram.train(all_papers, all_reviews, data)

    all_papers = {}
    all_reviews = {}
    for x in directory:
        paper_dev_dir = x + '/dev/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_dev_dir))
        review_dev_dir = x + '/dev/reviews'
        all_reviews.update(utils.parse_reviews(review_dev_dir))

    data = unigram.add_unseen_vocab(data, all_papers)
    data = unigram.smooth_words(data)

    print("Unigram test for ", directory)
    unigram.evaluate_data(data, all_papers, all_reviews)


if __name__ == '__main__':
    test_unigram('../data/iclr_2017')
    test_unigram('../data/arxiv.cs.ai_2007-2017')
    test_unigram('../data/arxiv.cs.cl_2007-2017')
    test_unigram('../data/arxiv.cs.lg_2007-2017')

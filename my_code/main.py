import unigram
import bigram
import background


if __name__ == '__main__':
    bernoulli = True
    #background_model = background.build_model('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017',
    #                                         '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017')
    unigram.test('../data/iclr_2017', bernoulli=bernoulli)
    unigram.test('../data/arxiv.cs.ai_2007-2017', bernoulli=bernoulli)
    unigram.test('../data/arxiv.cs.cl_2007-2017', bernoulli=bernoulli)
    unigram.test('../data/arxiv.cs.lg_2007-2017', bernoulli=bernoulli)
    unigram.test('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017', '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017', bernoulli=bernoulli)

    bigram.test('../data/iclr_2017', bernoulli=bernoulli, include_dev=True)
    bigram.test('../data/arxiv.cs.ai_2007-2017', bernoulli=bernoulli)
    bigram.test('../data/arxiv.cs.cl_2007-2017', bernoulli=bernoulli)
    bigram.test('../data/arxiv.cs.lg_2007-2017', bernoulli=bernoulli)
    bigram.test('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017', '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017', bernoulli=bernoulli)

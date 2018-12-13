import unigram
import bigram
import background


if __name__ == '__main__':
    #background_model = background.build_model('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017',
    #                                         '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017')
    unigram.test('../data/iclr_2017')#, background_model)
    #unigram.test('../data/arxiv.cs.ai_2007-2017')
    #unigram.test('../data/arxiv.cs.cl_2007-2017')
    #unigram.test('../data/arxiv.cs.lg_2007-2017')
    #unigram.test('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017', '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017')

    #bigram.test('../data/iclr_2017')
    #bigram.test('../data/arxiv.cs.ai_2007-2017')
    #bigram.test('../data/arxiv.cs.cl_2007-2017')
    #bigram.test('../data/arxiv.cs.lg_2007-2017')
    #bigram.test('../data/iclr_2017', '../data/arxiv.cs.ai_2007-2017', '../data/arxiv.cs.cl_2007-2017', '../data/arxiv.cs.lg_2007-2017')

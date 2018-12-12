import utils


def build_model(*directory):
    all_papers = {}
    for x in directory:
        paper_train_dir = x + '/train/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))
        paper_train_dir = x + '/dev/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))
        paper_train_dir = x + '/test/parsed_pdfs'
        all_papers.update(utils.parse_pdfs(paper_train_dir))

    background_model = dict()
    papers = utils.get_words_from_papers(all_papers)
    for id in papers.keys():
        for word in all_papers[id]:
            if word not in background_model:
                background_model[word] = 0
            background_model[word] += 1
    all_words = sum(background_model.values())
    for word in background_model.keys():
        background_model[word] = background_model[word] / all_words
    return background_model


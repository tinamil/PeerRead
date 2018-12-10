import json, os
import numpy as np


def read_json(json, tag: str):
    if tag in json:
        return json[tag]
    elif tag.upper() in json:
        return json[tag.upper()]
    elif tag.lower() in json:
        return json[tag.lower()]
    return ""


def parse_reviews(folder):
    data = dict()
    for root, dirs, files in os.walk(folder):
        for file in files:
            index = int(file.split('.')[0])
            title, abstract, accepted, reviews = parse_review_file(os.path.join(root, file))
            data[index] = (title, abstract, accepted, reviews)
    return data


def parse_review_file(file):
    with open(file, 'rb') as json_data:
        d = json.load(json_data, encoding="utf8")
        title = d['title']
        abstract = d['abstract']
        accepted = d['accepted']
        reviews = d['reviews']
        results = []
        for x in reviews:
            comments = read_json(x, 'COMMENTS')
            soundness = read_json(x, 'SOUNDNESS_CORRECTNESS')
            originality = read_json(x, 'ORIGINALITY')
            clarity = read_json(x, 'CLARITY')
            recommendation = read_json(x, 'RECOMMENDATION')
            confidence = read_json(x, 'REVIEWER_CONFIDENCE')
            results.append((comments, soundness, originality, clarity, recommendation, confidence))
        return title, abstract, accepted, results


def parse_pdfs(folder):
    data = dict()
    for root, dirs, files in os.walk(folder):
        for file in files:
            index = int(file.split('.')[0])
            title, sections, references, abstract, year, emails = parse_pdf(os.path.join(root, file))
            data[index] = (title, sections, references, abstract, year, emails)
    return data


def parse_pdf(pdf):
    with open(pdf, 'rb') as json_data:
        d = json.load(json_data, encoding="utf8")
        title = d['metadata']['title']
        sections = d['metadata']['sections']
        references = d['metadata']['references']
        abstract = d['metadata']['abstractText']
        year = d['metadata']['year']
        emails = d['metadata']['emails']
        return title, sections, references, abstract, year, emails

def train(papers, reviews):
    words = dict()
    for id in papers.keys():
        title, sections, references, abstract, year, emails = papers[id]
        rtitle, rabstract, accepted, per_reviews = reviews[id]
        if sections is not None:
            for x in sections:
                header = x['heading']
                text = x['text']
                for word in text.split():
                    if word not in words:
                        words[word] = (0, 0)

                    accepted_count, total = words[word]
                    if accepted:
                        words[word] = (accepted_count + 1, total + 1)
                    else:
                        words[word] = (accepted_count, total + 1)
    return words


def get_words_from_papers(paper_dict):
    papers = {}
    for id in paper_dict.keys():
        title, sections, references, abstract, year, emails = paper_dict[id]
        if sections is not None:
            for x in sections:
                text = x['text']
                papers[id] = text.split()
    return papers


def smooth_data(data, papers, reviews, smoothing_alpha=0.1):
    new_data = data.copy()

    paper_data = get_words_from_papers(papers)
    for id in paper_data.keys():
        for word in paper_data[id]:
            if word not in new_data:
                new_data[word] = (0, 0)

    for word in new_data.keys():
        apt, tot = new_data[word]
        new_data[word] = (apt + smoothing_alpha, tot + smoothing_alpha * 2)

    return new_data


if __name__ == '__main__':
    papers = parse_pdfs('../data/iclr_2017/train/parsed_pdfs')
    reviews = parse_reviews('../data/iclr_2017/train/reviews')

    data = train(papers, reviews)

    papers = parse_pdfs('../data/iclr_2017/dev/parsed_pdfs')
    reviews = parse_reviews('../data/iclr_2017/dev/reviews')

    data = smooth_data(data, papers, reviews)

    actual_matched = 0
    matched_total = 0
    total_papers = 0
    guess_matched = 0
    false_positive = 0
    false_negative = 0
    paper_words = get_words_from_papers(papers)
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

    print(matched_total, actual_matched, guess_matched, total_papers, matched_total/total_papers, actual_matched/total_papers, guess_matched/total_papers)
    print(false_negative, false_positive)

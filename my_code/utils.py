import json, os

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
            index = file.replace('.json', '').replace('.', '')
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
            index = file.replace('.pdf.json', '').replace('.', '')
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


def get_words_from_papers(paper_dict):
    papers = {}
    for id in paper_dict.keys():
        title, sections, references, abstract, year, emails = paper_dict[id]
        papers[id] = []
        if sections is not None:
            for x in sections:
                text = x['text']
                papers[id].extend(text.split())
    return papers


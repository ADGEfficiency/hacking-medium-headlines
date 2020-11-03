import json

from bs4 import BeautifulSoup
import requests


def clean_headline(string):
    string = string.replace('\u00a0', ' ')
    return string.encode('ascii', 'ignore').decode()


def test_clean_claps():
    assert clean_claps('10.6K') == 10600
    assert clean_claps('10') == 10


def clean_claps(claps):
    if 'K' in claps:
        thousands = float(claps.replace('K', ''))
        return thousands * 1000
    else:
        return float(claps)


def extract_article_data(divs):
    #  TODO - refactor this look to work on a single div
    dataset = []
    for d in divs:
        h2 = d.findAll('h2')
        h3 = d.findAll('h3')
        #  sometimes it's an h2 heading
        #  see https://towardsdatascience.com/archive/2014
        if len(h3) == 0:
            h3 = h2

        claps = d.findAll('button', {'class': 'button button--chromeless u-baseColor--buttonNormal js-multirecommendCountButton u-disablePointerEvents'})
        if len(h3) == len(claps) == 1:
            data = {}
            data['headline-raw'] = h3[0].text
            data['headline'] = clean_headline(h3[0].text)
            data['claps-raw'] = claps[0].text
            data['claps'] = clean_claps(claps[0].text)
            dataset.append(data)

    return dataset


def save_article_data(dataset, name):
    with open(f'./data/{name}.jsonl', 'w') as fi:
        for js in dataset:
            fi.write(json.dumps(js) + '\n')


def scrape_all_archive():
    print(f'scraping all archive')
    url = 'https://towardsdatascience.com/archive'
    divs = request_and_parse(url)

    #  should be 10 articles in the all years archive
    #  monthly / yearly archives have variable number
    assert len(divs) == 10

    dataset = extract_article_data(divs)
    assert len(dataset) == 10
    save_article_data(dataset, 'all')


def request_and_parse(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")
    return soup.findAll('div', {'class': 'streamItem streamItem--postPreview js-streamItem'})


def scrape_year(year):
    year = str(year)
    print(f'scraping {year}')
    url = f'https://towardsdatascience.com/archive/{year}'
    divs = request_and_parse(url)
    dataset = extract_article_data(divs)
    save_article_data(dataset, year)

def scrape_year_month(year, month):

    import datetime
    dt = datetime.datetime(year, month, 1)
    dt = datetime.datetime(year, month, 1)
    month = dt.strftime('%m')
    year = str(year)
    print(f'scraping {year}-{month}')

    url = f'https://towardsdatascience.com/archive/{year}/{month}'
    divs = request_and_parse(url)
    dataset = extract_article_data(divs)
    save_article_data(dataset, f'{year}-{month}')


if __name__ == '__main__':
    import os
    os.makedirs('./data', exist_ok=True)
    test_clean_claps()

    scrape_all_archive()
    for year in range(2010, 2015):
        scrape_year(year)

    year = 2015
    for month in range(3, 13):
        scrape_year_month(year, month)

    for year in range(2016, 2020):
        for month in range(1, 13):
            scrape_year_month(year, month)

    year = 2020
    for month in range(1, 12):
        scrape_year_month(year, month)

import datetime
import json
import os

from bs4 import BeautifulSoup
import requests

from src.dirs import HOME


def clean_headline(string):
    string = string.replace('\u00a0', ' ')
    return string.encode('ascii', 'ignore').decode()


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
    with open(os.path.join(HOME, 'data', 'raw', f'{name}.jsonl'), 'w') as fi:
        for js in dataset:
            fi.write(json.dumps(js) + '\n')


def scrape_all_archive(site):
    print(f'scraping all archive')
    url = f'https://{site}.com/archive'
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


def scrape_year(site, year):
    year = str(year)
    print(f'scraping {year}')
    url = f'https://{site}.com/archive{year}'
    if not check_history(url):
        print(f'{url} failed - returning early')
        return None
    else:
        divs = request_and_parse(url)
        dataset = extract_article_data(divs)
        save_article_data(dataset, year)


def scrape_year_month(site, year, month):

    dt = datetime.datetime(year, month, 1)
    dt = datetime.datetime(year, month, 1)
    month = dt.strftime('%m')
    year = str(year)
    print(f'scraping {year}-{month}')

    url = f'https://{site}.com/archive/{year}/{month}'
    if not check_history(url):
        print(f'{url} failed - returning early')
        return None
    else:
        divs = request_and_parse(url)
        dataset = extract_article_data(divs)
        save_article_data(dataset, f'{year}-{month}')


def check_history(url):
    res = requests.get(url)
    if len(res.history) > 2:
        return False
    else:
        return True


if __name__ == '__main__':
    import os
    os.makedirs('./data', exist_ok=True)
    test_clean_claps()

    def scrape_medium_publication(site):
        scrape_all_archive(site)
        for year in range(2010, 2021):
            scrape_year(site, year)

            #  no sites have monthly before 2015
            if year >= 2015:
                for month in range(1, 13):
                    scrape_year_month(site, year, month)

    scrape_medium_publication('towardsdatascience')


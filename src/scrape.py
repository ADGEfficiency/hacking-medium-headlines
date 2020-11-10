import datetime
import json
import os

from bs4 import BeautifulSoup
import requests

from src.dirs import DATAHOME


def clean_headline(string):
    string = string.replace('\u00a0', ' ')
    return string.encode('ascii', 'ignore').decode()


def clean_claps(claps):
    if 'K' in claps:
        thousands = float(claps.replace('K', ''))
        return thousands * 1000
    else:
        return float(claps)


def extract_article_data(divs, site_id):
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
            data['site_id'] = site_id
            dataset.append(data)

    return dataset


def save_article_data(dataset, site, name):
    with open(os.path.join(
        DATAHOME,
        'raw',
        get_site_id(site),
        f'{name}.jsonl'
    ), 'w') as fi:
        for js in dataset:
            fi.write(json.dumps(js) + '\n')


def request_and_parse(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")
    return soup.findAll('div', {'class': 'streamItem streamItem--postPreview js-streamItem'})

from random import randint
from time import sleep


def scrape(site, year=None, month=None):
    sleep(randint(1, 3))
    print(f'scraping {site} - {year}-{month}')

    if year == month == None:
        fname = 'all'
        url = f'https://{site}/archive'
    elif month == None:
        fname = str(year)
        url = f'https://{site}/archive{year}'
    else:
        fname = f'{year}-{month}'
        url = f'https://{site}/archive/{year}/{month}'

    if not check_history(url):
        print(f'{url} failed - returning early')
        return None
    else:
        divs = request_and_parse(url)
        dataset = extract_article_data(divs, get_site_id(site))

        for js in dataset:
            js['year'] = year
            js['month'] = month
            js['site'] = get_site_id(site)

        print(f'  found {len(dataset)} articles')
        save_article_data(dataset, site, fname)


def check_history(url):
    res = requests.get(url)
    if len(res.history) > 2:
        return False
    else:
        return True


if __name__ == '__main__':
    import os

    def get_site_id(site):
        site_id =  site.split('/')[-1].split('.')[0]
        assert '.' not in site_id
        assert '/' not in site_id
        return site_id

    def scrape_medium_publication(site):
        os.makedirs(
            os.path.join(DATAHOME, 'raw', get_site_id(site)),
            exist_ok=True
        )

        for year in range(2015, 2021):
            #  no sites have monthly before 2015
            if year >= 2015:
                for month in range(1, 13):
                    year = int(year)
                    month = int(month)
                    dt = datetime.datetime(year, month, 1)
                    dt = datetime.datetime(year, month, 1)
                    month = dt.strftime('%m')
                    year = str(year)
                    scrape(site, year=year, month=month)

    # scrape_medium_publication('medium.com/hackernoon')
    # scrape_medium_publication('medium.com/free-code-camp')
    # scrape_medium_publication('towardsdatascience.com')
    # scrape_medium_publication('medium.com/the-mission')
    # scrape_medium_publication('medium.com/personal-growth')
    # scrape_medium_publication('medium.com/swlh')
    # scrape_medium_publication('medium.com/level-up-web')
    # scrape_medium_publication('medium.com/better-programming')
    # scrape_medium_publication('medium.com/better-humans')
    scrape_medium_publication('medium.com/dailyjs')
    scrape_medium_publication('levelup.gitconnected.com')

    scrape_medium_publication('writingcooperative.com')


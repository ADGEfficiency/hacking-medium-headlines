import datetime
import os
import json
from random import randint
from time import sleep

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


def save_article_data(dataset, site, fname):
    with open(fname, 'w') as fi:
        for js in dataset:
            fi.write(json.dumps(js) + '\n')


def request_and_parse(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")
    return soup.findAll('div', {'class': 'streamItem streamItem--postPreview js-streamItem'})


def scrape(site, year=None, month=None):
    print(f'scraping {site} - {year}-{month}')
    fname = DATAHOME / 'raw' / get_site_id(site) / f'{year}-{month}.jsonl'

    if fname.is_file():
        print(f'  {fname} already exists - returning early')
        return None

    url = f'https://{site}/archive/{year}/{month}'
    if not check_history(url):
        print(f'  {url} failed - returning early')
        return None

    else:
        sleep(randint(1, 3))
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


def get_site_id(site):
    site_id =  site.split('/')[-1].split('.')[0]
    assert '.' not in site_id
    assert '/' not in site_id
    return site_id


def scrape_medium_publication(site):
    if '.com' not in site:
        site = f'medium.com/{site}'

    os.makedirs(DATAHOME / 'raw' / get_site_id(site)), exist_ok=True)

    #  no sites have monthly before 2015
    for year in range(2015, 2021):
        for month in range(1, 13):
            year = int(year)
            month = int(month)
            dt = datetime.datetime(year, month, 1)
            dt = datetime.datetime(year, month, 1)
            month = dt.strftime('%m')
            year = str(year)
            scrape(site, year=year, month=month)


if __name__ == '__main__':
    pubs = [
        'medium.com/hackernoon',
        'medium.com/free-code-camp',
        'towardsdatascience.com',
        'medium.com/the-mission',
        'medium.com/personal-growth',
        'medium.com/swlh',
        'medium.com/level-up-web',
        'medium.com/better-programming',
        'medium.com/better-humans',
        'medium.com/dailyjs',
        'levelup.gitconnected.com',
        'writingcooperative.com',
        'medium.com/the-ascent',
        'medium.com/javascript-in-plain-english',
        'levelup.gitconnected.com',
        'change-your-mind',
        'analytics-vidhya',
        'better-marketing',
        'matter',
        'startup-grind'
    ]
    for pub in pubs:
        scrape_medium_publication(pub)

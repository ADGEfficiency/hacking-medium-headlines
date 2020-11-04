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


def scrape_all_archive(site):
    print(f'scraping all archive')
    url = f'https://{site}/archive'
    divs = request_and_parse(url)

    #  should be 10 articles in the all years archive
    #  monthly / yearly archives have variable number
    #assert len(divs) == 10 # 9 for freecode camp!

    dataset = extract_article_data(divs, get_site_id(site))
    #assert len(dataset) == 10 as above
    save_article_data(dataset, site, 'all')


def request_and_parse(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html.parser")
    return soup.findAll('div', {'class': 'streamItem streamItem--postPreview js-streamItem'})


def scrape_year(site, year):
    year = str(year)
    print(f'scraping {year}')
    url = f'https://{site}/archive{year}'
    if not check_history(url):
        print(f'{url} failed - returning early')
        return None
    else:
        divs = request_and_parse(url)
        dataset = extract_article_data(divs, get_site_id(site))
        save_article_data(dataset, site, year)


def scrape_year_month(site, year, month):

    dt = datetime.datetime(year, month, 1)
    dt = datetime.datetime(year, month, 1)
    month = dt.strftime('%m')
    year = str(year)
    print(f'scraping {year}-{month}')

    url = f'https://{site}/archive/{year}/{month}'
    if not check_history(url):
        print(f'{url} failed - returning early')
        return None
    else:
        divs = request_and_parse(url)
        dataset = extract_article_data(divs, get_site_id(site))
        save_article_data(dataset, site, f'{year}-{month}')


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
        print(f'scraping {site}')
        scrape_all_archive(site)
        for year in range(2010, 2021):
            scrape_year(site, year)

            #  no sites have monthly before 2015
            if year >= 2015:
                for month in range(1, 13):
                    scrape_year_month(site, year, month)

    # scrape_medium_publication('medium.com/free-code-camp')
    # scrape_medium_publication('towardsdatascience.com')
    scrape_medium_publication('medium.com/the-mission')
    scrape_medium_publication('medium.com/hacker-daily')
    scrape_medium_publication('medium.com/personal-growth')
    scrape_medium_publication('medium.com/swlh')

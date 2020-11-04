import pytest
from archive import check_history, clean_claps


def test_clean_claps():
    assert clean_claps('10.6K') == 10600
    assert clean_claps('10') == 10


@pytest.mark.parametrize(
    'site, year, month, expected',
    [
        ['towardsdatascience', '2010', '01', False],
        ['towardsdatascience', '2015', '01', False],
        ['towardsdatascience', '2015', '03', True],
        ['towardsdatascience', '2016', '01', False],
        ['towardsdatascience', '2019', '12', True],
    ]
)
def test_request_history_check(site, year, month, expected):
    url = f'https://{site}.com/archive/{year}/{month}'
    assert check_history(url) == expected

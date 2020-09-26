import os
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup as bs


class Scraper:

    def __init__(self):
        self.years = list(range(2015, 2021))
        self.yearly_data = []
        self.output_path = "/Users/administrator/PycharmProjects/NFL_O_U/output"

    def scrape(self):
        base_url = "https://www.pro-football-reference.com/years/*YEARS*/games.htm"
        for curr_year in self.years:
            rows = []
            html_path = base_url.replace("*YEARS*", str(curr_year))
            url = urlopen(html_path)
            soup = bs(url, 'lxml')
            table = soup.find("table", {"class": "sortable stats_table"})
            for row in table.findAll("tr"):
                th = row.find_all('th')
                td = row.find_all('td')
                week_text = [t.text for t in th]
                row_text = [tr.text for tr in td]
                rows.append(week_text + row_text)
            columns = rows[0]
            rows.remove(columns)
            df = pd.DataFrame(rows, columns=columns)
            df["year"] = curr_year
            self.yearly_data.append(df)

        return self.yearly_data

    def save(self):
        curr_year = 2015
        for df in self.yearly_data:
            results_csv = os.path.join(self.output_path, "nfl_results_" + str(curr_year) + ".csv")
            df.to_csv(results_csv, index=False)
            curr_year += 1


if __name__ == '__main__':
    s = Scraper()
    s.scrape()
    s.save()

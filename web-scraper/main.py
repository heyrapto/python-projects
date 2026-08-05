from urllib.request import urlopen

from bs4 import BeautifulSoup

# https://calebships.com


def scrape(url: str):
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    print(soup.find("h1").string)
    return


url = input("Enter a valid url: ")
scrape(url)

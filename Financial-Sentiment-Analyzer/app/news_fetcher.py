import requests
from bs4 import BeautifulSoup

def get_financial_news(limit=5):
    url = "https://www.moneycontrol.com/news/business/"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    # Correct selector for MoneyControl news headlines
    articles = soup.select("li.clearfix h2")[:limit]


    news_list = []
    for a in articles:
        title = a.get_text(strip=True)

        # Filter out menu/header junk
        if (
            title 
            and "Login" not in title 
            and "English" not in title 
            and "Hindi" not in title 
            and "Gujarati" not in title
            and "Specials" not in title
        ):
            news_list.append(title)

    return news_list


print(get_financial_news(10))
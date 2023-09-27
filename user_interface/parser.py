from bs4 import BeautifulSoup

def get_fun_facts():
    fun_facts = []
    with open("fun_facts.html", 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = soup.find_all('div', class_='single-title-desc-wrap', attrs={'layout': 'title_&_description'})
    fun_facts = []
    for element in elements:
        text = element.get_text(strip=True)
        fun_facts.append(text)
    return fun_facts

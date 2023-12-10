# web scraping utility inspired by https://github.com/theDeepanshuMourya/Ikea-WebScraping-Classification/blob/master/LICENSE
# to query IKEA's website for downloading images related to decor
# This script was modified by Jacob Hammond on 12/01/2023 to work with the current website with modifications to url classes and 
# layer searching

#import necessary libraries
from bs4 import BeautifulSoup
import requests
import os
import urllib.request


website = 'https://www.ikea.com/us/en/cat/dinnerware-18860/'
data_dir = 'dinnerware'
page = requests.get(website)
soup = BeautifulSoup(page.content, 'html.parser')
soup =  soup.find('div', attrs={'class': 'vn__wrapper'})

# find all links of the subcategories
#all_items_outer_div = soup.find('div', attrs={'class' : 'plp-revamp-product-list__product'} )
inner_items_data = soup.find_all('a')

# save product name and it's link in a list 
product_data = []
for item in inner_items_data:
    product_name = item.text.strip()
    product_link = item.attrs['href']
    print(product_name,product_link)
    product_data.append([product_name, product_link])

''' 
Download images and save them into their respective directory
    Input: List containing names of subcategory and their page link
    Output: List containing names and link of subcategory which failed to download
'''
def download_images(product_data):
    skipped_data = []
    for product_name, product_link in product_data:
        product_dir = os.path.join(data_dir,product_name)
        if not os.path.exists(product_dir):
            os.makedirs(product_dir)
        next_page_available = True
        counter = 0
        pages_to_scrap = 3
        while pages_to_scrap:
            try:
                product_page = requests.get(product_link)
                soup = BeautifulSoup(product_page.text, 'html.parser')
                next_page_link = soup.find('a', {'class' : 'plp-btn plp-btn--small plp-btn--secondary' })
                soup =  soup.find("div", {"class": "plp-product-list__products"})
                images_data = soup.findAll('img')
                for image_data in images_data:
                    img_link = image_data.attrs['src'][:image_data.attrs['src'].rfind('?')] + '?f=xxxs'
                    urllib.request.urlretrieve(img_link, product_dir + '/' + str(counter) + '.jpg')
                    counter += 1
                if next_page_link is None:
                    break
                product_link = next_page_link.attrs['href']
                pages_to_scrap -= 1
            except Exception as e:
                print(e)
                skipped_data.append( [product_name,product_link] )
                break
        print('Total {} {} images downloaded'.format(counter,product_name))
    return skipped_data

skipped_data = download_images(product_data)
'''
check for skipped data. Sometimes our selected category contains a category which has sub categories 
inside it. i.e. it has separate page for it.
'''
skipped = None
for product_name, product_link in skipped_data:
    if 'www.ikea.com' not in product_link:
        product_page = requests.get('https://www.ikea.in/' + product_link)
        soup = BeautifulSoup(product_page.text, 'html.parser')
        soup =  soup.find("div", {"class": "product_cat_gallery"})
        all_items_outer_div = soup.find("div", {"class" : "row justify-content-center"} )
        inner_items_data = all_items_outer_div.find_all('a')
        product_data = []
        for item in inner_items_data:
            name = item.text.strip()
            link = item.attrs['href']
            print(name,link)
            product_data.append([name, link])
        skipped = download_images(product_data)
print(skipped)
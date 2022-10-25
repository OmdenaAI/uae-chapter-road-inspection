# import packages
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import FlickrImageCrawler
from datetime import date

# "scraping images from Google search engine"
# create folder to contain scraped Google images
crawler = GoogleImageCrawler(storage={'root_dir': 'pictures_Google'})
filters = dict(type='photo',license='commercial')
crawler.crawl(keyword="bad asphelt road", max_num=100, min_size = (300,300), 
filters=filters, file_idx_offset='auto')


# "scraping images from Flickr search engine"
# after creating non-commercial API key
# create folder to contain scraped Flickr images
flickr_crawler = FlickrImageCrawler('834f310e17f477be329ab57b4b1a729d',
                                    storage={'root_dir': 'pictures_Flickr'})
flickr_crawler.crawl(max_num=100, tags=['crack_road'], tag_mode = "any",
                    size_preference  = ['medium','medium 640','medium 800'])
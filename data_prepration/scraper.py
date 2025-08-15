import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
import re
import json
import logging
import os
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

class AIMattersSpider(scrapy.Spider):
    name = "ai_matters"
    allowed_domains = ["ai-matters.eu"]
    start_urls = ["https://ai-matters.eu/"]
    visited_urls = set()

    custom_settings = {
        # --- Logging setup ---
        'LOG_FILE': os.path.join('data', 'ai_matters.log'),
        'LOG_FORMAT': '%(asctime)s %(levelname)s: %(message)s',
        'LOG_DATEFORMAT': '%Y-%m-%d %H:%M:%S',
        'LOG_LEVEL': 'INFO',

        # --- Retry & throttle ---
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 0.5,
        'AUTOTHROTTLE_ENABLED': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 4,
        'RETRY_TIMES': 5,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 429, 403],
        'HTTPCACHE_ENABLED': True,

        # --- FEEDS ---
        'FEEDS': {
            os.path.join('data', 'ai_matters_data.jsonl'): {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
            }
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        # Open the failed-links file
        self.failed_file = open(os.path.join('data', 'failed_links.jsonl'), 'a', encoding='utf-8')
        self.logger.info("Opened data/failed_links.jsonl for appending")

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                errback=self.errback,
                dont_filter=True
            )

    def parse(self, response):
        # Log success
        self.logger.info(f"SCRAPED (200): {response.url}")

        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore').lower()
        if 'text/html' not in content_type:
            return

        title = response.css('title::text').get('').strip()
        content = self.extract_content(response)

        if title or content:
            yield {
                "url": response.url,
                "title": title,
                "content": content
            }

        for link in response.css('a::attr(href)').getall():
            next_url = response.urljoin(link)
            if self.should_follow(next_url):
                yield scrapy.Request(
                    next_url,
                    callback=self.parse,
                    errback=self.errback
                )

    def errback(self, failure):
        # Extract the URL that failed
        request = failure.request
        url = request.url

        # Determine failure reason
        if failure.check(DNSLookupError):
            reason = "DNS lookup failed"
        elif failure.check(TimeoutError, TCPTimedOutError):
            reason = "Timeout"
        else:
            reason = failure.getErrorMessage()

        # Log error to console/log
        self.logger.error(f"FAILED ({reason}): {url}")

        # Write to failed_links.jsonl
        record = {"url": url, "reason": reason}
        self.failed_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self, reason):
        # Close the failed-links file cleanly
        self.failed_file.close()
        self.logger.info("Closed data/failed_links.jsonl")
        super().close(self, reason)

    def should_follow(self, url):
        parsed = urlparse(url)
        if parsed.netloc not in self.allowed_domains:
            return False

        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.docx']):
            return False

        if url in self.visited_urls:
            return False

        self.visited_urls.add(url)
        return True

    def extract_content(self, response):
        """Comprehensive content extraction for AI Matters website"""
        content_selectors = [
            'main', 'article', '.main-content', '.content',
            '.page-content', '#content', '.entry-content',
            'section', '.container', '.text-block'
        ]

        content_areas = []
        for selector in content_selectors:
            areas = response.css(selector)
            if areas:
                content_areas.extend(areas)

        if not content_areas:
            content_areas = [response.css('body')]

        all_text_parts = []
        for area in content_areas:
            text_nodes = area.xpath('''
                .//text()[not(
                    ancestor::script or
                    ancestor::style or
                    ancestor::nav[contains(@class, 'navigation') or contains(@class, 'menu')] or
                    ancestor::header[contains(@class, 'site-header')] or
                    ancestor::footer or
                    ancestor::*[contains(@class, 'cookie')] or
                    ancestor::*[contains(@class, 'advertisement')] or
                    ancestor::*[contains(@class, 'social-media')]
                )]
            ''').getall()

            structured_content = area.css(
                'h1, h2, h3, h4, h5, h6, p, li, td, th, dt, dd, blockquote, div::text, span::text'
            ).getall()

            all_text_parts.extend(text_nodes)
            all_text_parts.extend(structured_content)

        # Clean and process text
        processed_parts = []
        for text in all_text_parts:
            if isinstance(text, str):
                cleaned = re.sub(r'\s+', ' ', text.strip())
                cleaned = re.sub(r'[\x00-\x1F]+', ' ', cleaned)
                cleaned = re.sub(r'[^\w\s\.,;:!?\-()[\]{}"\'/]', ' ', cleaned)
                if len(cleaned) > 2 and not cleaned.isspace():
                    noise_patterns = [
                        r'^(skip to|jump to|go to)',
                        r'^(menu|navigation|nav)',
                        r'^(cookies?|privacy)',
                        r'^\s*[\d\s\-\./]+\s*$',
                        r'^(back|next|previous|more)$'
                    ]
                    if not any(re.match(p, cleaned.lower()) for p in noise_patterns):
                        processed_parts.append(cleaned)

        final_content = ' '.join(processed_parts)
        final_content = re.sub(r'\s+', ' ', final_content)
        final_content = re.sub(r'(\s*[.!?]\s*){2,}', '. ', final_content)
        return final_content.strip()


if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    process = CrawlerProcess()
    process.crawl(AIMattersSpider)
    process.start()
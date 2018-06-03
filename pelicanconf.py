#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Cihan Soylu'
SITENAME = "Cihan Soylu's Blog"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Paris'

MARKUP = ('md', 'ipynb')

PLUGIN_PATHS = ['./plugins']
PLUGINS = ['ipynb.markup']

OUTPUT_PATH = 'docs/'

GITHUB_URL = 'https://github.com/CihanSoylu'
LINKEDIN_URL = 'https://www.linkedin.com/in/cihan-soylu-749b9088/'

THEME = 'themes/pelican-clean-blog'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('Github', 'https://github.com/CihanSoylu'),
          ('Linkedin', 'https://www.linkedin.com/in/cihan-soylu-749b9088/'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

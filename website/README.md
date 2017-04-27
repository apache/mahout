# Mahout Instructions

![Mahout Logo]({{ ASSET_BASE }}/img/mahout-logo.png)

## How do I edit Mahouts website? Jekyll

The Mahout website uses Jekyll to manage its website.  Refer to this page to setup your laptop with jekyll if you want to setup a local environment.
![Jekyll Setup](https://scotch.io/tutorials/getting-started-with-jekyll-plus-a-free-bootstrap-3-starter-theme)

Once you have Jekyll installed, next you will need to clone the mahout git repo
```
git clone https://github.com/apache/mahout.git mahout
cd website
```


## There are actually two sites!

#### `website/front`

This has information about the community, github, how to update website, developer resources, list of members. __*Things that when they change change forever and we don't need to keep a history of the old way (e.g. the old way of updating the website )*__

#### `website/docs`

This has user documentation, info on algorithms, programing guides, features. etc. 
__*things that change between versions*__

Follow the instructions below to serve either site, just know your links in community aren't going to work with the docs. Until you post it.

## Getting Started

To start editing the website first you need to open two terminals.  One terminal will run a continuous build of the mahout website locally, and the other will serve the website on localhost:4000

Terminal
```
bundle exec jekyll serve
```

Browser
```
localhost:4000
```

Start coding.



## Organization
website/_site   : this directory holds your static website.  don't modify anything in here directly!
website/_pages  : this directory holds most of the website content (more documentation to come)
website/_layouts  : this directory holds the basic layouts for page types

website/assets  : this directory holds the css and images used for the website


#### Themes

With Jekyll Builder we can easily swap out themes.  Currently the theme is `mahout`
`website/_includes/themes/mahout` : This directory has HTML for things you will include in your pages, e.g. navbar
`website/assets/themes/mahout`  : this directory holds the css and images used for the website (on the mahout theme, if we also may build new themes to try different looks and feels)

## How to port a page 

#### Copy to it's new home

There should be no chnages to `/website/docs/0.13.0/`  Need to create `/website/docs/0.13.1-SNAPSHOT/`

Other non-docs, e.g. things that go in developers or other static places, should go to where ever it is they are supposed to live.

If appropriate, change the file suffix from `.mdtext` or whatever wierd thing it is to `.md`

#### Change the Header

Most of the old stuff has a title like:

`Title: GSOC`

Change this too:

`---`
```
layout: default
title: GSoC
theme: mahout
---
```

#### Update PATHs where appropriate

Change hard linkes or relative links like this
```
For more information see [Handling GitHub PRs](http://mahout.apache.org/developers/github.html)
```

To utilize JB's `{{ BASE_PATH }}`

E.g. 
```
For more information see [Handling GitHub PRs]({{ BASE_PATH }}/developers/github.html)
```

This will make links in say github, refer to the github links. Same with images. 



### Changing themes

` rake theme:switch name="THEME-NAME"`

Options currently are `mahout` and `mahout2`

Mahout

![{{ BASE_PATH }}/img/mahout_theme.png]


Mahout2

![{{ BASE_PATH }}/img/mahout2_theme.png]

If you want to edit the style edit `assets/themes/<your_theme>/css/style.css` and override value.

This is a helpful tool for reference http://pikock.github.io/bootstrap-magic/3.0/app/index.html#!/editor


## Pressing ToDos for Reboot

- [ ] Fill out todo list
- [x] "flatten" everything (we shouldn't have a docs folder)
- [x] Port Distributed Linear Algebra pages
- [ ] Write up pages for existing 'pre canned algos'
- [ ] New themes require a lot less in `_imports` than dustin had initially, however- keeping his old files for now as they have lots of useful things that occastionally need to be scuttled. Eventaully need to delete though.
- [x] Clean up tiles on front page
- [ ] Add apache licenses all over the place - let dust settle first then we can go through methodically.
- [ ] Significant clean up of `/community/mahout-wiki.md`
- [ ] Update of `/community/powered-by-mahout.md`
- [ ] Folks need to review their contact info in `community/professional-support.md`
- [ ] Get rid of `developer/patch-check-list.md` and add it to the notes as a checkbox when opening a PR (see zeppelin)
- [ ] `developer/release-notes.md` stuck on 0.12.0... bump it. 
- [x] refactor to 'top-site' and 'docs' as we need a different jekyll build to change base path for new docs version
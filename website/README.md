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


## There are actually three sites!

#### `website/front`

This has information about the community, github, how to update website, developer resources, list of members. __*Things that when they change change forever and we don't need to keep a history of the old way (e.g. the old way of updating the website )*__

#### `website/docs`

This has user documentation, info on algorithms, programing guides, features. etc. 
__*things that change between versions*__

Follow the instructions below to serve either site, just know your links in community aren't going to work with the docs. Until you post it.

#### `website/old-site`

This is a full and mostly working port of the old site. It has been updated so that new themes can be applied to it, or the `mahout-retro` theme can be applied to the newer sites (thought this is somewhat messy).

There is a lot of content in here, and a lot of it isn't even available from the main site. We should start going through this and dragging it over page by page, updating as we go.

Eventually we'll use this to build `docs/0.13.0`

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

Within `mahout/website/docs` and `mahout/website/front` you'll find the following directories

- `./_site`   : this directory holds your static website which is created with `jekyll build`. Modifying things in here directly will have no effect (they will be overwritten)
- `./_layouts`  : this directory holds the basic layouts for page types
- `./_includes`  : this directory holds files which can be included in layouts, such as the HTML that powers the 
page, the navbar, etc. You will see in the html files referenced in `./_layouts` a line that says ` {{ content }}` this 
 is where the markdown compiled into HTML is injected. 
- `./assets`    : this directory holds the css and images used for the website
- `./[OTHER]`   : all other directories become directory structure of the site. They maybe filled with markdown files. E.g.
`./my-dir/myfile.md` will populate the site at `http://.../my-dir/myfile.html`

**NOTE** `_includes/` and `_assets/` are actually symlinks so that if you change the theme, it will apply evenly to all sites.

#### Themes

With Jekyll Builder we can easily swap out themes.  Currently the theme is `mahout3`
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

Options currently are `mahout`, `mahout2`, `mahout3` and `mahout-retro`

Mahout

![assets/img/mahout_theme.png]


Mahout2

![assets/img/mahout2_theme.png]

If you want to edit the style edit `assets/themes/<your_theme>/css/style.css` and override value.

This is a helpful tool for reference http://pikock.github.io/bootstrap-magic/3.0/app/index.html#!/editor













 










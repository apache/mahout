# Mahout Instructions

![Mahout](https://apache.mahout.org)

## How do I edit Mahouts website? Jekyll

The Mahout website uses Jekyll to manage its website.  Refer to this page to setup your laptop with jekyll if you want to setup a local environment.
![Jekyll Setup](https://scotch.io/tutorials/getting-started-with-jekyll-plus-a-free-bootstrap-3-starter-theme)

Once you have Jekyll installed, next you will need to clone the mahout git repo
```
git clone https://github.com/apache/mahout.git mahout
cd website
```

## Getting Started

To start editing the website first you need to open two terminals.  One terminal will run a continuous build of the mahout website locally, and the other will serve the website on localhost:4000


```
bundle exec jekyll build --safe
```


Browser
```
localhost:4000
```

Start coding.



## Organization
website/_site   : this directory holds your static website.  don't modify anything in here directly!
website/_pages  : this directory holds most of the website content (more documentation to come)
website/_posts  : this directory holds the posts that populate the homepage ..

website/assets  : this directory holds the css and images used for the website

## How to port a page .. TBD







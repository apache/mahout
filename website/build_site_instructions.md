# Building the Website

## Prerequisites
- Ensure Ruby is installed on your system.
- Install Bundler by running `gem install bundler`.

## Building the Site
1. Navigate to the website directory:
   ```
   cd website
   ```
2. Install required gems using Bundler:
   ```
   bundle install
   ```
3. Build the site:
   ```
   ./build_site.sh
   ```

This will generate the static site content which can be served using any standard web server.

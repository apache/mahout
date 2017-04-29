

## Website Migration Triage


### 1. `./old-site`

Original Mahout site was transferred to `mahout/website/oldsite` where it was
headers were replaced to be Jekyll complient as well as some witch craft on the
nav-bar to make the CSS compatible with the Jekyll Boot Strap Themes

All content was then moved to `mahout/website/old_site_migration/old_site`

ALCON please go through files and move them to one of the following directories

### 2a. `./dont_migrate` 

Content that is no longer relevant or is in such bad shape that needs to be redone completely goes here

### 2b. `./needs_work_convenience`

Content that should be migrated but needs updated with new information, or other work. Please leave a note
in the top of what needs to be done. This content can be migrated at convenience, e.g. is interesting and 
would be good to bring over, but is not critical (site can go live with out this content).

`./needs_work_convenience/map_reduce` has mapReduce related docs that may not actually need any work.

### 2c. `./needs_work_priority`

Content that should be migrated but needs updated.  This is critical information that needs to be migrated
before site goes live. 



### 3. `./completed`

When a file doesn't need work OR the work has been done on it- move a copy here, AND move a copy to the appropriate
location in `mahout/website/front` or `mahout/website/docs`

(don't forget to add page to nav-bar)




### New Module Structure

```
mahout
- core (contains former mahout-math, mahout-math-scala)
- engines
\- hdfs support
\- abstract (?)
\- spark
\- tensorflow(?)
- community (not tested/built under normal build conditions)
\- engines
\-- flink-batch
\-- h2o
\-- spark_1.6(?)
\- mr
\- Spark CLI Drivers
- Experimental
\- ViennaCL
\- ViennaCL-OMP
- examples (not implemented yet)
```



#### Todo

-[x] get `org.apache.mahout.math.drm` squared away
-[ ] see `core/pom.xml`
-[x] move tests for `org.apache.mahout.math.solver`s
-[x] move `org.apache.mahout.math.stats` esp OnlineSolver (but prob all of them)
-[x] move above's tests over.
-[x] IO Tests move over
-[x] Add drivers to Community (to avoid future `scopt` hold ups)
-[x] update all poms to dump files in `lib/`
-[ ] add pludin to delete everything from `lib/` on clean.
-[ ] Move MR to community engines. (Failling on OpenIntHash, etc.)
-[ ] Figure out where mising classes are ^^ OpenIntIntHash, etc.
-[x] failing tests on Naivebayes (needs 'Online Summarizer')
-[ ] Add `examples` module
-[x] ViennaCL to `experimental`
-[ ] Update `.travis.yml` for new module structures
-[ ] Add profiles back in.
-[x] Fix POMs (correct heirarcy / inheritance) (for now is done...)
-[ ] Fix POMs (add required plugins for release, see below)
-[ ] Clean up, delete directories no longer in use.
-[ ] Create profile to create spark-fat-jar. (of if you get pushback to make no fat jars)
-[ ] Move Kryo to top pom (spark, core and flink at least)
-[ ] Inspect all warnings
-[ ] Update Website 
-[ ] - Description of modules
-[ ] - Available profiles and what they do
-[ ] Update bin/mahout (probably moving most of it to mr-classic)
-[x] Add licenes to files
-[ ] Last thing- delete this file. 


### Plugins to add in
-[ ] Release
-[x] Ratcheck
-[ ] Checkstyle
-[ ] Maven-enforcer (Java 1.8 bump for spark 2.3+ compatability)
-[ ] Maven Surefire
-[ ] JavaDoc/Scala Doc Plugin

-[ ] profile for fat jars (spark/flink/h2o)
-[ ] profile to turn on flink / h2o / other non-essentials (then disable them in standard build)

### Current profiles
`mahout-mr` - builds the MapReduce stuff.
`apache-release` - contains the release plugin
`mahout_keys` - a profile used for releasing (actually lives in ~/.m2/settings.xml, see release instructions)
`flink-batch` - build flink batch community engine
`h2o` - build h2o community engine

add note in how to release about calling the tag `mahout-0.X.Y-rcZ`

### Release rollback notes

`mvn --batch-mode release:update-versions -DdevelopmentVersion=0.14.0-SNAPSHOT`

```bash
mvn -Papache-release release:rollback

mvn -Papache-release release:clean
```
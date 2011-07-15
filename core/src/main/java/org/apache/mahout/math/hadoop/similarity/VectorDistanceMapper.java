package org.apache.mahout.math.hadoop.similarity;


import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 *
 *
 **/
public class VectorDistanceMapper extends Mapper<WritableComparable<?>, VectorWritable, StringTuple, DoubleWritable> {
  private transient static Logger log = LoggerFactory.getLogger(VectorDistanceMapper.class);
  protected DistanceMeasure measure;
  protected List<NamedVector> seedVectors;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context) throws IOException, InterruptedException {
    String keyName;
    Vector valVec = value.get();
    if (valVec instanceof NamedVector) {
      keyName = ((NamedVector) valVec).getName();
    } else {
      keyName = key.toString();
    }
    for (NamedVector seedVector : seedVectors) {
      double distance = measure.distance(seedVector, valVec);
      StringTuple outKey = new StringTuple();
      outKey.add(seedVector.getName());
      outKey.add(keyName);
      context.write(outKey, new DoubleWritable(distance));
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      measure = ccl.loadClass(conf.get(VectorDistanceSimilarityJob.DISTANCE_MEASURE_KEY))
              .asSubclass(DistanceMeasure.class).newInstance();
      measure.configure(conf);


      String seedPathStr = conf.get(VectorDistanceSimilarityJob.SEEDS_PATH_KEY);
      if (seedPathStr != null && seedPathStr.length() > 0) {

        Path thePath = new Path(seedPathStr, "*");
        Collection<Path> result = Lists.newArrayList();

        // get all filtered file names in result list
        FileSystem fs = thePath.getFileSystem(conf);
        FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(thePath, PathFilters.partFilter())),
                PathFilters.partFilter());

        for (FileStatus match : matches) {
          result.add(fs.makeQualified(match.getPath()));
        }
        seedVectors = new ArrayList<NamedVector>(100);
        long item = 0;
        for (Path seedPath : result) {
          for (Writable value : new SequenceFileValueIterable<Writable>(seedPath, conf)) {
            Class<? extends Writable> valueClass = value.getClass();
            if (valueClass.equals(Cluster.class)) {
              // get the cluster info
              Cluster cluster = (Cluster) value;
              Vector vector = cluster.getCenter();
              if (vector instanceof NamedVector) {
                seedVectors.add((NamedVector) vector);
              } else {
                seedVectors.add(new NamedVector(vector, cluster.getIdentifier()));
              }
            } else if (valueClass.equals(Canopy.class)) {
              // get the cluster info
              Canopy canopy = (Canopy) value;
              Vector vector = canopy.getCenter();
              if (vector instanceof NamedVector) {
                seedVectors.add((NamedVector) vector);
              } else {
                seedVectors.add(new NamedVector(vector, canopy.getIdentifier()));
              }
            } else if (valueClass.equals(Vector.class)) {
              Vector vector = (Vector) value;
              if (vector instanceof NamedVector) {
                seedVectors.add((NamedVector) vector);
              } else {
                seedVectors.add(new NamedVector(vector, seedPath + "." + item++));
              }
            } else if (valueClass.equals(VectorWritable.class) || valueClass.isInstance(VectorWritable.class)) {
              VectorWritable vw = (VectorWritable) value;
              Vector vector = vw.get();
              if (vector instanceof NamedVector) {
                seedVectors.add((NamedVector) vector);
              } else {
                seedVectors.add(new NamedVector(vector, seedPath + "." + item++));
              }
            } else {
              throw new IllegalStateException("Bad value class: " + valueClass);
            }
          }
        }
        if (seedVectors.isEmpty()) {
          throw new IllegalStateException("No seeds found. Check your path: " + seedPathStr);
        } else {
          log.info("Seed Vectors size: " + seedVectors.size());
        }
      }
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
  }
}

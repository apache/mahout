package org.apache.mahout.clustering;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.cdbw.CDbwEvaluator;
import org.apache.mahout.clustering.evaluation.ClusterEvaluator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.junit.Test;

public class MAHOUT1045Test {
  
  @Test
  public void testClusterEvaluator() {
    Configuration conf = new Configuration();
    conf.set(RepresentativePointsDriver.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.CosineDistanceMeasure");
    conf.set(RepresentativePointsDriver.STATE_IN_KEY, "/Users/jeff/Desktop/jeff/representative/representativePoints-5");
    ClusterEvaluator ce = new ClusterEvaluator(conf, new Path(
        "/Users/jeff/Desktop/jeff/kmeans-clusters/clusters-27-final"));
    double interClusterDensity = ce.interClusterDensity();
    double intraClusterDensity = ce.intraClusterDensity();
    System.out.println("Inter-cluster Density = " + interClusterDensity);
    System.out.println("Intra-cluster Density = " + intraClusterDensity);
  }
  
  @Test
  public void testCDbwEvaluator() {
    Configuration conf = new Configuration();
    conf.set(RepresentativePointsDriver.DISTANCE_MEASURE_KEY, "org.apache.mahout.common.distance.CosineDistanceMeasure");
    conf.set(RepresentativePointsDriver.STATE_IN_KEY, "/Users/jeff/Desktop/jeff/representative/representativePoints-5");
    CDbwEvaluator cd = new CDbwEvaluator(conf, new Path("/Users/jeff/Desktop/jeff/kmeans-clusters/clusters-27-final"));
    double cdInterClusterDensity = cd.interClusterDensity();
    double cdIntraClusterDensity = cd.intraClusterDensity();
    double cdSeparation = cd.separation();
    double cdbw = cd.getCDbw();
    System.out.println("CDbw Inter-cluster Density = " + cdInterClusterDensity);
    System.out.println("CDbw Intra-cluster Density = " + cdIntraClusterDensity);
    System.out.println("CDbw Separation = " + cdSeparation);
    System.out.println("CDbw = " + cdbw);
  }
  
}

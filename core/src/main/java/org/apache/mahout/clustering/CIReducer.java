package org.apache.mahout.clustering;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.Cluster;

public class CIReducer extends Reducer<IntWritable,Cluster,IntWritable,Cluster> {
  
  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.mapreduce.Reducer#reduce(java.lang.Object,
   * java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void reduce(IntWritable key, Iterable<Cluster> values,
      Context context) throws IOException, InterruptedException {
    Iterator<Cluster> iter =values.iterator();
    Cluster first = null;
    while(iter.hasNext()){
      Cluster cl = iter.next();
      if (first == null){
        first = cl;
      }
      else {
        first.observe(cl);
      }
    }
    first.computeParameters();
    context.write(key, first);
  }
  
}

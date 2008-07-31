package org.apache.mahout.ga.watchmaker.cd;

import junit.framework.TestCase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class DataLineTest extends TestCase {

  private static final String[] datalines = {
      "842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189",
      "8510426,B,13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259",
      "852781,M,18.61,20.25,122.1,1094,0.0944,0.1066,0.149,0.07731,0.1697,0.05699,0.8529,1.849,5.632,93.54,0.01075,0.02722,0.05081,0.01911,0.02293,0.004217,21.31,27.26,139.9,1403,0.1338,0.2117,0.3446,0.149,0.2341,0.07421" };

  public void testSet() throws Exception {
    FileSystem fs = FileSystem.get(new Configuration());
    Path inpath = new Path("build/examples-test-classes/wdbc");
    DataSet.initialize(FileInfoParser.parseFile(fs, inpath));
    
    DataLine dl = new DataLine();
    
    dl.set(datalines[0]);
    assertEquals(1, dl.getLabel());
    
    dl.set(datalines[1]);
    assertEquals(0, dl.getLabel());
    
    dl.set(datalines[2]);
    assertEquals(1, dl.getLabel());
  }

}

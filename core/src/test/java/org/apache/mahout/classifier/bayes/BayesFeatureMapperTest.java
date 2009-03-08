package org.apache.mahout.classifier.bayes;

import junit.framework.TestCase;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.classifier.bayes.common.BayesFeatureMapper;
import org.apache.mahout.utils.DummyOutputCollector;

import java.util.List;
import java.util.Map;

public class BayesFeatureMapperTest extends TestCase {

  public void test() throws Exception {
    BayesFeatureMapper mapper = new BayesFeatureMapper();
    JobConf conf = new JobConf();
    conf.set("io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<Integer> intStringifier = new DefaultStringifier<Integer>(conf, Integer.class);
    conf.set("bayes.gramSize", intStringifier.toString(3));
    mapper.configure(conf);

    DummyOutputCollector<Text, DoubleWritable> output = new DummyOutputCollector<Text, DoubleWritable>();
    mapper.map(new Text("foo"), new Text("big brown shoe"), output, null);
    Map<String, List<DoubleWritable>> outMap = output.getData();
    System.out.println("Map: " + outMap);
    assertNotNull("outMap is null and it shouldn't be", outMap);
    //TODO: How about not such a lame test here?
    for (Map.Entry<String, List<DoubleWritable>> entry : outMap.entrySet()) {
      assertTrue("entry.getKey() Size: " + entry.getKey().length() + " is not greater than: 0", entry.getKey().length() > 0);
      assertEquals("entry.getValue() Size: " + entry.getValue().size() + " is not: 1", 1, entry.getValue().size());
      assertTrue("value is not valie", entry.getValue().get(0).get() > 0);
    }

  }

}

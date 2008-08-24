package org.apache.mahout.utils.parameters;

import org.apache.hadoop.mapred.JobConf;

public class ClassParameter extends AbstractParameter<Class> {

  public ClassParameter(String prefix, String name, JobConf jobConf, Class defaultValue, String description) {
    super(Class.class, prefix, name, jobConf, defaultValue, description);
  }

  public void setStringValue(String stringValue) {
    try {
      set(Class.forName(stringValue));
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  public String getStringValue() {
    if (value == null) {
      return null;
    }
    return get().getName();
  }
}

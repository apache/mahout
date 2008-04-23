package org.apache.mahout.utils.parameters;

import org.apache.hadoop.mapred.JobConf;

import java.util.Collection;

/**
 * A placeholder for some sort of class with more parameters.
 */
public class CompositeParameter<T extends Parametered> extends AbstractParameter<T> {

  public CompositeParameter(Class<T> type, String prefix, String name, JobConf jobConf, T defaultValue, String description) {
    super(type, prefix, name, jobConf, defaultValue, description);
  }

  public void createParameters(String prefix, JobConf jobConf) {
    get().createParameters(prefix, jobConf);
  }

  public Collection<Parameter> getParameters() {
    return get().getParameters();
  }


  public void configure(JobConf jobConf) {
    get().configure(jobConf);
  }

  @SuppressWarnings("unchecked")
  public void setStringValue(String className) {
    try {
      set((T) Class.forName(className).newInstance());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public String getStringValue() {
    if (value == null) {
      return null;
    }
    return value.getClass().getName();
  }
}

/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils.parameters;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobConfigurable;

import java.util.Collection;

/**
 * Meta information and accessors for configuring a job.
 */
public interface Parametered extends JobConfigurable {

  public static final Log log = LogFactory.getLog(Parametered.class);


  public abstract Collection<Parameter> getParameters();

  /**
   * EXPERT: consumers should never have to call this method. It would be friendly visible to {@link ParameteredGeneralizations} if java supported it.
   * Calling this method should create a new list of parameters and is called
   * @see ParameteredGeneralizations#configureParameters(String,Parametered,org.apache.hadoop.mapred.JobConf) invoking method
   * @see ParameteredGeneralizations#configureParametersRecusivly(Parametered,String,org.apache.hadoop.mapred.JobConf) invoking method
   * @param prefix ends with a dot if not empty.
   * @param jobConf configuration used for retreiving values
   */
  public abstract void createParameters(String prefix, JobConf jobConf);


  /**
   * "multiple inheritance"
   */
  public static class ParameteredGeneralizations {


    /**
     * @see #configureParameters(Parametered,org.apache.hadoop.mapred.JobConf)
     * Uses parametered simple class name as prefix.
     *
     * @param parametered instance to be configured
     * @param jobConf configuration used for retreiving values
     */
    public static void configureParameters(Parametered parametered, JobConf jobConf) {
      configureParameters(parametered.getClass().getSimpleName() + ".", parametered, jobConf);

    }

    /**
     * Calls {@link org.apache.mahout.utils.parameters.Parametered#createParameters(String,org.apache.hadoop.mapred.JobConf)}
     * on parameter parmetered,
     * and then recurse down its composite tree
     * to invoke {@link org.apache.mahout.utils.parameters.Parametered#createParameters(String,org.apache.hadoop.mapred.JobConf)}
     * and {@link org.apache.hadoop.mapred.JobConfigurable#configure(org.apache.hadoop.mapred.JobConf)}
     * on each composite part.
     *
     * @param prefix ends with a dot if not empty.
     * @param parametered instance to be configured
     * @param jobConf configuration used for retreiving values
     */
    public static void configureParameters(String prefix, Parametered parametered, JobConf jobConf) {
      parametered.createParameters(prefix, jobConf);
      configureParametersRecusivly(parametered, prefix, jobConf);
    }

    private static void configureParametersRecusivly(Parametered parametered, String prefix, JobConf jobConf) {
      for (Parameter parameter : parametered.getParameters()) {
        if (log.isDebugEnabled()) {
          log.debug("Configuring " + prefix + parameter.name());
        }
        String name = prefix + parameter.name() + ".";
        parameter.createParameters(name, jobConf);
        parameter.configure(jobConf);
        if (parameter.getParameters().size() > 0) {
          configureParametersRecusivly(parameter, name, jobConf);
        }
      }
    }

    public static String help(Parametered parametered) {
      return new Help(parametered).toString();
    }

    public static String conf(Parametered parametered) {
      return new Conf(parametered).toString();
    }

    private static class Help {
      private StringBuilder sb;

      public String toString() {
        return sb.toString();
      }

      private int longestName = 0;
      private int numChars = 100; // a few extra just to be sure

      int distanceBetweenNameAndDescription = 8; // todo: hmmm in the end this is 5 letters less that it says.. not sure why  

      private void recurseCount(Parametered parametered) {
        for (Parameter parameter : parametered.getParameters()) {
          int parameterNameLength = parameter.name().length();
          if (parameterNameLength > longestName) {
            longestName = parameterNameLength;
          }
          recurseCount(parameter);
          numChars += parameter.description().length();
        }
      }

      private Help(Parametered parametered) {

        recurseCount(parametered);

        numChars += (longestName + distanceBetweenNameAndDescription) * parametered.getParameters().size();
        sb = new StringBuilder(numChars);

        recurseWrite(parametered);

      }

      private void recurseWrite(Parametered parametered) {
        for (Parameter parameter : parametered.getParameters()) {
          sb.append(parameter.prefix());
          sb.append(parameter.name());
          for (int i = 0; i < longestName - parameter.name().length() - parameter.prefix().length() + distanceBetweenNameAndDescription; i++)
          {
            sb.append(' ');
          }
          sb.append(parameter.description());
          if (parameter.defaultValue() != null) {
            sb.append(" (default value '");
            sb.append(parameter.defaultValue());
            sb.append("')");
          }
          sb.append('\n');
          recurseWrite(parameter);
        }
      }
    }


    private static class Conf {
      private StringBuilder sb;

      public String toString() {
        return sb.toString();
      }

      private int longestName = 0;
      private int numChars = 100; // a few extra just to be sure

      int distanceBetweenNameAndDescription = 4;

      private void recurseCount(Parametered parametered) {
        for (Parameter parameter : parametered.getParameters()) {
          int parameterNameLength = parameter.prefix().length() + parameter.name().length();
          if (parameterNameLength > longestName) {
            longestName = parameterNameLength;
          }

          numChars += parameterNameLength;
          numChars += 5; // # $0\n$1 = $2\n\n
          numChars += parameter.description().length();
          if (parameter.getStringValue() != null) {
            numChars += parameter.getStringValue().length();
          }

          recurseCount(parameter);
        }
      }

      private Conf(Parametered parametered) {

        recurseCount(parametered);

        sb = new StringBuilder(numChars);

        recurseWrite(parametered);

      }

      private void recurseWrite(Parametered parametered) {
        for (Parameter parameter : parametered.getParameters()) {
          sb.append("# ");
          sb.append(parameter.description());
          sb.append('\n');
          sb.append(parameter.prefix());
          sb.append(parameter.name());
          sb.append(" = ");
          if (parameter.getStringValue() != null) {
            sb.append(parameter.getStringValue());
          }
          sb.append('\n');
          sb.append('\n');
          recurseWrite(parameter);
        }
      }
    }

  }


}

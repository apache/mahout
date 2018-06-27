/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.model.jdbc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.jdbc.AbstractJDBCComponent;

/**
 * <p>
 * A generic {@link org.apache.mahout.cf.taste.model.DataModel} designed for use with other JDBC data sources;
 * one just specifies all necessary SQL queries to the constructor here. Optionally, the queries can be
 * specified from a {@link Properties} object, {@link File}, or {@link InputStream}. This class is most
 * appropriate when other existing implementations of {@link AbstractJDBCDataModel} are not suitable. If you
 * are using this class to support a major database, consider contributing a specialized implementation of
 * {@link AbstractJDBCDataModel} to the project for this database.
 * </p>
 */
public final class GenericJDBCDataModel extends AbstractJDBCDataModel {
  
  public static final String DATA_SOURCE_KEY = "dataSource";
  public static final String GET_PREFERENCE_SQL_KEY = "getPreferenceSQL";
  public static final String GET_PREFERENCE_TIME_SQL_KEY = "getPreferenceTimeSQL";
  public static final String GET_USER_SQL_KEY = "getUserSQL";
  public static final String GET_ALL_USERS_SQL_KEY = "getAllUsersSQL";
  public static final String GET_NUM_USERS_SQL_KEY = "getNumUsersSQL";
  public static final String GET_NUM_ITEMS_SQL_KEY = "getNumItemsSQL";
  public static final String SET_PREFERENCE_SQL_KEY = "setPreferenceSQL";
  public static final String REMOVE_PREFERENCE_SQL_KEY = "removePreferenceSQL";
  public static final String GET_USERS_SQL_KEY = "getUsersSQL";
  public static final String GET_ITEMS_SQL_KEY = "getItemsSQL";
  public static final String GET_PREFS_FOR_ITEM_SQL_KEY = "getPrefsForItemSQL";
  public static final String GET_NUM_PREFERENCE_FOR_ITEM_KEY = "getNumPreferenceForItemSQL";
  public static final String GET_NUM_PREFERENCE_FOR_ITEMS_KEY = "getNumPreferenceForItemsSQL";
  public static final String GET_MAX_PREFERENCE_KEY = "getMaxPreferenceSQL";
  public static final String GET_MIN_PREFERENCE_KEY = "getMinPreferenceSQL";

  /**
   * <p>
   * Specifies all SQL queries in a {@link Properties} object. See the {@code *_KEY} constants in this
   * class (e.g. {@link #GET_USER_SQL_KEY}) for a list of all keys which must map to a value in this object.
   * </p>
   *
   * @param props
   *          {@link Properties} object containing values
   * @throws TasteException
   *           if anything goes wrong during initialization
   */
  public GenericJDBCDataModel(Properties props) throws TasteException {
    super(AbstractJDBCComponent.lookupDataSource(props.getProperty(DATA_SOURCE_KEY)),
          props.getProperty(GET_PREFERENCE_SQL_KEY),
          props.getProperty(GET_PREFERENCE_TIME_SQL_KEY),
          props.getProperty(GET_USER_SQL_KEY),
          props.getProperty(GET_ALL_USERS_SQL_KEY),
          props.getProperty(GET_NUM_ITEMS_SQL_KEY),
          props.getProperty(GET_NUM_USERS_SQL_KEY),
          props.getProperty(SET_PREFERENCE_SQL_KEY),
          props.getProperty(REMOVE_PREFERENCE_SQL_KEY),
          props.getProperty(GET_USERS_SQL_KEY),
          props.getProperty(GET_ITEMS_SQL_KEY),
          props.getProperty(GET_PREFS_FOR_ITEM_SQL_KEY),
          props.getProperty(GET_NUM_PREFERENCE_FOR_ITEM_KEY),
          props.getProperty(GET_NUM_PREFERENCE_FOR_ITEMS_KEY),
          props.getProperty(GET_MAX_PREFERENCE_KEY),
          props.getProperty(GET_MIN_PREFERENCE_KEY));
  }
  
  /**
   * <p>
   * See {@link #GenericJDBCDataModel(Properties)}. This constructor reads values from a file
   * instead, as if with {@link Properties#load(InputStream)}. So, the file should be in standard Java
   * properties file format -- containing {@code key=value} pairs, one per line.
   * </p>
   *
   * @param propertiesFile
   *          properties file
   * @throws TasteException
   *           if anything goes wrong during initialization
   */
  public GenericJDBCDataModel(File propertiesFile) throws TasteException {
    this(getPropertiesFromFile(propertiesFile));
  }
  
  /**
   * <p>
   * See {@link #GenericJDBCDataModel(Properties)}. This constructor reads values from a resource available in
   * the classpath, as if with {@link Class#getResourceAsStream(String)} and
   * {@link Properties#load(InputStream)}. This is useful if your configuration file is, for example, packaged
   * in a JAR file that is in the classpath.
   * </p>
   * 
   * @param resourcePath
   *          path to resource in classpath (e.g. "/com/foo/TasteSQLQueries.properties")
   * @throws TasteException
   *           if anything goes wrong during initialization
   */
  public GenericJDBCDataModel(String resourcePath) throws TasteException {
    this(getPropertiesFromStream(GenericJDBCDataModel.class
        .getResourceAsStream(resourcePath)));
  }
  
  private static Properties getPropertiesFromFile(File file) throws TasteException {
    try {
      return getPropertiesFromStream(new FileInputStream(file));
    } catch (FileNotFoundException fnfe) {
      throw new TasteException(fnfe);
    }
  }
  
  private static Properties getPropertiesFromStream(InputStream is) throws TasteException {
    try {
      try {
        Properties props = new Properties();
        props.load(is);
        return props;
      } finally {
        Closeables.close(is, true);
      }
    } catch (IOException ioe) {
      throw new TasteException(ioe);
    }
  }
  
}

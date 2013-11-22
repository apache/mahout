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

package org.apache.mahout.cf.taste.impl.model.mongodb;

import java.text.DateFormat;
import java.text.ParseException;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import java.net.UnknownHostException;
import java.text.SimpleDateFormat;
import java.util.regex.Pattern;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.NoSuchItemException;

import org.bson.types.ObjectId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import com.mongodb.Mongo;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;

/**
 * <p>A {@link DataModel} backed by a MongoDB database. This class expects a
 * collection in the database which contains a user ID ({@code long} or
 * {@link ObjectId}), item ID ({@code long} or
 * {@link ObjectId}), preference value (optional) and timestamps
 * ("created_at", "deleted_at").</p>
 *
 * <p>An example of a document in MongoDB:</p>
 *
 * <p>{@code { "_id" : ObjectId("4d7627bf6c7d47ade9fc7780"),
 * "user_id" : ObjectId("4c2209fef3924d31102bd84b"),
 * "item_id" : ObjectId(4c2209fef3924d31202bd853),
 * "preference" : 0.5,
 * "created_at" : "Tue Mar 23 2010 20:48:43 GMT-0400 (EDT)" }
 * }</p>
 *
 * <p>Preference value is optional to accommodate applications that have no notion
 * of a preference value (that is, the user simply expresses a preference for
 * an item, but no degree of preference).</p>
 *
 * <p>The preference value is assumed to be parseable as a {@code double}.</p>
 *
 * <p>The user IDs and item IDs are assumed to be parseable as {@code long}s
 * or {@link ObjectId}s. In case of {@link ObjectId}s, the
 * model creates a {@code Map<ObjectId>}, {@code long}>
 * (collection "mongo_data_model_map") inside the MongoDB database. This
 * conversion is needed since Mahout uses the long datatype to feed the
 * recommender, and MongoDB uses 12 bytes to create its identifiers.</p>
 *
 * <p>The timestamps ("created_at", "deleted_at"), if present, are assumed to be
 * parseable as a {@code long} or {@link Date}. To express
 * timestamps as {@link Date}s, a {@link DateFormat}
 * must be provided in the class constructor. The default Date format is
 * {@code "EE MMM dd yyyy HH:mm:ss 'GMT'Z (zzz)"}. If this parameter
 * is set to null, timestamps are assumed to be parseable as {@code long}s.
 * </p>
 *
 * <p>It is also acceptable for the documents to contain additional fields.
 * Those fields will be ignored.</p>
 *
 * <p>This class will reload data from the MondoDB database when
 * {@link #refresh(Collection)} is called. MongoDBDataModel keeps the
 * timestamp of the last update. This variable and the fields "created_at"
 * and "deleted_at" help the model to determine if the triple
 * (user, item, preference) must be added or deleted.</p>
 */
public final class MongoDBDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(MongoDBDataModel.class);

  /** Default MongoDB host. Default: localhost */
  private static final String DEFAULT_MONGO_HOST = "localhost";

  /** Default MongoDB port. Default: 27017 */
  private static final int DEFAULT_MONGO_PORT = 27017;

  /** Default MongoDB database. Default: recommender */
  private static final String DEFAULT_MONGO_DB = "recommender";

  /**
   * Default MongoDB authentication flag.
   * Default: false (authentication is not required)
   */
  private static final boolean DEFAULT_MONGO_AUTH = false;

  /** Default MongoDB user. Default: recommender */
  private static final String DEFAULT_MONGO_USERNAME = "recommender";

  /** Default MongoDB password. Default: recommender */
  private static final String DEFAULT_MONGO_PASSWORD = "recommender";

  /** Default MongoDB table/collection. Default: items */
  private static final String DEFAULT_MONGO_COLLECTION = "items";

  /**
   * Default MongoDB update flag. When this flag is activated, the
   * DataModel updates both model and database. Default: true
   */
  private static final boolean DEFAULT_MONGO_MANAGE = true;

  /** Default MongoDB user ID field. Default: user_id */
  private static final String DEFAULT_MONGO_USER_ID = "user_id";

  /** Default MongoDB item ID field. Default: item_id */
  private static final String DEFAULT_MONGO_ITEM_ID = "item_id";

  /** Default MongoDB preference value field. Default: preference */
  private static final String DEFAULT_MONGO_PREFERENCE = "preference";

  /** Default MongoDB final remove flag. Default: false */
  private static final boolean DEFAULT_MONGO_FINAL_REMOVE = false;

  /**
   * Default MongoDB date format.
   * Default: "EE MMM dd yyyy HH:mm:ss 'GMT'Z (zzz)"
   */
  private static final DateFormat DEFAULT_DATE_FORMAT =
      new SimpleDateFormat("EE MMM dd yyyy HH:mm:ss 'GMT'Z (zzz)", Locale.ENGLISH);

  public static final String DEFAULT_MONGO_MAP_COLLECTION = "mongo_data_model_map";

  private static final Pattern ID_PATTERN = Pattern.compile("[a-f0-9]{24}");

  /** MongoDB host */
  private String mongoHost = DEFAULT_MONGO_HOST;
  /** MongoDB port */
  private int mongoPort = DEFAULT_MONGO_PORT;
  /** MongoDB database */
  private String mongoDB = DEFAULT_MONGO_DB;
  /**
   * MongoDB authentication flag. If this flag is set to false,
   * authentication is not required.
   */
  private boolean mongoAuth = DEFAULT_MONGO_AUTH;
  /** MongoDB user */
  private String mongoUsername = DEFAULT_MONGO_USERNAME;
  /** MongoDB pass */
  private String mongoPassword = DEFAULT_MONGO_PASSWORD;
  /** MongoDB table/collection */
  private String mongoCollection = DEFAULT_MONGO_COLLECTION;
  /** MongoDB mapping table/collection */
  private String mongoMapCollection = DEFAULT_MONGO_MAP_COLLECTION;
  /**
   * MongoDB update flag. When this flag is activated, the
   * DataModel updates both model and database
   */
  private boolean mongoManage = DEFAULT_MONGO_MANAGE;
  /** MongoDB user ID field */
  private String mongoUserID = DEFAULT_MONGO_USER_ID;
  /** MongoDB item ID field */
  private String mongoItemID = DEFAULT_MONGO_ITEM_ID;
  /** MongoDB preference value field */
  private String mongoPreference = DEFAULT_MONGO_PREFERENCE;
  /** MongoDB final remove flag. Default: false */
  private boolean mongoFinalRemove = DEFAULT_MONGO_FINAL_REMOVE;
  /** MongoDB date format */
  private DateFormat dateFormat = DEFAULT_DATE_FORMAT;
  private DBCollection collection;
  private DBCollection collectionMap;
  private Date mongoTimestamp;
  private final ReentrantLock reloadLock;
  private DataModel delegate;
  private boolean userIsObject;
  private boolean itemIsObject;
  private boolean preferenceIsString;
  private long idCounter;

  /**
   * Creates a new MongoDBDataModel
   */
  public MongoDBDataModel() throws UnknownHostException {
    this.reloadLock = new ReentrantLock();
    buildModel();
  }

  /**
   * Creates a new MongoDBDataModel with MongoDB basic configuration
   * (without authentication)
   *
   * @param host        MongoDB host.
   * @param port        MongoDB port. Default: 27017
   * @param database    MongoDB database
   * @param collection  MongoDB collection/table
   * @param manage      If true, the model adds and removes users and items
   *                    from MongoDB database when the model is refreshed.
   * @param finalRemove If true, the model removes the user/item completely
   *                    from the MongoDB database. If false, the model adds the "deleted_at"
   *                    field with the current date to the "deleted" user/item.
   * @param format      MongoDB date format. If null, the model uses timestamps.
   * @throws UnknownHostException if the database host cannot be resolved
   */
  public MongoDBDataModel(String host,
                          int port,
                          String database,
                          String collection,
                          boolean manage,
                          boolean finalRemove,
                          DateFormat format) throws UnknownHostException {
    mongoHost = host;
    mongoPort = port;
    mongoDB = database;
    mongoCollection = collection;
    mongoManage = manage;
    mongoFinalRemove = finalRemove;
    dateFormat = format;
    this.reloadLock = new ReentrantLock();
    buildModel();
  }

  /**
   * Creates a new MongoDBDataModel with MongoDB advanced configuration
   * (without authentication)
   *
   * @param userIDField     Mongo user ID field
   * @param itemIDField     Mongo item ID field
   * @param preferenceField Mongo preference value field
   * @throws UnknownHostException if the database host cannot be resolved
   * @see #MongoDBDataModel(String, int, String, String, boolean, boolean, DateFormat)
   */
  public MongoDBDataModel(String host,
                          int port,
                          String database,
                          String collection,
                          boolean manage,
                          boolean finalRemove,
                          DateFormat format,
                          String userIDField,
                          String itemIDField,
                          String preferenceField,
                          String mappingCollection) throws UnknownHostException {
    mongoHost = host;
    mongoPort = port;
    mongoDB = database;
    mongoCollection = collection;
    mongoManage = manage;
    mongoFinalRemove = finalRemove;
    dateFormat = format;
    mongoUserID = userIDField;
    mongoItemID = itemIDField;
    mongoPreference = preferenceField;
    mongoMapCollection = mappingCollection;
    this.reloadLock = new ReentrantLock();
    buildModel();
  }

  /**
   * Creates a new MongoDBDataModel with MongoDB basic configuration
   * (with authentication)
   *
   * @param user     Mongo username (authentication)
   * @param password Mongo password (authentication)
   * @throws UnknownHostException if the database host cannot be resolved
   * @see #MongoDBDataModel(String, int, String, String, boolean, boolean, DateFormat)
   */
  public MongoDBDataModel(String host,
                          int port,
                          String database,
                          String collection,
                          boolean manage,
                          boolean finalRemove,
                          DateFormat format,
                          String user,
                          String password) throws UnknownHostException {
    mongoHost = host;
    mongoPort = port;
    mongoDB = database;
    mongoCollection = collection;
    mongoManage = manage;
    mongoFinalRemove = finalRemove;
    dateFormat = format;
    mongoAuth = true;
    mongoUsername = user;
    mongoPassword = password;
    this.reloadLock = new ReentrantLock();
    buildModel();
  }

  /**
   * Creates a new MongoDBDataModel with MongoDB advanced configuration
   * (with authentication)
   *
   * @throws UnknownHostException if the database host cannot be resolved
   * @see #MongoDBDataModel(String, int, String, String, boolean, boolean, DateFormat, String, String)
   */
  public MongoDBDataModel(String host,
                          int port,
                          String database,
                          String collection,
                          boolean manage,
                          boolean finalRemove,
                          DateFormat format,
                          String user,
                          String password,
                          String userIDField,
                          String itemIDField,
                          String preferenceField,
                          String mappingCollection) throws UnknownHostException {
    mongoHost = host;
    mongoPort = port;
    mongoDB = database;
    mongoCollection = collection;
    mongoManage = manage;
    mongoFinalRemove = finalRemove;
    dateFormat = format;
    mongoAuth = true;
    mongoUsername = user;
    mongoPassword = password;
    mongoUserID = userIDField;
    mongoItemID = itemIDField;
    mongoPreference = preferenceField;
    mongoMapCollection = mappingCollection;
    this.reloadLock = new ReentrantLock();
    buildModel();
  }

  /**
   * <p>
   * Adds/removes (user, item) pairs to/from the model.
   * </p>
   *
   * @param userID MongoDB user identifier
   * @param items  List of pairs (item, preference) which want to be added or
   *               deleted
   * @param add    If true, this flag indicates that the    pairs (user, item)
   *               must be added to the model. If false, it indicates deletion.
   * @see #refresh(Collection)
   */
  public void refreshData(String userID,
                          Iterable<List<String>> items,
                          boolean add) throws NoSuchUserException, NoSuchItemException {
    checkData(userID, items, add);
    long id = Long.parseLong(fromIdToLong(userID, true));
    for (List<String> item : items) {
      item.set(0, fromIdToLong(item.get(0), false));
    }
    if (reloadLock.tryLock()) {
      try {
        if (add) {
          delegate = addUserItem(id, items);
        } else {
          delegate = removeUserItem(id, items);
        }
      } finally {
        reloadLock.unlock();
      }
    }
  }


  /**
   * <p>
   * Triggers "refresh" -- whatever that means -- of the implementation.
   * The general contract is that any should always leave itself in a
   * consistent, operational state, and that the refresh atomically updates
   * internal state from old to new.
   * </p>
   *
   * @param alreadyRefreshed s that are known to have already been refreshed as
   *                         a result of an initial call to a method on some object. This ensures
   *                         that objects in a refresh dependency graph aren't refreshed twice
   *                         needlessly.
   * @see #refreshData(String, Iterable, boolean)
   */
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    BasicDBObject query = new BasicDBObject();
    query.put("deleted_at", new BasicDBObject("$gt", mongoTimestamp));
    DBCursor cursor = collection.find(query);
    Date ts = new Date(0);
    while (cursor.hasNext()) {
      Map<String,Object> user = (Map<String,Object>) cursor.next().toMap();
      String userID = getID(user.get(mongoUserID), true);
      Collection<List<String>> items = Lists.newArrayList();
      List<String> item = Lists.newArrayList();
      item.add(getID(user.get(mongoItemID), false));
      item.add(Float.toString(getPreference(user.get(mongoPreference))));
      items.add(item);
      try {
        refreshData(userID, items, false);
      } catch (NoSuchUserException e) {
        log.warn("No such user ID: {}", userID);
      } catch (NoSuchItemException e) {
        log.warn("No such items: {}", items);
      }
      if (ts.compareTo(getDate(user.get("created_at"))) < 0) {
        ts = getDate(user.get("created_at"));
      }
    }
    query = new BasicDBObject();
    query.put("created_at", new BasicDBObject("$gt", mongoTimestamp));
    cursor = collection.find(query);
    while (cursor.hasNext()) {
      Map<String,Object> user = (Map<String,Object>) cursor.next().toMap();
      if (!user.containsKey("deleted_at")) {
        String userID = getID(user.get(mongoUserID), true);
        Collection<List<String>> items = Lists.newArrayList();
        List<String> item = Lists.newArrayList();
        item.add(getID(user.get(mongoItemID), false));
        item.add(Float.toString(getPreference(user.get(mongoPreference))));
        items.add(item);
        try {
          refreshData(userID, items, true);
        } catch (NoSuchUserException e) {
          log.warn("No such user ID: {}", userID);
        } catch (NoSuchItemException e) {
          log.warn("No such items: {}", items);
        }
        if (ts.compareTo(getDate(user.get("created_at"))) < 0) {
          ts = getDate(user.get("created_at"));
        }
      }
    }
    if (mongoTimestamp.compareTo(ts) < 0) {
      mongoTimestamp = ts;
    }
  }

  /**
   * <p>
   * Translates the MongoDB identifier to Mahout/MongoDBDataModel's internal
   * identifier, if required.
   * </p>
   * <p>
   * If MongoDB identifiers are long datatypes, it returns the id.
   * </p>
   * <p>
   * This conversion is needed since Mahout uses the long datatype to feed the
   * recommender, and MongoDB uses 12 bytes to create its identifiers.
   * </p>
   *
   * @param id     MongoDB identifier
   * @param isUser
   * @return String containing the translation of the external MongoDB ID to
   *         internal long ID (mapping).
   * @see #fromLongToId(long)
   * @see <a href="http://www.mongodb.org/display/DOCS/Object%20IDs">
   *      Mongo Object IDs</a>
   */
  public String fromIdToLong(String id, boolean isUser) {
    DBObject objectIdLong = collectionMap.findOne(new BasicDBObject("element_id", id));
    if (objectIdLong != null) {
      Map<String,Object> idLong = (Map<String,Object>) objectIdLong.toMap();
      Object value = idLong.get("long_value");
      return value == null ? null : value.toString();
    } else {
      objectIdLong = new BasicDBObject();
      String longValue = Long.toString(idCounter++);
      objectIdLong.put("element_id", id);
      objectIdLong.put("long_value", longValue);
      collectionMap.insert(objectIdLong);
      log.info("Adding Translation {}: {} long_value: {}",
               isUser ? "User ID" : "Item ID", id, longValue);
      return longValue;
    }
  }

  /**
   * <p>
   * Translates the Mahout/MongoDBDataModel's internal identifier to MongoDB
   * identifier, if required.
   * </p>
   * <p>
   * If MongoDB identifiers are long datatypes, it returns the id in String
   * format.
   * </p>
   * <p>
   * This conversion is needed since Mahout uses the long datatype to feed the
   * recommender, and MongoDB uses 12 bytes to create its identifiers.
   * </p>
   *
   * @param id Mahout's internal identifier
   * @return String containing the translation of the internal long ID to
   *         external MongoDB ID (mapping).
   * @see #fromIdToLong(String, boolean)
   * @see <a href="http://www.mongodb.org/display/DOCS/Object%20IDs">
   *      Mongo Object IDs</a>
   */
  public String fromLongToId(long id) {
    DBObject objectIdLong = collectionMap.findOne(new BasicDBObject("long_value", Long.toString(id)));
    Map<String,Object> idLong = (Map<String,Object>) objectIdLong.toMap();
    Object value = idLong.get("element_id");
    return value == null ? null : value.toString();
  }

  /**
   * <p>
   * Checks if an ID is currently in the model.
   * </p>
   *
   * @param ID user or item ID
   * @return true: if ID is into the model; false: if it's not.
   */
  public boolean isIDInModel(String ID) {
    DBObject objectIdLong = collectionMap.findOne(new BasicDBObject("element_id", ID));
    return objectIdLong != null;
  }

  /**
   * <p>
   * Date of the latest update of the model.
   * </p>
   *
   * @return Date with the latest update of the model.
   */
  public Date mongoUpdateDate() {
    return mongoTimestamp;
  }

  private void buildModel() throws UnknownHostException {
    userIsObject = false;
    itemIsObject = false;
    idCounter = 0;
    preferenceIsString = true;
    Mongo mongoDDBB = new Mongo(mongoHost, mongoPort);
    DB db = mongoDDBB.getDB(mongoDB);
    mongoTimestamp = new Date(0);
    FastByIDMap<Collection<Preference>> userIDPrefMap = new FastByIDMap<Collection<Preference>>();
    if (!mongoAuth || db.authenticate(mongoUsername, mongoPassword.toCharArray())) {
      collection = db.getCollection(mongoCollection);
      collectionMap = db.getCollection(mongoMapCollection);
      DBObject indexObj = new BasicDBObject();
      indexObj.put("element_id", 1);
      collectionMap.ensureIndex(indexObj);
      indexObj = new BasicDBObject();
      indexObj.put("long_value", 1);
      collectionMap.ensureIndex(indexObj);
      collectionMap.remove(new BasicDBObject());
      DBCursor cursor = collection.find();
      while (cursor.hasNext()) {
        Map<String,Object> user = (Map<String,Object>) cursor.next().toMap();
        if (!user.containsKey("deleted_at")) {
          long userID = Long.parseLong(fromIdToLong(getID(user.get(mongoUserID), true), true));
          long itemID = Long.parseLong(fromIdToLong(getID(user.get(mongoItemID), false), false));
          float ratingValue = getPreference(user.get(mongoPreference));
          Collection<Preference> userPrefs = userIDPrefMap.get(userID);
          if (userPrefs == null) {
            userPrefs = Lists.newArrayListWithCapacity(2);
            userIDPrefMap.put(userID, userPrefs);
          }
          userPrefs.add(new GenericPreference(userID, itemID, ratingValue));
          if (user.containsKey("created_at")
              && mongoTimestamp.compareTo(getDate(user.get("created_at"))) < 0) {
            mongoTimestamp = getDate(user.get("created_at"));
          }
        }
      }
    }
    delegate = new GenericDataModel(GenericDataModel.toDataMap(userIDPrefMap, true));
  }

  private void removeMongoUserItem(String userID, String itemID) {
    String userId = fromLongToId(Long.parseLong(userID));
    String itemId = fromLongToId(Long.parseLong(itemID));
    if (isUserItemInDB(userId, itemId)) {
      mongoTimestamp = new Date();
      BasicDBObject query = new BasicDBObject();
      query.put(mongoUserID, userIsObject ? new ObjectId(userId) : userId);
      query.put(mongoItemID, itemIsObject ? new ObjectId(itemId) : itemId);
      if (mongoFinalRemove) {
        log.info(collection.remove(query).toString());
      } else {
        BasicDBObject update = new BasicDBObject();
        update.put("$set", new BasicDBObject("deleted_at", mongoTimestamp));
        log.info(collection.update(query, update).toString());
      }
      log.info("Removing userID: {} itemID: {}", userID, itemId);
    }
  }

  private void addMongoUserItem(String userID, String itemID, String preferenceValue) {
    String userId = fromLongToId(Long.parseLong(userID));
    String itemId = fromLongToId(Long.parseLong(itemID));
    if (!isUserItemInDB(userId, itemId)) {
      mongoTimestamp = new Date();
      BasicDBObject user = new BasicDBObject();
      Object userIdObject = userIsObject ? new ObjectId(userId) : userId;
      Object itemIdObject = itemIsObject ? new ObjectId(itemId) : itemId;
      user.put(mongoUserID, userIdObject);
      user.put(mongoItemID, itemIdObject);
      user.put(mongoPreference, preferenceIsString ? preferenceValue : Double.parseDouble(preferenceValue));
      user.put("created_at", mongoTimestamp);
      collection.insert(user);
      log.info("Adding userID: {} itemID: {} preferenceValue: {}", userID, itemID, preferenceValue);
    }
  }

  private boolean isUserItemInDB(String userID, String itemID) {
    BasicDBObject query = new BasicDBObject();
    Object userId = userIsObject ? new ObjectId(userID) : userID;
    Object itemId = itemIsObject ? new ObjectId(itemID) : itemID;
    query.put(mongoUserID, userId);
    query.put(mongoItemID, itemId);
    return collection.findOne(query) != null;
  }

  private DataModel removeUserItem(long userID, Iterable<List<String>> items) {
    FastByIDMap<PreferenceArray> rawData = ((GenericDataModel) delegate).getRawUserData();
    for (List<String> item : items) {
      PreferenceArray prefs = rawData.get(userID);
      long itemID = Long.parseLong(item.get(0));
      if (prefs != null) {
        boolean exists = false;
        int length = prefs.length();
        for (int i = 0; i < length; i++) {
          if (prefs.getItemID(i) == itemID) {
            exists = true;
            break;
          }
        }
        if (exists) {
          rawData.remove(userID);
          if (length > 1) {
            PreferenceArray newPrefs = new GenericUserPreferenceArray(length - 1);
            for (int i = 0, j = 0; i < length; i++, j++) {
              if (prefs.getItemID(i) == itemID) {
                j--;
              } else {
                newPrefs.set(j, prefs.get(i));
              }
            }
            rawData.put(userID, newPrefs);
          }
          log.info("Removing userID: {} itemID: {}", userID, itemID);
          if (mongoManage) {
            removeMongoUserItem(Long.toString(userID), Long.toString(itemID));
          }
        }
      }
    }
    return new GenericDataModel(rawData);
  }

  private DataModel addUserItem(long userID, Iterable<List<String>> items) {
    FastByIDMap<PreferenceArray> rawData = ((GenericDataModel) delegate).getRawUserData();
    PreferenceArray prefs = rawData.get(userID);
    for (List<String> item : items) {
      long itemID = Long.parseLong(item.get(0));
      float preferenceValue = Float.parseFloat(item.get(1));
      boolean exists = false;
      if (prefs != null) {
        for (int i = 0; i < prefs.length(); i++) {
          if (prefs.getItemID(i) == itemID) {
            exists = true;
            prefs.setValue(i, preferenceValue);
            break;
          }
        }
      }
      if (!exists) {
        if (prefs == null) {
          prefs = new GenericUserPreferenceArray(1);
        } else {
          PreferenceArray newPrefs = new GenericUserPreferenceArray(prefs.length() + 1);
          for (int i = 0, j = 1; i < prefs.length(); i++, j++) {
            newPrefs.set(j, prefs.get(i));
          }
          prefs = newPrefs;
        }
        prefs.setUserID(0, userID);
        prefs.setItemID(0, itemID);
        prefs.setValue(0, preferenceValue);
        log.info("Adding userID: {} itemID: {} preferenceValue: {}", userID, itemID, preferenceValue);
        rawData.put(userID, prefs);
        if (mongoManage) {
          addMongoUserItem(Long.toString(userID),
                           Long.toString(itemID),
                           Float.toString(preferenceValue));
        }
      }
    }
    return new GenericDataModel(rawData);
  }

  private Date getDate(Object date) {
    if (date.getClass().getName().contains("Date")) {
      return (Date) date;
    }
    if (date.getClass().getName().contains("String")) {
      try {
        synchronized (dateFormat) {
          return dateFormat.parse(date.toString());
        }
      } catch (ParseException ioe) {
        log.warn("Error parsing timestamp", ioe);
      }
    }
    return new Date(0);
  }

  private float getPreference(Object value) {
    if (value != null) {
      if (value.getClass().getName().contains("String")) {
        preferenceIsString = true;
        return Float.parseFloat(value.toString());
      } else {
        preferenceIsString = false;
        return Double.valueOf(value.toString()).floatValue();
      }
    } else {
      return 0.5f;
    }
  }

  private String getID(Object id, boolean isUser) {
    if (id.getClass().getName().contains("ObjectId")) {
      if (isUser) {
        userIsObject = true;
      } else {
        itemIsObject = true;
      }
      return ((ObjectId) id).toStringMongod();
    } else {
      return id.toString();
    }
  }

  private void checkData(String userID,
                         Iterable<List<String>> items,
                         boolean add) throws NoSuchUserException, NoSuchItemException {
    Preconditions.checkNotNull(userID);
    Preconditions.checkNotNull(items);
    Preconditions.checkArgument(!userID.isEmpty(), "userID is empty");
    for (List<String> item : items) {
      Preconditions.checkNotNull(item.get(0));
      Preconditions.checkArgument(!item.get(0).isEmpty(), "item is empty");
    }
    if (userIsObject && !ID_PATTERN.matcher(userID).matches()) {
      throw new IllegalArgumentException();
    }
    for (List<String> item : items) {
      if (itemIsObject && !ID_PATTERN.matcher(item.get(0)).matches()) {
        throw new IllegalArgumentException();
      }
    }
    if (!add && !isIDInModel(userID)) {
      throw new NoSuchUserException();
    }
    for (List<String> item : items) {
      if (!add && !isIDInModel(item.get(0))) {
        throw new NoSuchItemException();
      }
    }
  }

  /**
   * Cleanup mapping collection.
   */
  public void cleanupMappingCollection() {
    collectionMap.drop();
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    return delegate.getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long id) throws TasteException {
    return delegate.getPreferencesFromUser(id);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return delegate.getItemIDsFromUser(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    return delegate.getItemIDs();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return delegate.getPreferencesForItem(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceValue(userID, itemID);
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceTime(userID, itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
    return delegate.getNumItems();
  }

  @Override
  public int getNumUsers() throws TasteException {
    return delegate.getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemID);
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemID1, itemID2);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void removePreference(long userID, long itemID) {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean hasPreferenceValues() {
    return delegate.hasPreferenceValues();
  }

  @Override
  public float getMaxPreference() {
    return delegate.getMaxPreference();
  }

  @Override
  public float getMinPreference() {
    return delegate.getMinPreference();
  }

  @Override
  public String toString() {
    return "MongoDBDataModel";
  }

}

/*
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

package org.apache.mahout.cf.taste.impl.model;

import java.util.Iterator;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class PlusAnonymousConcurrentUserDataModelTest extends MahoutTestCase {

	/**
	 * Prepares a testable object without delegate data
	 */
	private static PlusAnonymousConcurrentUserDataModel getTestableWithoutDelegateData(int maxConcurrentUsers) {
		FastByIDMap<PreferenceArray> delegatePreferences = new FastByIDMap<PreferenceArray>();
		return new PlusAnonymousConcurrentUserDataModel(new GenericDataModel(delegatePreferences), maxConcurrentUsers);
	}

	/**
	 * Prepares a testable object with delegate data
	 */
  private static PlusAnonymousConcurrentUserDataModel getTestableWithDelegateData(
        int maxConcurrentUsers, FastByIDMap<PreferenceArray> delegatePreferences) {
		return new PlusAnonymousConcurrentUserDataModel(new GenericDataModel(delegatePreferences), maxConcurrentUsers);
	}

	/**
	 * Test taking the first available user
	 */
	@Test
	public void testTakeFirstAvailableUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		Long expResult = PlusAnonymousUserDataModel.TEMP_USER_ID;
		Long result = instance.takeAvailableUser();
		assertEquals(expResult, result);
	}

	/**
	 * Test taking the next available user
	 */
	@Test
	public void testTakeNextAvailableUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
    // Skip first user
    instance.takeAvailableUser();
		Long result = instance.takeAvailableUser();
    Long expResult = PlusAnonymousUserDataModel.TEMP_USER_ID + 1;
    assertEquals(expResult, result);
	}

	/**
	 * Test taking an unavailable user
	 */
	@Test
	public void testTakeUnavailableUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(1);
		// Take the only available user
		instance.takeAvailableUser();
		// There are no more users available
		assertNull(instance.takeAvailableUser());
	}

	/**
	 * Test releasing a valid previously taken user
	 */
	@Test
	public void testReleaseValidUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		Long takenUserID = instance.takeAvailableUser();
		assertTrue(instance.releaseUser(takenUserID));
	}

	/**
	 * Test releasing an invalid user
	 */
	@Test
	public void testReleaseInvalidUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		assertFalse(instance.releaseUser(Long.MAX_VALUE));
	}

	/**
	 * Test releasing a user which had been released earlier
	 */
	@Test
	public void testReleasePreviouslyReleasedUser() {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		Long takenUserID = instance.takeAvailableUser();
		assertTrue(instance.releaseUser(takenUserID));
		assertFalse(instance.releaseUser(takenUserID));
	}

	/**
	 * Test setting anonymous user preferences
	 */
	@Test
	public void testSetAndGetTempPreferences() throws TasteException {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		Long anonymousUserID = instance.takeAvailableUser();
		PreferenceArray tempPrefs = new GenericUserPreferenceArray(1);
		tempPrefs.setUserID(0, anonymousUserID);
		tempPrefs.setItemID(0, 1);
		instance.setTempPrefs(tempPrefs, anonymousUserID);
		assertEquals(tempPrefs, instance.getPreferencesFromUser(anonymousUserID));
		instance.releaseUser(anonymousUserID);
	}

	/**
	 * Test setting and getting preferences from several concurrent anonymous users
	 */
	@Test
	public void testSetMultipleTempPreferences() throws TasteException {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);

		Long anonymousUserID1 = instance.takeAvailableUser();
		Long anonymousUserID2 = instance.takeAvailableUser();

		PreferenceArray tempPrefs1 = new GenericUserPreferenceArray(1);
		tempPrefs1.setUserID(0, anonymousUserID1);
		tempPrefs1.setItemID(0, 1);

		PreferenceArray tempPrefs2 = new GenericUserPreferenceArray(2);
		tempPrefs2.setUserID(0, anonymousUserID2);
		tempPrefs2.setItemID(0, 2);
		tempPrefs2.setUserID(1, anonymousUserID2);
		tempPrefs2.setItemID(1, 3);

		instance.setTempPrefs(tempPrefs1, anonymousUserID1);
		instance.setTempPrefs(tempPrefs2, anonymousUserID2);

		assertEquals(tempPrefs1, instance.getPreferencesFromUser(anonymousUserID1));
		assertEquals(tempPrefs2, instance.getPreferencesFromUser(anonymousUserID2));
	}

	/**
	 * Test counting the number of delegate users
	 */
	@Test
	public void testGetNumUsersWithDelegateUsersOnly() throws TasteException {
    PreferenceArray prefs = new GenericUserPreferenceArray(1);
    long sampleUserID = 1;
		prefs.setUserID(0, sampleUserID);
    long sampleItemID = 11;
    prefs.setItemID(0, sampleItemID);

		FastByIDMap<PreferenceArray> delegatePreferences = new FastByIDMap<PreferenceArray>();
		delegatePreferences.put(sampleUserID, prefs);

		PlusAnonymousConcurrentUserDataModel instance = getTestableWithDelegateData(10, delegatePreferences);

		assertEquals(1, instance.getNumUsers());
	}

	/**
	 * Test counting the number of anonymous users
	 */
	@Test
	public void testGetNumAnonymousUsers() throws TasteException {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);

		Long anonymousUserID1 = instance.takeAvailableUser();

		PreferenceArray tempPrefs1 = new GenericUserPreferenceArray(1);
		tempPrefs1.setUserID(0, anonymousUserID1);
		tempPrefs1.setItemID(0, 1);

		instance.setTempPrefs(tempPrefs1, anonymousUserID1);

		// Anonymous users should not be included into the universe.
		assertEquals(0, instance.getNumUsers());
	}

	/**
	 * Test retrieve a single preference value of an anonymous user
	 */
	@Test
	public void testGetPreferenceValue() throws TasteException {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);

		Long anonymousUserID = instance.takeAvailableUser();

		PreferenceArray tempPrefs = new GenericUserPreferenceArray(1);
    tempPrefs.setUserID(0, anonymousUserID);
    long sampleItemID = 1;
    tempPrefs.setItemID(0, sampleItemID);
    tempPrefs.setValue(0, Float.MAX_VALUE);

		instance.setTempPrefs(tempPrefs, anonymousUserID);

		assertEquals(Float.MAX_VALUE, instance.getPreferenceValue(anonymousUserID, sampleItemID), EPSILON);
	}

	/**
	 * Test retrieve preferences for existing non-anonymous user
	 */
	@Test
	public void testGetPreferencesForNonAnonymousUser() throws TasteException {
    PreferenceArray prefs = new GenericUserPreferenceArray(1);
    long sampleUserID = 1;
		prefs.setUserID(0, sampleUserID);
    long sampleItemID = 11;
    prefs.setItemID(0, sampleItemID);

		FastByIDMap<PreferenceArray> delegatePreferences = new FastByIDMap<PreferenceArray>();
		delegatePreferences.put(sampleUserID, prefs);

		PlusAnonymousConcurrentUserDataModel instance = getTestableWithDelegateData(10, delegatePreferences);

		assertEquals(prefs, instance.getPreferencesFromUser(sampleUserID));
	}

	/**
	 * Test retrieve preferences for non-anonymous and non-existing user
	 */
	@Test(expected=NoSuchUserException.class)
	public void testGetPreferencesForNonExistingUser() throws TasteException {
		PlusAnonymousConcurrentUserDataModel instance = getTestableWithoutDelegateData(10);
		// Exception is expected since such user does not exist
		instance.getPreferencesFromUser(1);
	}

	/**
	 * Test retrieving the user IDs and verifying that anonymous ones are not included
	 */
	@Test
	public void testGetUserIDs() throws TasteException {
    PreferenceArray prefs = new GenericUserPreferenceArray(1);
    long sampleUserID = 1;
		prefs.setUserID(0, sampleUserID);
    long sampleItemID = 11;
    prefs.setItemID(0, sampleItemID);

		FastByIDMap<PreferenceArray> delegatePreferences = new FastByIDMap<PreferenceArray>();
		delegatePreferences.put(sampleUserID, prefs);

		PlusAnonymousConcurrentUserDataModel instance = getTestableWithDelegateData(10, delegatePreferences);

		Long anonymousUserID = instance.takeAvailableUser();

		PreferenceArray tempPrefs = new GenericUserPreferenceArray(1);
		tempPrefs.setUserID(0, anonymousUserID);
		tempPrefs.setItemID(0, 22);

		instance.setTempPrefs(tempPrefs, anonymousUserID);

		Iterator<Long> userIDs = instance.getUserIDs();

		assertSame(sampleUserID, userIDs.next());
		assertFalse(userIDs.hasNext());
	}

	/**
	 * Test getting preferences for an item.
	 *
	 * @throws TasteException
	 */
	@Test
	public void testGetPreferencesForItem() throws TasteException {
    PreferenceArray prefs = new GenericUserPreferenceArray(2);
    long sampleUserID = 4;
		prefs.setUserID(0, sampleUserID);
    long sampleItemID = 11;
    prefs.setItemID(0, sampleItemID);
		prefs.setUserID(1, sampleUserID);
    long sampleItemID2 = 22;
    prefs.setItemID(1, sampleItemID2);

		FastByIDMap<PreferenceArray> delegatePreferences = new FastByIDMap<PreferenceArray>();
		delegatePreferences.put(sampleUserID, prefs);

		PlusAnonymousConcurrentUserDataModel instance = getTestableWithDelegateData(10, delegatePreferences);

		Long anonymousUserID = instance.takeAvailableUser();

		PreferenceArray tempPrefs = new GenericUserPreferenceArray(2);
		tempPrefs.setUserID(0, anonymousUserID);
		tempPrefs.setItemID(0, sampleItemID);
		tempPrefs.setUserID(1, anonymousUserID);
    long sampleItemID3 = 33;
    tempPrefs.setItemID(1, sampleItemID3);

		instance.setTempPrefs(tempPrefs, anonymousUserID);

		assertEquals(sampleUserID, instance.getPreferencesForItem(sampleItemID).get(0).getUserID());
		assertEquals(2, instance.getPreferencesForItem(sampleItemID).length());
		assertEquals(1, instance.getPreferencesForItem(sampleItemID2).length());
		assertEquals(1, instance.getPreferencesForItem(sampleItemID3).length());

		assertEquals(2, instance.getNumUsersWithPreferenceFor(sampleItemID));
		assertEquals(1, instance.getNumUsersWithPreferenceFor(sampleItemID, sampleItemID2));
		assertEquals(1, instance.getNumUsersWithPreferenceFor(sampleItemID, sampleItemID3));
	}

}

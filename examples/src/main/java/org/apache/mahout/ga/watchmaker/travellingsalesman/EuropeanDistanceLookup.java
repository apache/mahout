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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class contains data about cities in Europe and the distances
 * between them.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
public final class EuropeanDistanceLookup implements DistanceLookup
{
    private static final Map<String, Map<String, Integer>> DISTANCES = new HashMap<String, Map<String, Integer>>(15);
    static
    {
        // Distances are in km as the crow flies (from http://www.indo.com/distance/)

        Map<String, Integer> amsterdam = new HashMap<String, Integer>(20);
        amsterdam.put("Amsterdam", 0);
        amsterdam.put("Athens", 2162);
        amsterdam.put("Berlin", 576);
        amsterdam.put("Brussels", 171);
        amsterdam.put("Copenhagen", 622);
        amsterdam.put("Dublin", 757);
        amsterdam.put("Helsinki", 1506);
        amsterdam.put("Lisbon", 1861);
        amsterdam.put("London", 356);
        amsterdam.put("Luxembourg", 318);
        amsterdam.put("Madrid", 1477);
        amsterdam.put("Paris", 429);
        amsterdam.put("Rome", 1304);
        amsterdam.put("Stockholm", 1132);
        amsterdam.put("Vienna", 938);
        DISTANCES.put("Amsterdam", amsterdam);

        Map<String, Integer> athens = new HashMap<String, Integer>(20);
        athens.put("Amsterdam", 2162);
        athens.put("Athens", 0);
        athens.put("Berlin", 1801);
        athens.put("Brussels", 2089);
        athens.put("Copenhagen", 2140);
        athens.put("Dublin", 2860);
        athens.put("Helsinki", 2464);
        athens.put("Lisbon", 2854);
        athens.put("London", 2391);
        athens.put("Luxembourg", 1901);
        athens.put("Madrid", 2374);
        athens.put("Paris", 2097);
        athens.put("Rome", 1040);
        athens.put("Stockholm", 2410);
        athens.put("Vienna", 1280);
        DISTANCES.put("Athens", athens);

        Map<String, Integer> berlin = new HashMap<String, Integer>(20);
        berlin.put("Amsterdam", 576);
        berlin.put("Athens", 1801);
        berlin.put("Berlin", 0);
        berlin.put("Brussels", 648);
        berlin.put("Copenhagen", 361);
        berlin.put("Dublin", 1315);
        berlin.put("Helsinki", 1108);
        berlin.put("Lisbon", 2310);
        berlin.put("London", 929);
        berlin.put("Luxembourg", 595);
        berlin.put("Madrid", 1866);
        berlin.put("Paris", 877);
        berlin.put("Rome", 1185);
        berlin.put("Stockholm", 818);
        berlin.put("Vienna", 525);
        DISTANCES.put("Berlin", berlin);

        Map<String, Integer> brussels = new HashMap<String, Integer>(20);
        brussels.put("Amsterdam", 171);
        brussels.put("Athens", 2089);
        brussels.put("Berlin", 648);
        brussels.put("Brussels", 0);
        brussels.put("Copenhagen", 764);
        brussels.put("Dublin", 780);
        brussels.put("Helsinki", 1649);
        brussels.put("Lisbon", 1713);
        brussels.put("London", 321);
        brussels.put("Luxembourg", 190);
        brussels.put("Madrid", 1315);
        brussels.put("Paris", 266);
        brussels.put("Rome", 1182);
        brussels.put("Stockholm", 1284);
        brussels.put("Vienna", 917);
        DISTANCES.put("Brussels", brussels);

        Map<String, Integer> copenhagen = new HashMap<String, Integer>(20);
        copenhagen.put("Amsterdam", 622);
        copenhagen.put("Athens", 2140);
        copenhagen.put("Berlin", 361);
        copenhagen.put("Brussels", 764);
        copenhagen.put("Copenhagen", 0);
        copenhagen.put("Dublin", 1232);
        copenhagen.put("Helsinki", 885);
        copenhagen.put("Lisbon", 2477);
        copenhagen.put("London", 953);
        copenhagen.put("Luxembourg", 799);
        copenhagen.put("Madrid", 2071);
        copenhagen.put("Paris", 1028);
        copenhagen.put("Rome", 1540);
        copenhagen.put("Stockholm", 526);
        copenhagen.put("Vienna", 876);
        DISTANCES.put("Copenhagen", copenhagen);

        Map<String, Integer> dublin = new HashMap<String, Integer>(20);
        dublin.put("Amsterdam", 757);
        dublin.put("Athens", 2860);
        dublin.put("Berlin", 1315);
        dublin.put("Brussels", 780);
        dublin.put("Copenhagen", 1232);
        dublin.put("Dublin", 0);
        dublin.put("Helsinki", 2021);
        dublin.put("Lisbon", 1652);
        dublin.put("London", 469);
        dublin.put("Luxembourg", 961);
        dublin.put("Madrid", 1458);
        dublin.put("Paris", 787);
        dublin.put("Rome", 1903);
        dublin.put("Stockholm", 1625);
        dublin.put("Vienna", 1687);
        DISTANCES.put("Dublin", dublin);

        Map<String, Integer> helsinki = new HashMap<String, Integer>(20);
        helsinki.put("Amsterdam", 1506);
        helsinki.put("Athens", 2464);
        helsinki.put("Berlin", 1108);
        helsinki.put("Brussels", 1649);
        helsinki.put("Copenhagen", 885);
        helsinki.put("Dublin", 2021);
        helsinki.put("Helsinki", 0);
        helsinki.put("Lisbon", 3362);
        helsinki.put("London", 1823);
        helsinki.put("Luxembourg", 1667);
        helsinki.put("Madrid", 2949);
        helsinki.put("Paris", 1912);
        helsinki.put("Rome", 2202);
        helsinki.put("Stockholm", 396);
        helsinki.put("Vienna", 1439);
        DISTANCES.put("Helsinki", helsinki);

        Map<String, Integer> lisbon = new HashMap<String, Integer>(20);
        lisbon.put("Amsterdam", 1861);
        lisbon.put("Athens", 2854);
        lisbon.put("Berlin", 2310);
        lisbon.put("Brussels", 1713);
        lisbon.put("Copenhagen", 2477);
        lisbon.put("Dublin", 1652);
        lisbon.put("Helsinki", 3362);
        lisbon.put("Lisbon", 0);
        lisbon.put("London", 1585);
        lisbon.put("Luxembourg", 1716);
        lisbon.put("Madrid", 501);
        lisbon.put("Paris", 1452);
        lisbon.put("Rome", 1873);
        lisbon.put("Stockholm", 2993);
        lisbon.put("Vienna", 2300);
        DISTANCES.put("Lisbon", lisbon);

        Map<String, Integer> london = new HashMap<String, Integer>(20);
        london.put("Amsterdam", 356);
        london.put("Athens", 2391);
        london.put("Berlin", 929);
        london.put("Brussels", 321);
        london.put("Copenhagen", 953);
        london.put("Dublin", 469);
        london.put("Helsinki", 1823);
        london.put("Lisbon", 1585);
        london.put("London", 0);
        london.put("Luxembourg", 494);
        london.put("Madrid", 1261);
        london.put("Paris", 343);
        london.put("Rome", 1444);
        london.put("Stockholm", 1436);
        london.put("Vienna", 1237);
        DISTANCES.put("London", london);

        Map<String, Integer> luxembourg = new HashMap<String, Integer>(20);
        luxembourg.put("Amsterdam", 318);
        luxembourg.put("Athens", 1901);
        luxembourg.put("Berlin", 595);
        luxembourg.put("Brussels", 190);
        luxembourg.put("Copenhagen", 799);
        luxembourg.put("Dublin", 961);
        luxembourg.put("Helsinki", 1667);
        luxembourg.put("Lisbon", 1716);
        luxembourg.put("London", 494);
        luxembourg.put("Luxembourg", 0);
        luxembourg.put("Madrid", 1282);
        luxembourg.put("Paris", 294);
        luxembourg.put("Rome", 995);
        luxembourg.put("Stockholm", 1325);
        luxembourg.put("Vienna", 761);
        DISTANCES.put("Luxembourg", luxembourg);

        Map<String, Integer> madrid = new HashMap<String, Integer>(20);
        madrid.put("Amsterdam", 1477);
        madrid.put("Athens", 2374);
        madrid.put("Berlin", 1866);
        madrid.put("Brussels", 1315);
        madrid.put("Copenhagen", 2071);
        madrid.put("Dublin", 1458);
        madrid.put("Helsinki", 2949);
        madrid.put("Lisbon", 501);
        madrid.put("London", 1261);
        madrid.put("Luxembourg", 1282);
        madrid.put("Madrid", 0);
        madrid.put("Paris", 1050);
        madrid.put("Rome", 1377);
        madrid.put("Stockholm", 2596);
        madrid.put("Vienna", 1812);
        DISTANCES.put("Madrid", madrid);

        Map<String, Integer> paris = new HashMap<String, Integer>(20);
        paris.put("Amsterdam", 429);
        paris.put("Athens", 2097);
        paris.put("Berlin", 877);
        paris.put("Brussels", 266);
        paris.put("Copenhagen", 1028);
        paris.put("Dublin", 787);
        paris.put("Helsinki", 1912);
        paris.put("Lisbon", 1452);
        paris.put("London", 343);
        paris.put("Luxembourg", 294);
        paris.put("Madrid", 1050);
        paris.put("Paris", 0);
        paris.put("Rome", 1117);
        paris.put("Stockholm", 1549);
        paris.put("Vienna", 1037);
        DISTANCES.put("Paris", paris);

        Map<String, Integer> rome = new HashMap<String, Integer>(20);
        rome.put("Amsterdam", 1304);
        rome.put("Athens", 1040);
        rome.put("Berlin", 1185);
        rome.put("Brussels", 1182);
        rome.put("Copenhagen", 1540);
        rome.put("Dublin", 1903);
        rome.put("Helsinki", 2202);
        rome.put("Lisbon", 1873);
        rome.put("London", 1444);
        rome.put("Luxembourg", 995);
        rome.put("Madrid", 1377);
        rome.put("Paris", 1117);
        rome.put("Rome", 0);
        rome.put("Stockholm", 1984);
        rome.put("Vienna", 765);
        DISTANCES.put("Rome", rome);

        Map<String, Integer> stockholm = new HashMap<String, Integer>(20);
        stockholm.put("Amsterdam", 1132);
        stockholm.put("Athens", 2410);
        stockholm.put("Berlin", 818);
        stockholm.put("Brussels", 1284);
        stockholm.put("Copenhagen", 526);
        stockholm.put("Dublin", 1625);
        stockholm.put("Helsinki", 396);
        stockholm.put("Lisbon", 2993);
        stockholm.put("London", 1436);
        stockholm.put("Luxembourg", 1325);
        stockholm.put("Madrid", 2596);
        stockholm.put("Paris", 1549);
        stockholm.put("Rome", 1984);
        stockholm.put("Stockholm", 0);
        stockholm.put("Vienna", 1247);
        DISTANCES.put("Stockholm", stockholm);

        Map<String, Integer> vienna = new HashMap<String, Integer>(20);
        vienna.put("Amsterdam", 938);
        vienna.put("Athens", 1280);
        vienna.put("Berlin", 525);
        vienna.put("Brussels", 917);
        vienna.put("Copenhagen", 876);
        vienna.put("Dublin", 1687);
        vienna.put("Helsinki", 1439);
        vienna.put("Lisbon", 2300);
        vienna.put("London", 1237);
        vienna.put("Luxembourg", 761);
        vienna.put("Madrid", 1812);
        vienna.put("Paris", 1037);
        vienna.put("Rome", 765);
        vienna.put("Stockholm", 1247);
        vienna.put("Vienna", 0);
        DISTANCES.put("Vienna", vienna);
    }

    @Override
    public List<String> getKnownCities()
    {
        List<String> cities = new ArrayList<String>(DISTANCES.keySet());
        Collections.sort(cities);
        return cities;
    }

    @Override
    public int getDistance(String startingCity, String destinationCity)
    {
        return DISTANCES.get(startingCity).get(destinationCity);
    }
}

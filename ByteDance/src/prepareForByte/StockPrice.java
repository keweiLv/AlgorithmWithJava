package prepareForByte;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * 股票价格波动
 */

public class StockPrice {
    int cur;
    Map<Integer, Integer> map = new HashMap<>();
    TreeMap<Integer, Integer> treeMap = new TreeMap<>();

    public void update(int timestamp, int price) {
        cur = Math.max(cur, timestamp);
        if (map.containsKey(timestamp)) {
            int old = map.get(timestamp);
            int cnt = treeMap.get(old);
            if (cnt == 1) {
                treeMap.remove(old);
            } else {
                treeMap.put(old, cnt - 1);
            }
        }
        map.put(timestamp, price);
        treeMap.put(price, treeMap.getOrDefault(price, 0) + 1);
    }

    public int current() {
        return map.get(cur);
    }

    public int maximum() {
        return treeMap.lastKey();
    }

    public int minimum() {
        return treeMap.firstKey();
    }
}

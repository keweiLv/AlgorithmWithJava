package newStart;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * 价格波动
 */
public class StockPrice {

    int cur;
    Map<Integer, Integer> map = new HashMap<>();
    TreeMap<Integer, Integer> tm = new TreeMap<>();

    public StockPrice() {
    }

    public void update(int timestamp, int price) {
        cur = Math.max(cur, timestamp);
        if (map.containsKey(timestamp)) {
            int old = map.get(timestamp);
            int cnt = tm.get(old);
            if (cnt == 1) {
                tm.remove(old);
            } else {
                tm.put(old, cnt - 1);
            }
        }
        map.put(timestamp, price);
        tm.put(price, tm.getOrDefault(price, 0) + 1);
    }

    public int current() {
        return map.get(cur);
    }

    public int maximum() {
        return tm.lastKey();
    }

    public int minimum() {
        return tm.firstKey();
    }
}

package newStart;

import java.util.LinkedHashMap;

public class LRUCache {

    int cap;
    LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();

    public LRUCache(int capacity) {
        this.cap = capacity;
    }

    public int get(int key) {
        if (!cache.containsKey(key)) {
            return -1;
        }
        makeRecently(key);
        return cache.get(key);
    }

    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            cache.put(key, value);
            makeRecently(key);
            return;
        }
        if (cache.size() >= this.cap) {
            Integer next = cache.keySet().iterator().next();
            cache.remove(next);
        }
        cache.put(key, value);
    }

    private void makeRecently(int key) {
        int val = cache.get(key);
        cache.remove(key);
        cache.put(key, val);
    }
}

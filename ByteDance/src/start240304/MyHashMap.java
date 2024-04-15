package start240304;

import java.util.Iterator;
import java.util.LinkedList;

public class MyHashMap {

    private class Pair {
        private int key;
        private int value;

        public Pair(int key, int value) {
            this.key = key;
            this.value = value;
        }

        public int getKey() {
            return key;
        }

        public int getValue() {
            return value;
        }

        public void setValue(int value) {
            this.value = value;
        }
    }

    private static final int BASE = 769;
    private LinkedList[] data;

    public MyHashMap() {
        data = new LinkedList[BASE];
        for (int i = 0; i < BASE; i++) {
            data[i] = new LinkedList<Pair>();
        }
    }

    public void put(int key, int value) {
        int h = hash(key);
        Iterator<Pair> iterator = data[h].iterator();
        while (iterator.hasNext()) {
            Pair next = iterator.next();
            if (next.getKey() == key) {
                next.setValue(value);
                return;
            }
        }
        data[h].offerLast(new Pair(key, value));
    }

    private int hash(int key) {
        return key % BASE;
    }

    public int get(int key) {
        int h = hash(key);
        Iterator<Pair> iterator = data[h].iterator();
        while (iterator.hasNext()) {
            Pair next = iterator.next();
            if (next.getKey() == key) {
                return next.value;
            }
        }
        return -1;
    }

    public void remove(int key) {
        int h = hash(key);
        Iterator<Pair> iterator = data[h].iterator();
        while (iterator.hasNext()) {
            Pair next = iterator.next();
            if (next.key == key) {
                data[h].remove(next);
                return;
            }
        }
    }
}

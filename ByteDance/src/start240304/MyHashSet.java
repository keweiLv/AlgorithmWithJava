package start240304;

/**
 * 设计哈希集合
 */
public class MyHashSet {

    boolean[] nodes = new boolean[1000009];

    public MyHashSet() {

    }

    public void add(int key) {
        nodes[key] = true;
    }

    public void remove(int key) {
        nodes[key] = false;
    }

    public boolean contains(int key) {
        return nodes[key];
    }
}

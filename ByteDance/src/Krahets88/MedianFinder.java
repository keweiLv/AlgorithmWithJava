package Krahets88;

import java.util.PriorityQueue;

/**
 * 数据流的中位数
 */
public class MedianFinder {

    PriorityQueue<Integer> l = new PriorityQueue<>((a, b) -> b - a);
    PriorityQueue<Integer> r = new PriorityQueue<>((a, b) -> a - b);

    public MedianFinder() {

    }

    public void addNum(int num) {
        int s1 = l.size(), s2 = r.size();
        if (s1 == s2) {
            if (r.isEmpty() || num <= r.peek()) {
                l.add(num);
            } else {
                l.add(r.poll());
                r.add(num);
            }
        } else {
            if (l.peek() <= num) {
                r.add(num);
            } else {
                r.add(l.poll());
                l.add(num);
            }
        }
    }

    public double findMedian() {
        int s1 = l.size(), s2 = r.size();
        if (s1 == s2) {
            return (l.peek() + r.peek()) / 2.0;
        } else {
            return l.peek();
        }
    }
}

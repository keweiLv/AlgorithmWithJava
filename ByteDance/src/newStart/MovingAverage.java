package newStart;

import java.util.Deque;
import java.util.LinkedList;

/**
 * 数据流中的移动平均值
 */
public class MovingAverage {


    int size;
    Deque<Integer> deque;
    int sum;

    /**
     * Initialize your data structure here.
     */
    public MovingAverage(int size) {
        this.size = size;
        deque = new LinkedList<>();
        sum = 0;
    }

    public double next(int val) {
        deque.addLast(val);
        sum += val;
        if (deque.size() > size) {
            Integer temp = deque.removeFirst();
            sum -= temp;
        }
        return (double) sum / deque.size();
    }

}

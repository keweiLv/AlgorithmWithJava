package newStart;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * 数据流中的移动平均值
 */
public class MovingAverage {

    Queue<Integer> queue;
    int size;
    double sum;

    public MovingAverage(int size) {
        queue = new ArrayDeque<>();
        this.size = size;
        sum = 0;
    }

    public double next(int val) {
        if (queue.size() == size) {
            sum -= queue.poll();
        }
        queue.offer(val);
        sum += val;
        return sum / queue.size();
    }
}

package swordFingerProvided;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * @author Kezi
 * @date 2022年07月17日 22:28
 * 最近请求次数
 */
public class RecentCounter {
	Queue<Integer> queue;

	public RecentCounter() {
		queue = new ArrayDeque<>();
	}

	public int ping(int t) {
		queue.offer(t);
		while (queue.peek() < t - 3000){
			queue.poll();
		}
		return queue.size();
	}
}

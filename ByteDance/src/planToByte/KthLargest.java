package planToByte;

import java.util.PriorityQueue;

/**
 * @author Kezi
 * @date 2023年02月03日 23:48
 * 数据流中diK大的元素
 */
public class KthLargest {
	PriorityQueue<Integer> pq;
	int k;

	public KthLargest(int k, int[] nums) {
		this.k = k;
		pq = new PriorityQueue<>();
		for (int num : nums) {
			add(num);
		}
	}
	public int add(int val) {
		pq.offer(val);
		if (pq.size() > k) {
			pq.poll();
		}
		return pq.peek();
	}

}

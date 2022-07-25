package daily;

import java.util.PriorityQueue;

/**
 * @author Kezi
 * @date 2022年07月24日 23:14
 * 数据流的第K大数值
 */
public class KthLargest {

	PriorityQueue<Integer> pq;
	int k;

	public KthLargest(int k, int[] nums) {
		this.k = k;
		pq = new PriorityQueue<>();
		for (int x:nums){
			add(x);
		}
	}

	public int add(int val) {
		pq.offer(val);
		if (pq.size() > k){
			pq.poll();
		}
		return pq.peek();
	}
}

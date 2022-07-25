package daily;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * @author Kezi
 * @date 2022年07月23日 22:39
 */
public class SolutionTwo {

	// 公交站台的距离
	public int distanceBetweenBusStops(int[] dist, int s, int t) {
		int n = dist.length, i = s, j = s, a = 0, b = 0;
		while (i != t) {
			a += dist[i];
			if (++i == n) {
				i = 0;
			}
		}
		while (j != t) {
			if (--j < 0) {
				j = n - 1;
			}
			b += dist[t];
		}
		return Math.min(a, b);
	}

	// 出现频率最高的K个数字
	public int[] topKFrequent(int[] nums, int k) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int num : nums) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}
		// 使用优先队列
		PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			int num = entry.getKey(), cnt = entry.getValue();
			pq.offer(new int[]{num, cnt});
			if (pq.size() > k) {
				pq.poll();
			}
		}
		int[] ans = new int[pq.size()];
		for (int i = 0; i < k; i++) {
			ans[i] = pq.poll()[0];
		}
		return ans;
	}
}

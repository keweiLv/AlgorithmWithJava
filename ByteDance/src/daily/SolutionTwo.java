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

	// 山峰数组额顶部
	public int peakIndexInMountainArray(int[] arr) {
		int n = arr.length;
		int left = 1, right = n - 2, ans = 0;
		while (left <= right) {
			int mid = (left + right) / 2;
			if (arr[mid] > arr[mid + 1]) {
				ans = mid;
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return ans;
	}

	// 排序数组中只出现一次的数字
	public int singleNonDuplicate(int[] nums) {
		int n = nums.length, l = 0, r = n - 1;
		int ans = -1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (mid < n - 1 && nums[mid] == nums[mid + 1]) {
				if (mid % 2 == 0) {
					l = mid + 2;
				} else {
					r = mid - 1;
				}
			} else if (mid > 0 && nums[mid] == nums[mid - 1]) {
				if (mid % 2 == 0) {
					r = mid - 2;
				} else {
					l = mid + 1;
				}
			}else {
				ans = nums[mid];
				break;
			}
		}
		return ans;
	}


}
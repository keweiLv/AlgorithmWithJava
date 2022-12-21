package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年12月19日 22:34
 */
public class AnotherSolution {
	// 最长重复子串
	long[] h, p;

	public String longestDupSubstring(String s) {
		int P = 13131, n = s.length();
		h = new long[n + 10];
		p = new long[n + 10];
		p[0] = 1;
		for (int i = 0; i < n; i++) {
			p[i + 1] = p[i] * P;
			h[i + 1] = h[i] * P + s.charAt(i);
		}
		String ans = "";
		int l = 0, r = n;
		while (l < r) {
			int mid = l + r + 1 >> 1;
			String t = check(s, mid);
			if (t.length() != 0) {
				l = mid;
			} else {
				r = mid - 1;
			}
			ans = t.length() > ans.length() ? t : ans;
		}
		return ans;
	}

	private String check(String s, int len) {
		int n = s.length();
		Set<Long> set = new HashSet<>();
		for (int i = 1; i + len - 1 <= n; i++) {
			int j = i + len - 1;
			long cur = h[j] - h[i - 1] * p[j - i + 1];
			if (set.contains(cur)) {
				return s.substring(i - 1, j);
			}
			set.add(cur);
		}
		return "";
	}

	// 寻求图中是否存在路径
	int[] pathP;

	public boolean validPath(int n, int[][] edges, int source, int destination) {
		pathP = new int[n];
		for (int i = 0; i < n; i++) {
			pathP[i] = i;
		}
		for (int[] e : edges) {
			pathP[find(e[0])] = find(e[1]);
		}
		return find(source) == find(destination);
	}

	private int find(int i) {
		if (pathP[i] != i) {
			pathP[i] = find(pathP[i]);
		}
		return pathP[i];
	}

	// 转化数字的最小幸运数
	public int minimumOperations(int[] nums, int start, int goal) {
		Deque<Integer> deque = new ArrayDeque<>();
		Map<Integer, Integer> map = new HashMap<>();
		deque.addLast(start);
		map.put(start, 0);
		while (!deque.isEmpty()) {
			int cur = deque.pollFirst();
			int step = map.get(cur);
			for (int num : nums) {
				int[] result = new int[]{cur + num, cur - num, cur ^ num};
				for (int next : result) {
					if (next == goal) {
						return step + 1;
					}
					if (next < 0 || next > 1000) {
						continue;
					}
					if (map.containsKey(next)) {
						continue;
					}
					map.put(next, step + 1);
					deque.addLast(next);
				}
			}
		}
		return -1;
	}

	// 移除石子的最大得分
	public int maximumScore(int a, int b, int c) {
		int[] rec = new int[]{a, b, c};
		Arrays.sort(rec);
		if (rec[0] + rec[1] <= rec[2]) {
			return rec[0] + rec[1];
		} else {
			return (a + b + c) >> 1;
		}
	}
}

package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年12月30日 22:22
 * @deprecated 考场就坐
 */
public class ExamRoom {

	private TreeSet<int[]> ts = new TreeSet<>((a, b) -> {
		int d1 = dist(a), d2 = dist(b);
		return d1 == d2 ? a[0] - b[0] : d2 - d1;
	});
	private Map<Integer, Integer> left = new HashMap<>();
	private Map<Integer, Integer> right = new HashMap<>();
	private int n;

	public ExamRoom(int n) {
		this.n = n;
		add(new int[]{-1, n});
	}

	public int seat() {
		int[] s = ts.first();
		int p = (s[0] + s[1]) >> 1;
		if (s[0] == -1) {
			p = 0;
		} else if (s[1] == n) {
			p = n - 1;
		}
		del(s);
		add(new int[]{s[0], p});
		add(new int[]{p, s[1]});
		return p;
	}

	public void leave(int p) {
		int l = left.get(p), r = right.get(p);
		del(new int[]{l, p});
		del(new int[]{p, r});
		add(new int[]{l, r});
	}

	private void add(int[] s) {
		ts.add(s);
		left.put(s[1], s[0]);
		right.put(s[0], s[1]);
	}

	private void del(int[] s) {
		ts.remove(s);
		left.remove(s[1]);
		right.remove(s[0]);
	}

	private int dist(int[] s) {
		int l = s[0], r = s[1];
		return l == -1 || r == n ? r - l - 1 : (r - l) >> 1;
	}

	// 第一次出现两次的字母
	public char repeatedCharacter(String s) {
		int mask = 0;
		for (char c : s.toCharArray()) {
			int t = 1 << (c - 'a');
			if ((mask & t) != 0) {
				return c;
			}
			mask |= t;
		}
		return 0;
	}

	// 组合
	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> ans = new ArrayList<>();
		if (k <= 0 || n < k) {
			return ans;
		}
		Deque<Integer> deque = new ArrayDeque<>();
		dfs(n, k, 1, deque, ans);
		return ans;
	}

	private void dfs(int n, int k, int index, Deque<Integer> path, List<List<Integer>> ans) {
		if (path.size() == k) {
			ans.add(new ArrayList<>(path));
			return;
		}
		for (int i = index; i <= n - (k - path.size()) + 1; i++) {
			path.addLast(i);
			dfs(n, k, i + 1, path, ans);
			path.removeLast();
		}
	}

	// 积压订单中的订单总数
	public int getNumberOfBacklogOrders(int[][] orders) {
		PriorityQueue<int[]> buy = new PriorityQueue<>((a, b) -> b[0] - a[0]);
		PriorityQueue<int[]> sell = new PriorityQueue<>((a, b) -> a[0] - b[0]);
		for (int[] order : orders) {
			int p = order[0], a = order[1], t = order[2];
			if (t == 0) {
				while (a > 0 && !sell.isEmpty() && sell.peek()[0] <= p) {
					int[] q = sell.poll();
					int x = q[0], y = q[1];
					if (a >= y) {
						a -= y;
					} else {
						sell.offer(new int[]{x, y - a});
						a = 0;
					}
				}
				if (a > 0) {
					buy.offer(new int[]{p, a});
				}
			} else {
				while (a > 0 && !buy.isEmpty() && buy.peek()[0] >= p) {
					int[] q = buy.poll();
					int x = q[0], y = q[1];
					if (a >= y) {
						a -= y;
					} else {
						buy.offer(new int[]{x, y - a});
						a = 0;
					}
				}
				if (a > 0){
					sell.offer(new int[]{p,a});
				}
			}
		}
		long ans = 0;
		final int mod = (int) 1e9+7;
		while (!buy.isEmpty()){
			ans += buy.poll()[1];
		}
		while (!sell.isEmpty()){
			ans += sell.poll()[1];
		}
		return (int) (ans % mod);
	}
}

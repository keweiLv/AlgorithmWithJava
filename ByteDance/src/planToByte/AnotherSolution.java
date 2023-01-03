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

	// 丑数Ⅱ
	int[] nums = new int[]{2, 3, 5};

	public int nthUglyNumber(int n) {
		Set<Long> set = new HashSet<>();
		Queue<Long> pq = new PriorityQueue<>();
		set.add(1L);
		pq.add(1L);
		for (int i = 1; i <= n; i++) {
			long x = pq.poll();
			if (i == n) {
				return (int) x;
			}
			for (int num : nums) {
				long t = num * x;
				if (!set.contains(t)) {
					set.add(t);
					pq.add(t);
				}
			}
		}
		return -1;
	}

	// 超级丑数
	public int nthSuperUglyNumber(int n, int[] primes) {
		Queue<Integer> pq = new PriorityQueue<>();
		pq.add(1);
		while (n-- > 0) {
			int x = pq.poll();
			if (n == 0) {
				return x;
			}
			for (int k : primes) {
				if (k <= Integer.MAX_VALUE / x) {
					pq.add(k * x);
				}
				if (x % k == 0) {
					break;
				}
			}
		}
		return -1;
	}

	// 构造字典序最大的合并字符串
	public String largestMerge(String word1, String word2) {
		StringBuilder sb = new StringBuilder();
		while (word1.length() + word2.length() > 0) {
			if (word1.compareTo(word2) > 0) {
				sb.append(word1.charAt(0));
				word1 = word1.substring(1);
			} else {
				sb.append(word2.charAt(0));
				word2 = word2.substring(1);
			}
		}
		return sb.toString();
	}

	// 放置盒子
	public int minimumBoxes(int n) {
		int s = 0, k = 1;
		while (s + k * (k + 1) / 2 <= n) {
			s += k * (k + 1) / 2;
			++k;
		}
		--k;
		int ans = k * (k + 1) / 2;
		k = 1;
		while (s < n) {
			++ans;
			s += k;
			++k;
		}
		return ans;
	}

	// 序列化二叉树
	// Encodes a tree to a single string.
	public String serialize1(TreeNode root) {
		if (root == null) {
			return "[]";
		}
		StringBuilder res = new StringBuilder("[");
		Queue<TreeNode> queue = new LinkedList<TreeNode>() {{
			add(root);
		}};
		while (!queue.isEmpty()) {
			TreeNode node = queue.poll();
			if (node != null) {
				res.append(node.val + ",");
				queue.add(node.left);
				queue.add(node.right);
			} else {
				res.append("null,");
			}
		}
		res.deleteCharAt(res.length() - 1);
		res.append("]");
		return res.toString();
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize1(String data) {
		if (data.equals("[]")) {
			return null;
		}
		String[] vals = data.substring(1, data.length() - 1).split(",");
		TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
		Queue<TreeNode> queue = new LinkedList<TreeNode>() {{
			add(root);
		}};
		int i = 1;
		while (!queue.isEmpty()) {
			TreeNode node = queue.poll();
			if (!vals[i].equals("null")) {
				node.left = new TreeNode(Integer.parseInt(vals[i]));
				queue.add(node.left);
			}
			i++;
			if (!vals[i].equals("null")) {
				node.right = new TreeNode(Integer.parseInt(vals[i]));
				queue.add(node.right);
			}
			i++;
		}
		return root;
	}

	// 序列化和反序列化二查搜索树
	// Encodes a tree to a single string.
	public String serialize2(TreeNode root) {
		if (root == null) {
			return null;
		}
		List<String> list = new ArrayList<>();
		dfs1(root, list);
		int n = list.size();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++) {
			sb.append(list.get(i));
			if (i != n - 1) {
				sb.append(",");
			}
		}
		return sb.toString();
	}

	private void dfs1(TreeNode root, List<String> list) {
		if (root == null) {
			return;
		}
		list.add(String.valueOf(root.val));
		dfs1(root.left, list);
		dfs1(root.right, list);
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize2(String data) {
		if (data == null) {
			return null;
		}
		String[] ss = data.split(",");
		return dfs2(0, ss.length - 1, ss);
	}

	private TreeNode dfs2(int l, int r, String[] ss) {
		if (l > r) {
			return null;
		}
		int j = l + 1, t = Integer.parseInt(ss[l]);
		TreeNode ans = new TreeNode(t);
		while (j <= r && Integer.parseInt(ss[j]) <= t) {
			j++;
		}
		ans.left = dfs2(l + 1, j - 1, ss);
		ans.right = dfs2(j, r, ss);
		return ans;
	}

	// 统计同构子字符串的数目
	private static final int MOD = (int) 1e9 + 7;

	public int countHomogenous(String s) {
		int n = s.length();
		long ans = 0;
		for (int i = 0, j = 0; i < n; i = j) {
			j = i;
			while (j < n && s.charAt(j) == s.charAt(i)) {
				j++;
			}
			int cnt = j - i;
			ans += (long) (1 + cnt) * cnt / 2;
			ans %= MOD;
		}
		return (int) ans;
	}

	// 网络延迟时间
	public int networkDelayTime(int[][] times, int n, int k) {
		final int INF = Integer.MAX_VALUE / 2;
		int[][] g = new int[n][n];
		for (int i = 0; i < n; i++) {
			Arrays.fill(g[i], INF);
		}
		for (int[] t : times) {
			int x = t[0] - 1, y = t[1] - 1;
			g[x][y] = t[2];
		}
		int[] dist = new int[n];
		Arrays.fill(dist, INF);
		dist[k - 1] = 0;
		boolean[] used = new boolean[n];
		for (int i = 0; i < n; i++) {
			int x = -1;
			for (int y = 0; y < n; ++y) {
				if (!used[y] && (x == -1 || dist[y] < dist[x])) {
					x = y;
				}
			}
			used[x] = true;
			for (int y = 0; y < n; ++y) {
				dist[y] = Math.min(dist[y], dist[x] + g[x][y]);
			}
		}
		int ans = Arrays.stream(dist).max().getAsInt();
		return ans == INF ? -1 : ans;
	}

	// 转换字符串的最少操作次数
	public int minimumMoves(String s) {
		int ans = 0;
		for (int i = 0; i < s.length(); ++i) {
			if (s.charAt(i) == 'X') {
				++ans;
				i += 2;
			}
		}
		return ans;
	}

	// 颠倒二进制位
	public int reverseBits(int n) {
		int rev = 0;
		for (int i = 0; i < 32 && n != 0; ++i) {
			rev |= (n & 1) << (31 - i);
			n >>>= 1;
		}
		return rev;
	}

	// 删除字符串两端相同字符后的最短长度
	public int minimumLength(String S) {
		int l = 0, r = S.length() - 1;
		char[] s = S.toCharArray();
		while (l < r && s[l] == s[r]) {
			char c = s[l];
			while (l <= r && s[l] == c) {
				l++;
			}
			while (l <= r && s[r] == c) {
				r--;
			}
		}
		return r - l + 1;
	}

	// 旋转矩阵
	public void rotate(int[][] matrix) {
		int n = matrix.length;
		// 水平翻转
		for (int i = 0; i < n / 2; ++i) {
			for (int j = 0; j < n; ++j) {
				int temp = matrix[i][j];
				matrix[i][j] = matrix[n - i - 1][j];
				matrix[n - i - 1][j] = temp;
			}
		}
		// 主对角线翻转
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				int temp = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = temp;
			}
		}
	}

	// 访问所有节点的最短路径
	public int shortestPathLength(int[][] graph) {
		int n = graph.length;
		// 三个属性分别为 idx, mask, dist
		Queue<int[]> queue = new LinkedList<>();
		boolean[][] vis = new boolean[n][1 << n];
		for (int i = 0; i < n; i++) {
			queue.offer(new int[]{i, 1 << i, 0});
			vis[i][1 << i] = true;
		}
		while (!queue.isEmpty()) {
			int[] temp = queue.poll();
			int idx = temp[0], mask = temp[1], dist = temp[2];
			if (mask == (1 << n) - 1) {
				return dist;
			}
			for (int x : graph[idx]) {
				int nextMask = mask | (1 << x);
				if (!vis[x][nextMask]) {
					queue.offer(new int[]{x, nextMask, dist + 1});
					vis[x][nextMask] = true;
				}
			}
		}
		return 0;
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

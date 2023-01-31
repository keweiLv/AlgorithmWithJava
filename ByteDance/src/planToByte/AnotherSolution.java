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
				if (a > 0) {
					sell.offer(new int[]{p, a});
				}
			}
		}
		long ans = 0;
		final int mod = (int) 1e9 + 7;
		while (!buy.isEmpty()) {
			ans += buy.poll()[1];
		}
		while (!sell.isEmpty()) {
			ans += sell.poll()[1];
		}
		return (int) (ans % mod);
	}

	// 有界数组中指定下标的最大值
	public int maxValue(int n, int index, int maxSum) {
		int left = 1, right = maxSum;
		while (left < right) {
			int mid = (left + right + 1) >> 1;
			if (sum(mid - 1, index) + sum(mid, n - index) <= maxSum) {
				left = mid;
			} else {
				right = mid - 1;
			}
		}
		return left;
	}

	private long sum(long x, int cnt) {
		return x >= cnt ? (x + x - cnt + 1) * cnt / 2 : (x + 1) * x / 2 + cnt - x;
	}

	// 分隔链表
	public ListNode partition(ListNode head, int x) {
		ListNode small = new ListNode(0);
		ListNode smallHead = small;
		ListNode large = new ListNode(0);
		ListNode largeHead = large;
		while (head != null) {
			if (head.val < x) {
				small.next = head;
				small = small.next;
			} else {
				large.next = head;
				large = large.next;
			}
			head = head.next;
		}
		large.next = null;
		small.next = largeHead.next;
		return smallHead.next;
	}

	// 将x 减到0的最小操作数
	public int minOperations(int[] nums, int x) {
		int target = -x;
		for (int num : nums) {
			target += num;
		}
		if (target < 0) {
			return -1;
		}
		int ans = -1, left = 0, sum = 0, n = nums.length;
		for (int right = 0; right < n; right++) {
			sum += nums[right];
			while (sum > target) {
				sum -= nums[left++];
			}
			if (sum == target) {
				ans = Math.max(ans, right - left + 1);
			}
		}
		return ans < 0 ? -1 : n - ans;
	}

	// 还原排列的最少操作步数
	public int reinitializePermutation(int n) {
		int ans = 0;
		for (int i = 1; ; ) {
			if ((i & 1) == 0) {
				i >>= 1;
			} else {
				i = (n >> 1) + (i - 1 >> 1);
			}
			++ans;
			if (i == 1) {
				break;
			}
		}
		return ans;
	}

	// 最长重复子数组
	public int findLength(int[] nums1, int[] nums2) {
		int n = nums1.length, m = nums2.length;
		int ret = 0;
		for (int i = 0; i < n; i++) {
			int len = Math.min(m, n - i);
			int maxLen = maxLength(nums1, nums2, i, 0, len);
			ret = Math.max(ret, maxLen);
		}
		for (int i = 0; i < m; i++) {
			int len = Math.min(n, m - i);
			int maxLen = maxLength(nums1, nums2, 0, i, len);
			ret = Math.max(ret, maxLen);
		}
		return ret;
	}

	private int maxLength(int[] nums1, int[] nums2, int addA, int addB, int len) {
		int ret = 0, k = 0;
		for (int i = 0; i < len; i++) {
			if (nums1[addA + i] == nums2[addB + i]) {
				k++;
			} else {
				k = 0;
			}
			ret = Math.max(ret, k);
		}
		return ret;
	}

	// 判断一个数的数字计数是否等于数位的值
	public boolean digitCount(String num) {
		int[] tmp = new int[num.length()];
		for (char c : num.toCharArray()) {
			tmp[c - '0']++;
		}
		for (int i = 0; i < num.length(); i++) {
			if (tmp[i] != num.charAt(i) - '0') {
				return false;
			}
		}
		return true;
	}

	/**
	 * 最佳买卖股票时机含冷冻期
	 * dp[0][0]=0;//本来就不持有，啥也没干
	 * dp[0][1]=-1*prices[0];//第0天只买入
	 * dp[0][2]=0; 卖出(第0天买入又卖出)
	 */
	public int maxProfit(int[] prices) {
		int n = prices.length;
		if (n <= 1) {
			return 0;
		}
		int[][] dp = new int[n][3];
		dp[0][0] = 0;
		dp[0][1] = -1 * prices[0];
		dp[0][2] = 0;
		for (int i = 1; i < n; i++) {
			dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][2]);
			dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
			dp[i][2] = dp[i - 1][1] + prices[i];
		}
		return Math.max(dp[n - 1][0], dp[n - 1][2]);
	}

	// 替换字符串中的括号内容
	public String evaluate(String s, List<List<String>> knowledge) {
		Map<String, String> map = new HashMap<>(knowledge.size());
		for (List<String> list : knowledge) {
			map.put(list.get(0), list.get(1));
		}
		StringBuilder ans = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				int j = s.indexOf(')', i + 1);
				ans.append(map.getOrDefault(s.substring(i + 1, j), "?"));
				i = j;
			} else {
				ans.append(s.charAt(i));
			}
		}
		return ans.toString();
	}

	// 重排字符形成目标字符串
	public int rearrangeCharacters(String s, String target) {
		int[] cnt1 = new int[26];
		int[] cnt2 = new int[26];
		for (int i = 0; i < s.length(); i++) {
			cnt1[s.charAt(i) - 'a']++;
		}
		for (int i = 0; i < target.length(); i++) {
			cnt2[target.charAt(i) - 'a']++;
		}
		int ans = 100;
		for (int i = 0; i < 26; i++) {
			if (cnt2[i] > 0) {
				ans = Math.min(ans, cnt1[i] / cnt2[i]);
			}
		}
		return ans;
	}

	// 寻找重复数
	public int findDuplicate(int[] nums) {
		int len = nums.length;
		int left = 1;
		int right = len - 1;
		while (left < right) {
			int mid = (left + right) / 2;
			int count = 0;
			for (int num : nums) {
				if (num <= mid) {
					count++;
				}
			}
			if (count > mid) {
				right = mid;
			} else {
				left = mid + 1;
			}
		}
		return left;
	}

	// 极大极小游戏
	public int minMaxGame(int[] nums) {
		int n = nums.length;
		if (n == 1) {
			return nums[0];
		}
		int[] newNums = new int[n / 2];
		for (int i = 0; i < newNums.length; i++) {
			if (i % 2 == 0) {
				newNums[i] = Math.min(nums[2 * i], nums[2 * i + 1]);
			} else {
				newNums[i] = Math.max(nums[2 * i], nums[2 * i + 1]);
			}
		}
		return minMaxGame(newNums);
	}

	// 回文子串
	public int countSubstrings(String s) {
		int ans = 0;
		int n = s.length();
		for (int i = 0; i < n; i++) {
			// 中心扩展法，实则就是遍历，j=0,中心是一个点，j=1,中心是两个点
			for (int j = 0; j <= 1; j++) {
				int l = i;
				int r = i + j;
				while (l >= 0 && r < n && s.charAt(l--) == s.charAt(r++)) {
					ans++;
				}
			}
		}
		return ans;
	}

	/**
	 * 最长回文子串
	 * i为回文子串的中间位置 减去 回文子串的一半长度 就等于回文子串的start；End同理。至于为什么len-1，我理解的是单字符为轴的回文子串减不减没关系，但是已双字符为轴的回文子串不减的话算出来start的位置出错了。
	 */
	public static String longestPalindrome(String s) {
		int start = 0, end = 0;
		int n = s.length();
		for (int i = 0; i < n; i++) {
			int len1 = expandAroundCenter(s, i, i);
			int len2 = expandAroundCenter(s, i, i + 1);
			int len = Math.max(len1, len2);
			if (len > end - start) {
				start = i - (len - 1) / 2;
				end = i + len / 2;
			}
		}
		return s.substring(start, end + 1);
	}

	private static int expandAroundCenter(String s, int left, int right) {
		int L = left, R = right;
		while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
			L--;
			R++;
		}
		// 因为用的while，所以最后退出循环时L和R时已经不符合要求，真正子串长度是R-L-1;
		return R - L - 1;
	}

	// 句子相似性
	public static boolean areSentencesSimilar(String sentence1, String sentence2) {
		if (sentence1.length() > sentence2.length()) {
			return areSentencesSimilar(sentence2, sentence1);
		}
		String[] a1 = sentence1.split(" "), a2 = sentence2.split(" ");
		int n = a1.length, m = a2.length, l = 0, r = 0;
		while (l < n && a1[l].equals(a2[l])) {
			l++;
		}
		while (r < n - l && a1[n - r - 1].equals(a2[m - r - 1])) {
			r++;
		}
		return l + r == n;
	}

	// 优势洗牌
	public int[] advantageCount(int[] nums1, int[] nums2) {
		int n = nums1.length;
		TreeSet<Integer> tset = new TreeSet<>();
		Map<Integer, Integer> map = new HashMap<>(16);
		for (int x : nums1) {
			map.put(x, map.getOrDefault(x, 0) + 1);
			if (map.get(x) == 1) {
				tset.add(x);
			}
		}
		int[] ans = new int[n];
		for (int i = 0; i < n; i++) {
			Integer cur = tset.ceiling(nums2[i] + 1);
			if (cur == null) {
				cur = tset.ceiling(-1);
			}
			ans[i] = cur;
			map.put(cur, map.get(cur) - 1);
			if (map.get(cur) == 0) {
				tset.remove(cur);
			}
		}
		return ans;
	}

	// 统计一个数组中好对子的数目
	public int countNicePairs(int[] nums) {
		final int MOD = (int) 1e9 + 7;
		int count = 0;
		Map<Integer, Integer> map = new HashMap<>();
		for (int num : nums) {
			int temp = num, rev = 0;
			while (temp > 0) {
				rev = rev * 10 + temp % 10;
				temp /= 10;
			}
			count = (count + map.getOrDefault(num - rev, 0)) % MOD;
			map.put(num - rev, map.getOrDefault(num - rev, 0) + 1);
		}
		return count;
	}

	// 和为K的子数组
	public int subarraySum(int[] nums, int k) {
		int count = 0, pre = 0;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, 1);
		for (int num : nums) {
			pre += num;
			if (map.containsKey(pre - k)) {
				count += map.get(pre - k);
			}
			map.put(pre, map.getOrDefault(pre, 0) + 1);
		}
		return count;
	}

	// 课程表
	public boolean canFinish(int numCourses, int[][] prerequisites) {
		Map<Integer, Integer> in = new HashMap<>();
		for (int i = 0; i < numCourses; i++) {
			in.put(i, 0);
		}
		Map<Integer, List<Integer>> adj = new HashMap<>();
		for (int[] site : prerequisites) {
			int next = site[0];
			int cur = site[1];
			in.put(next, in.get(next) + 1);
			if (!adj.containsKey(cur)) {
				adj.put(cur, new ArrayList<>());
			}
			adj.get(cur).add(next);
		}
		Queue<Integer> queue = new LinkedList<>();
		for (int key : in.keySet()) {
			if (in.get(key) == 0) {
				queue.offer(key);
			}
		}
		while (!queue.isEmpty()) {
			int cur = queue.poll();
			if (!adj.containsKey(cur)) {
				continue;
			}
			List<Integer> succ = adj.get(cur);
			for (int k : succ) {
				in.put(k, in.get(k) - 1);
				if (in.get(k) == 0) {
					queue.offer(k);
				}
			}
		}
		for (int key : in.keySet()) {
			if (in.get(key) != 0) {
				return false;
			}
		}
		return true;
	}

	// 强密码校验器二
	public boolean strongPasswordCheckerII(String password) {
		if (password.length() < 8) {
			return false;
		}
		int mask = 0;
		for (int i = 0; i < password.length(); i++) {
			char c = password.charAt(i);
			if (i > 0 && c == password.charAt(i - 1)) {
				return false;
			}
			if (Character.isLowerCase(c)) {
				mask |= 1;
			} else if (Character.isUpperCase(c)) {
				mask |= 2;
			} else if (Character.isDigit(c)) {
				mask |= 4;
			} else {
				mask |= 8;
			}
		}
		return mask == 15;
	}

	// 合并两个链表
	public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
		ListNode p = list1, q = list1;
		while (--a > 0) {
			p = p.next;
		}
		while (b-- > 0) {
			q = q.next;
		}
		p.next = list2;
		while (p.next != null) {
			p = p.next;
		}
		p.next = q.next;
		q.next = null;
		return list1;
	}

	// 判断矩阵是否是一个X矩阵
	public boolean checkXMatrix(int[][] grid) {
		int n = grid.length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; i < n; j++) {
				if ((grid[i][j] == 0) == (i == j || i + j == n - 1)) {
					return false;
				}
			}
		}
		return true;
	}
}

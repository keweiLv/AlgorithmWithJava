package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2023年02月03日 23:07
 */
public class NewSolution {

	// 二叉树着色游戏
	private int x, lsz, rsz;

	public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
		this.x = x;
		dfs(root);
		return Math.max(Math.max(lsz, rsz), n - 1 - lsz - rsz) * 2 > n;
	}

	private int dfs(TreeNode node) {
		if (node == null) {
			return 0;
		}
		int ls = dfs(node.left);
		int rs = dfs(node.right);
		if (node.val == x) {
			lsz = ls;
			rsz = rs;
		}
		return ls + rs + 1;
	}

	// 你能构造出连续值的最大数目
	public int getMaximumConsecutive(int[] coins) {
		int m = 0;
		Arrays.sort(coins);
		for (int coin : coins) {
			if (coin > m + 1) {
				break;
			}
			m += coin;
		}
		return m + 1;
	}

	// 最长连续序列
	public int longestConsecutive(int[] nums) {
		Set<Integer> set = new HashSet<>();
		for (int num : nums) {
			set.add(num);
		}
		int longestLen = 0;
		for (int num : set) {
			if (!set.contains(num - 1)) {
				int curNum = num + 1;
				int curLen = 1;
				while (set.contains(curNum + 1)) {
					curLen++;
					curNum++;
				}
				longestLen = Math.max(longestLen, curLen);
			}
		}
		return longestLen;
	}

	// 有效的括号
	private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
		put('{', '}');
		put('[', ']');
		put('(', ')');
		put('?', '?');
	}};

	public boolean isValid(String s) {
		if (s.length() > 0 && !map.containsKey(s.charAt(0))) {
			return false;
		}
		LinkedList<Character> stack = new LinkedList<Character>() {{
			add('?');
		}};
		for (Character c : s.toCharArray()) {
			if (map.containsKey(c)) {
				stack.addLast(c);
			} else if (!map.get(stack.removeLast()).equals(c)) {
				return false;
			}
		}
		return stack.size() == 1;
	}

	// 爬楼梯
	public int climbStairs(int n) {
		if (n <= 2) {
			return n;
		}
		int[] f = new int[n + 1];
		f[1] = 1;
		f[2] = 2;
		for (int i = 3; i <= n; i++) {
			f[i] = f[i - 1] + f[i - 2];
		}
		return f[n];
	}

	// 计算布尔二叉树的值
	public boolean evaluateTree(TreeNode root) {
		if (root.left == null) {
			return root.val == 1;
		}
		boolean l = evaluateTree(root.left);
		boolean r = evaluateTree(root.right);
		return root.val == 2 ? l || r : l && r;
	}

	// 警告一小时内使用相同员工卡大于等于三次的人
	public List<String> alertNames(String[] keyName, String[] keyTime) {
		Map<String, List<Integer>> timeMap = new HashMap<>();
		for (int i = 0; i < keyName.length; i++) {
			String name = keyName[i];
			String time = keyTime[i];
			timeMap.putIfAbsent(name, new ArrayList<>());
			int cnt = Integer.parseInt(time.substring(0, 2)) * 60 + Integer.parseInt(time.substring(3));
			timeMap.get(name).add(cnt);
		}
		List<String> res = new ArrayList<>();
		Set<String> strings = timeMap.keySet();
		for (String name : strings) {
			List<Integer> list = timeMap.get(name);
			Collections.sort(list);
			for (int i = 0; i < list.size(); i++) {
				int time1 = list.get(i - 2), time2 = list.get(i);
				int diff = time2 - time1;
				if (diff > 60) {
					res.add(name);
					break;
				}
			}
		}
		Collections.sort(res);
		return res;
	}

	// 删除子文件夹
	public List<String> removeSubfolders(String[] folder) {
		Arrays.sort(folder);
		List<String> res = new ArrayList<>();
		res.add(folder[0]);
		for (int i = 1; i < folder.length; i++) {
			int m = res.get(res.size() - 1).length();
			int n = folder[i].length();
			if (m >= n || !(res.get(res.size() - 1).equals(folder[i].substring(0, m)) && folder[i].charAt(m) == '/')) {
				res.add(folder[i]);
			}
		}
		return res;
	}

	// 数组能形成多少数对
	public int[] numberOfPairs(int[] nums) {
		int[] cnt = new int[101];
		for (int x : nums) {
			cnt[x]++;
		}
		int s = 0;
		for (int v : cnt) {
			s += v / 2;
		}
		return new int[]{s, nums.length - s * 2};
	}

	// 装满杯子需要的最短总时长
	public int fillCups(int[] amount) {
		Arrays.sort(amount);
		int a = amount[0], b = amount[1], c = amount[2];
		if (a + b <= c) {
			return c;
		} else {
			int t = a + b - c;
			return t % 2 == 0 ? t / 2 + c : t / 2 + c + 1;
		}
	}

	// 最大平均通过率
	public double maxAverageRatio(int[][] classes, int extraStudents) {
		PriorityQueue<double[]> pq = new PriorityQueue<>((a, b) -> {
			double x = (a[0] + 1) / (a[1] + 1) - a[0] / a[1];
			double y = (b[0] + 1) / (b[1] + 1) - b[0] / b[1];
			return Double.compare(y, x);
		});
		for (int[] item : classes) {
			pq.offer(new double[]{item[0], item[1]});
		}
		while (extraStudents-- > 0) {
			double[] poll = pq.poll();
			double a = poll[0] + 1, b = poll[1] + 1;
			pq.offer(new double[]{a, b});
		}
		double ans = 0;
		while (!pq.isEmpty()) {
			double[] poll = pq.poll();
			ans += poll[0] / poll[1];
		}
		return ans / classes.length;
	}

	// 最好的扑克手牌
	public String bestHand(int[] ranks, char[] suits) {
		boolean flush = true;
		for (int i = 1; i < 5 && flush; i++) {
			flush = suits[i] == suits[i - 1];
		}
		if (flush) {
			return "FLUSH";
		}
		int[] cnt = new int[14];
		boolean pair = false;
		for (int x : ranks) {
			if (++cnt[x] == 3) {
				return "Three of a Kind";
			}
			pair = pair || cnt[x] == 2;
		}
		return pair ? "Pair" : "High Card";
	}

	// 石子游戏Ⅱ
	public int stoneGameII(int[] piles) {
		int n = piles.length, sum = 0;
		int[][] dp = new int[n][n + 1];
		for (int i = n - 1; i >= 0; i--) {
			sum += piles[i];
			for (int M = 1; M <= n; M++) {
				if (i + 2 * M >= n) {
					dp[i][M] = sum;
				} else {
					for (int x = 1; x <= 2 * M; x++) {
						dp[i][M] = Math.max(dp[i][M], sum - dp[i + x][Math.max(M, x)]);
					}
				}
			}
		}
		return dp[0][1];
	}

	// 循环码排列
	public List<Integer> circularPermutation(int n, int start) {
		List<Integer> ans = new ArrayList<>();
		for (int i = 0; i < 1 << n; i++) {
			ans.add(i ^ (i >> 1) ^ start);
		}
		return ans;
	}

	// 使数组中所有元素都等于零
	public int minimumOperations(int[] nums) {
		boolean[] set = new boolean[101];
		set[0] = true;
		int ans = 0;
		for (int x : nums) {
			if (!set[x]) {
				ans++;
				set[x] = true;
			}
		}
		return ans;
	}

	// 递减元素使数组呈锯齿状
	public int movesToMakeZigzag(int[] nums) {
		int[] s = new int[2];
		for (int i = 0, n = nums.length; i < n; i++) {
			int left = i > 0 ? nums[i - 1] : Integer.MAX_VALUE;
			int right = i < n - 1 ? nums[i + 1] : Integer.MAX_VALUE;
			s[i % 2] += Math.max(nums[i] - Math.min(left, right) + 1, 0);
		}
		return Math.min(s[0], s[1]);
	}

	// 合并相似的物品
	public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
		int[] cnt = new int[1001];
		for (int[] x : items1) {
			cnt[x[0]] += x[1];
		}
		for (int[] x : items2) {
			cnt[x[0]] += x[1];
		}
		List<List<Integer>> ans = new ArrayList<>();
		for (int i = 0; i < cnt.length; i++) {
			if (cnt[i] > 0) {
				ans.add(Arrays.asList(i, cnt[i]));
			}
		}
		return ans;
	}

	// 连续子数组的最大和
	public int maxSubArray(int[] nums) {
		int res = nums[0];
		for (int i = 1; i < nums.length; i++) {
			nums[i] += Math.max(nums[i - 1], 0);
			res = Math.max(res, nums[i]);
		}
		return res;
	}

	// 矩阵中的局部最大值
	public int[][] largestLocal(int[][] grid) {
		int n = grid.length;
		int[][] res = new int[n - 2][n - 2];
		for (int i = 0; i < n - 2; i++) {
			for (int j = 0; j < n - 2; j++) {
				res[i][j] = localMax(grid, i, j);
			}
		}
		return res;
	}

	private int localMax(int[][] grid, int left, int top) {
		int max = 0;
		for (int i = left; i < left + 3; i++) {
			for (int j = top; j < top + 3; j++) {
				max = Math.max(max, grid[i][j]);
			}
		}
		return max;
	}

	// 青蛙过河
	Map<Integer, Integer> crossMap = new HashMap<>();
	Map<String, Boolean> cache = new HashMap<>();

	public boolean canCross(int[] ss) {
		int n = ss.length;
		for (int i = 0; i < n; i++) {
			crossMap.put(ss[i], i);
		}
		if (!crossMap.containsKey(1)) {
			return false;
		}
		return dfs(ss, n, 1, 1);
	}

	boolean dfs(int[] ss, int n, int u, int k) {
		String key = u + "_" + k;
		if (cache.containsKey(key)) {
			return cache.get(key);
		}
		if (u == n - 1) {
			return true;
		}
		for (int i = -1; i <= 1; i++) {
			if (k + i == 0) {
				continue;
			}
			int next = ss[u] + k + i;
			if (crossMap.containsKey(next)) {
				boolean cur = dfs(ss, n, crossMap.get(next), k + i);
				cache.put(key, cur);
				if (cur) {
					return true;
				}
			}
		}
		cache.put(key, false);
		return false;
	}

	// 十进制小数转二进制小数
	public String printBin(double num) {
		StringBuilder sb = new StringBuilder();
		sb.append("0.");
		while (sb.length() < 32 && num != 0) {
			num *= 2;
			int x = (int) num;
			sb.append(x);
			num -= x;
		}
		return num != 0 ? "ERROR" : sb.toString();
	}

	// 我能赢吗
	Map<Integer, Boolean> booleanMap = new HashMap<>();

	public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
		if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal) {
			return false;
		}
		return winDfs(maxChoosableInteger, 0, desiredTotal, 0);
	}

	private boolean winDfs(int maxChoosableInteger, int userNumber, int desiredTotal, int curTol) {
		if (!booleanMap.containsKey(userNumber)) {
			boolean res = false;
			for (int i = 0; i < maxChoosableInteger; i++) {
				if (((userNumber >> i) & 1) == 0) {
					if (i + 1 + curTol >= desiredTotal) {
						res = true;
						break;
					}
					if (!winDfs(maxChoosableInteger, userNumber | (1 << i), desiredTotal, curTol + i + 1)) {
						res = true;
						break;
					}
				}
			}
			booleanMap.put(userNumber, res);
		}
		return booleanMap.get(userNumber);
	}

	// 保证文件名唯一
	public String[] getFolderNames(String[] names) {
		Map<String, Integer> folderMap = new HashMap<>();
		for (int i = 0; i < names.length; i++) {
			if (folderMap.containsKey(names[i])) {
				int k = folderMap.get(names[i]);
				while (folderMap.containsKey(names[i] + "(" + k + ")")) {
					k++;
				}
				folderMap.put(names[i], k);
				names[i] += "(" + k + ")";
			}
			folderMap.put(names[i], 1);
		}
		return names;
	}

	// 寻找旋转数组中的最小值
	public int findMin(int[] nums) {
		int l = 0, r = nums.length - 1;
		while (l < r) {
			int mid = l + r >> 1;
			if (nums[mid] > nums[nums.length - 1]) {
				l = mid + 1;
			} else {
				r = mid;
			}
		}
		return nums[l];
	}

	// 寻找旋转数组中的最小值Ⅱ
	public int findMinTwo(int[] nums) {
		int n = nums.length;
		int l = 0, r = n - 1;
		while (l < r && nums[0] == nums[r]) {
			r--;
		}
		while (l < r) {
			int mid = l + r + 1 >> 1;
			if (nums[mid] >= nums[0]) {
				l = mid;
			} else {
				r = mid - 1;
			}
		}
		return r + 1 < n ? nums[r + 1] : nums[0];
	}

	// 经营摩天轮的最大利润
	public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
		int ans = -1;
		int max = 0, cur = 0;
		int wait = 0, i = 0;
		while (wait > 0 || i < customers.length) {
			wait += i < customers.length ? customers[i] : 0;
			int up = Math.min(4, wait);
			wait -= up;
			i++;
			cur += up * boardingCost - runningCost;
			if (cur > max) {
				max = cur;
				ans = i;
			}
		}
		return ans;
	}

	// 使字符串平衡的最少删除次数
	public int minimumDeletions(String s) {
		int f = 0, cntB = 0;
		for (char c : s.toCharArray()) {
			if (c == 'b') {
				++cntB;
			} else {
				f = Math.min(f + 1, cntB);
			}
		}
		return f;
	}

	// 得到K个黑块额最少涂色次数
	public int minimumRecolors(String blocks, int k) {
		int cnt = 0;
		for (int i = 0; i < k; i++) {
			cnt += blocks.charAt(i) == 'W' ? 1 : 0;
		}
		int ans = cnt;
		for (int i = k; i < blocks.length(); i++) {
			cnt += blocks.charAt(i) == 'W' ? 1 : 0;
			cnt -= blocks.charAt(i - k) == 'W' ? 1 : 0;
			ans = Math.min(ans, cnt);
		}
		return ans;
	}

	// 统计作战单位数
	static int N = (int) 1e5 + 10;
	static int[] tr1 = new int[N], tr2 = new int[N];

	int lowbit(int x) {
		return x & -x;
	}

	void update(int[] tr, int x, int y) {
		for (int i = x; i < N; i += lowbit(i)) {
			tr[i] += y;
		}
	}

	int query(int[] tr, int x) {
		int ans = 0;
		for (int i = x; i > 0; i -= lowbit(i)) {
			ans += tr[i];
		}
		return ans;
	}

	public int numTeams(int[] rating) {
		int n = rating.length, ans = 0;
		Arrays.fill(tr1, 0);
		Arrays.fill(tr2, 0);
		for (int i : rating) {
			update(tr2, i, 1);
		}
		for (int i = 0; i < n; i++) {
			int t = rating[i];
			update(tr2, t, -1);
			ans += query(tr1, t - 1) * (query(tr2, N - 1) - query(tr2, t));
			ans += (query(tr1, N - 1) - query(tr1, t)) * query(tr2, t - 1);
			update(tr1, t, 1);
		}
		return ans;
	}

	// 分隔两个字符串得到回文串
	public boolean checkPalindromeFormation(String a, String b) {
		return check(a, b) || check(b, a);
	}

	private boolean check(String a, String b) {
		int i = 0, j = a.length() - 1;
		while (i < j && a.charAt(i) == b.charAt(j)) {
			++i;
			--j;
		}
		return isForm(a, i, j) || isForm(b, i, j);
	}

	private boolean isForm(String s, int i, int j) {
		while (i < j && s.charAt(i) == s.charAt(j)) {
			i++;
			j--;
		}
		return i >= j;
	}

	// 下一个更大元素一
	public int[] nextGreaterElement(int[] nums1, int[] nums2) {
		int n = nums1.length, m = nums2.length;
		Deque<Integer> deque = new ArrayDeque<>();
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = m - 1; i >= 0; i--) {
			int x = nums2[i];
			while (!deque.isEmpty() && x > deque.peekLast()) {
				deque.pollLast();
			}
			map.put(x, deque.isEmpty() ? -1 : deque.peekLast());
			deque.addLast(x);
		}
		int[] ans = new int[n];
		for (int i = 0; i < n; i++) {
			ans[i] = map.get(nums1[i]);
		}
		return ans;
	}

	// 下一个更大元素二
	public int[] nextGreaterElements(int[] nums) {
		int n = nums.length;
		int[] ans = new int[n];
		Arrays.fill(ans, -1);
		Deque<Integer> deque = new ArrayDeque<>();
		for (int i = 0; i < n * 2; i++) {
			while (!deque.isEmpty() && nums[i % n] > nums[deque.peekLast()]) {
				int idx = deque.pollLast();
				ans[idx] = nums[i % n];
			}
			deque.addLast(i % n);
		}
		return ans;
	}
}
package prepareForByte;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Kezi
 * @date 2023年02月18日 0:10
 */
public class Solution {


	// 买卖股票的最佳时机
	public int maxProfit(int[] prices) {
		int minPrices = Integer.MAX_VALUE;
		int maxProfit = 0;
		for (int i = 0; i < prices.length; i++) {
			if (prices[i] < minPrices) {
				minPrices = prices[i];
			} else if (prices[i] - minPrices > maxProfit) {
				maxProfit = prices[i] - minPrices;
			}
		}
		return maxProfit;
	}

	// 打家劫舍Ⅱ
	public int rob(int[] nums) {
		if (nums.length == 0) {
			return 0;
		}
		if (nums.length == 1) {
			return nums[0];
		}
		return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
	}

	private int myRob(int[] nums) {
		int pre = 0, cur = 0, tmp;
		for (int num : nums) {
			tmp = cur;
			cur = Math.max(pre + num, cur);
			pre = tmp;
		}
		return cur;
	}

	// 字符串的排列
	List<String> res = new LinkedList<>();
	char[] c;

	public String[] permutation(String s) {
		c = s.toCharArray();
		dfs(0);
		return res.toArray(new String[res.size()]);
	}

	private void dfs(int x) {
		if (x == c.length - 1) {
			res.add(String.valueOf(c));
			return;
		}
		Set<Character> set = new HashSet<>();
		for (int i = x; i < c.length; i++) {
			if (set.contains(c[i])) {
				continue;
			}
			set.add(c[i]);
			swap(i, x);
			dfs(x + 1);
			swap(i, x);
		}
	}

	private void swap(int i, int x) {
		char tmp = c[i];
		c[i] = c[x];
		c[x] = tmp;
	}

	// 环形链表
	public ListNode detectCycle(ListNode head) {
		ListNode fast = head, slow = head;
		while (true) {
			if (fast == null || fast.next == null) {
				return null;
			}
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				break;
			}
		}
		fast = head;
		while (fast != slow) {
			fast = fast.next;
			slow = slow.next;
		}
		return fast;
	}

	// 排序数组中只出现一次的数字
	public int singleNonDuplicate(int[] nums) {
		int n = nums.length, l = 0, r = nums.length - 1;
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
			} else {
				ans = nums[mid];
				break;
			}
		}
		return ans;
	}

	// 子集
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		backtrack(0, nums, res, new ArrayList<>());
		return res;
	}

	private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
		res.add(new ArrayList<>(tmp));
		for (int j = i; j < nums.length; j++) {
			tmp.add(nums[j]);
			backtrack(j + 1, nums, res, tmp);
			tmp.remove(tmp.size() - 1);
		}
	}

	// 重排链表
	public void reorderList(ListNode head) {
		if (head == null) {
			return;
		}
		ListNode mid = middleNode(head);
		ListNode l1 = head;
		ListNode l2 = mid.next;
		mid.next = null;
		l2 = reverseList(l2);
		mergeList(l1, l2);
	}

	private void mergeList(ListNode l1, ListNode l2) {
		ListNode l1_tmp;
		ListNode l2_tmp;
		while (l1 != null && l2 != null) {
			l1_tmp = l1.next;
			l2_tmp = l2.next;
			l1.next = l2;
			l1 = l1_tmp;
			l2.next = l1;
			l2 = l2_tmp;
		}
	}

	private ListNode reverseList(ListNode head) {
		ListNode prev = null;
		ListNode curr = head;
		while (curr != null) {
			ListNode next = curr.next;
			curr.next = prev;
			prev = curr;
			curr = next;
		}
		return prev;
	}

	private ListNode middleNode(ListNode head) {
		ListNode slow = head;
		ListNode fast = head;
		while (fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	// 下一个最大元素二
	public int[] nextGreaterElements(int[] nums) {
		int n = nums.length;
		int[] ans = new int[n];
		Arrays.fill(ans, -1);
		Deque<Integer> deque = new ArrayDeque<>();
		for (int i = 0; i < n * 2; i++) {
			while (!deque.isEmpty() && nums[i % n] > nums[deque.peekLast()]) {
				int poll = deque.pollLast();
				ans[poll] = nums[i % n];
			}
			deque.addLast(i % n);
		}
		return ans;
	}

	// 链表中倒数第K哥节点
	public ListNode getKthFromEnd(ListNode head, int k) {
		ListNode fast = head, slow = head;
		for (int i = 0; i < k; i++) {
			fast = fast.next;
		}
		while (fast != null) {
			fast = fast.next;
			slow = slow.next;
		}
		return slow;
	}

	// 划分字母区间
	public List<Integer> partitionLabels(String s) {
		int[] last = new int[26];
		int len = s.length();
		for (int i = 0; i < len; i++) {
			last[s.charAt(i) - 'a'] = i;
		}
		List<Integer> partition = new ArrayList<>();
		int start = 0, end = 0;
		for (int i = 0; i < len; i++) {
			end = Math.max(end, last[s.charAt(i) - 'a']);
			if (i == end) {
				partition.add(end - start + 1);
				start = end + 1;
			}
		}
		return partition;
	}

	// 二叉树的深度
	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
	}

	// 平衡二叉树
	public boolean isBalanced(TreeNode root) {
		if (root == null) {
			return true;
		} else {
			return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
		}
	}

	public int depth(TreeNode root) {
		if (root == null) {
			return 0;
		} else {
			return Math.max(depth(root.left), depth(root.right)) + 1;
		}
	}

	// 七进制数
	public String convertToBase7(int num) {
		boolean flag = num < 0;
		if (flag) {
			num = -num;
		}
		StringBuilder ans = new StringBuilder();
		do {
			ans.append(num % 7);
			num /= 7;
		} while (num != 0);
		return flag ? ans.append("-").reverse().toString() : ans.reverse().toString();
	}

	// 去除重复字母
	public String removeDuplicateLetters(String s) {
		Deque<Character> stack = new ArrayDeque<>();
		int[] count = new int[256];
		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i)]++;
		}
		boolean[] inStack = new boolean[256];
		for (Character c : s.toCharArray()) {
			count[c]--;
			if (inStack[c]) {
				continue;
			}
			while (!stack.isEmpty() && stack.peek() > c) {
				if (count[stack.peek()] == 0) {
					break;
				}
				inStack[stack.poll()] = false;
			}
			stack.push(c);
			inStack[c] = true;
		}
		StringBuilder sb = new StringBuilder();
		while (!stack.isEmpty()) {
			sb.append(stack.poll());
		}
		return sb.reverse().toString();
	}

	// 第N个斐波那契数
	public int tribonacci(int n) {
		if (n <= 1) {
			return n;
		}
		if (n == 2) {
			return 1;
		}
		int a = 0, b = 1, c = 1;
		for (int i = 3; i <= n; i++) {
			int d = a + b + c;
			a = b;
			b = c;
			c = d;
		}
		return c;
	}

	// 两数之和
	public int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> record = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			if (record.containsKey(target - nums[i])) {
				return new int[]{record.get(target - nums[i]), i};
			}
			record.put(nums[i], i);
		}
		return new int[0];
	}

	// 寻找峰值
	public int findPeakElement(int[] nums) {
		int n = nums.length;
		int l = 0, r = n - 1;
		while (l < r) {
			int mid = l + r >> 1;
			if (nums[mid] > nums[mid + 1]) {
				r = mid;
			} else {
				l = mid + 1;
			}
		}
		return r;
	}

	// 长度最小的子数组
	public int minSubArrayLen(int target, int[] nums) {
		int n = nums.length;
		if (n == 0) {
			return 0;
		}
		int ans = Integer.MAX_VALUE;
		int start = 0, end = 0;
		int sum = 0;
		while (end < n) {
			sum += nums[end];
			while (sum >= target) {
				ans = Math.min(ans, end - start + 1);
				sum -= nums[start];
				start++;
			}
			end++;
		}
		return ans == Integer.MAX_VALUE ? 0 : ans;
	}

	// 多个数组求交集
	public List<Integer> intersection(int[][] nums) {
		int[] cnt = new int[1001];
		int n = nums.length;
		for (int[] tmp : nums) {
			for (int num : tmp) {
				cnt[num]++;
			}
		}
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i <= 1000; i++) {
			if (cnt[i] == n) {
				list.add(i);
			}
		}
		return list;
	}

	// 你能在你最喜欢的那天吃到你最喜欢的糖果吗？
	public boolean[] canEat(int[] candiesCount, int[][] queries) {
		int m = candiesCount.length, n = queries.length;
		boolean[] ans = new boolean[n];
		long[] sum = new long[m + 1];
		for (int i = 1; i <= m; i++) {
			sum[i] = sum[i - 1] + candiesCount[i - 1];
		}
		for (int i = 0; i < n; i++) {
			int t = queries[i][0], d = queries[i][1] + 1, c = queries[i][2];
			long a = sum[t] / c + 1, b = sum[t + 1];
			ans[i] = a <= d && d <= b;
		}
		return ans;
	}

	// 非重叠矩形中的随机点
	int[][] rs;
	int[] sum;
	int n;
	Random random = new Random();

	public Solution(int[][] rects) {
		rs = rects;
		n = rects.length;
		sum = new int[n + 1];
		for (int i = 1; i <= n; i++) {
			sum[i] = sum[i - 1] + (rs[i - 1][2] - rs[i - 1][0] + 1) * (rs[i - 1][3] - rs[i - 1][1] + 1);
		}
	}

	public int[] pick() {
		int val = random.nextInt(sum[n]) + 1;
		int l = 0, r = n;
		while (l < r) {
			int mid = l + r >> 1;
			if (sum[mid] >= val) {
				r = mid;
			} else {
				l = mid + 1;
			}
		}
		int[] cur = rs[r - 1];
		int x = random.nextInt(cur[2] - cur[0] + 1) + cur[0], y = random.nextInt(cur[3] - cur[1] + 1) + cur[1];
		return new int[]{x, y};
	}

	// 和为K的子数组
	public int subarraySum(int[] nums, int k) {
		int count = 0, pre = 0;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, 1);
		for (int i = 0; i < nums.length; i++) {
			pre += nums[i];
			if (map.containsKey(pre - k)) {
				count += map.get(pre - k);
			}
			map.put(pre, map.getOrDefault(pre, 0) + 1);
		}
		return count;
	}

	// 礼物的最大价值
	public int maxValue(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		for (int j = 1; j < n; j++) {
			grid[0][j] += grid[0][j - 1];
		}
		for (int j = 1; j < m; j++) {
			grid[j][0] += grid[j - 1][0];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
			}
		}
		return grid[m - 1][n - 1];
	}

	// 二叉树中和为某一值的路径
	List<List<Integer>> ans = new ArrayList<>();
	int t;

	public List<List<Integer>> pathSum(TreeNode root, int target) {
		t = target;
		dfs(root, 0, new ArrayList<>());
		return ans;
	}

	void dfs(TreeNode root, int cur, List<Integer> list) {
		if (root == null) {
			return;
		}
		list.add(root.val);
		if (cur + root.val == t && root.left == null && root.right == null) {
			ans.add(new ArrayList<>(list));
		}
		dfs(root.left, cur + root.val, list);
		dfs(root.right, cur + root.val, list);
		list.remove(list.size() - 1);
	}

	// 子数组异或查询
	public int[] xorQueries(int[] arr, int[][] queries) {
		int n = arr.length, m = queries.length;
		int[] sum = new int[n + 1];
		for (int i = 1; i <= n; i++) {
			sum[i] = sum[i - 1] ^ arr[i - 1];
		}
		int[] ans = new int[m];
		for (int i = 0; i < m; i++) {
			int l = queries[i][0] + 1, r = queries[i][1] + 1;
			ans[i] = sum[r] ^ sum[l - 1];
		}
		return ans;
	}

	// 使数组和能被P整除
	public int minSubarray(int[] nums, int p) {
		int k = 0;
		for (int x : nums) {
			k = (k + x) % p;
		}
		if (k == 0) {
			return 0;
		}
		Map<Integer, Integer> last = new HashMap<>();
		last.put(0, -1);
		int n = nums.length;
		int ans = n;
		int cur = 0;
		for (int i = 0; i < n; i++) {
			cur = (cur + nums[i]) % p;
			int target = (cur - k + p) % p;
			if (last.containsKey(target)) {
				ans = Math.min(ans, i - last.get(target));
			}
			last.put(cur, i);
		}
		return ans == n ? -1 : ans;
	}

	// 字母与数字
	public String[] findLongestSubarray(String[] array) {
		int n = array.length;
		int[] sum = new int[n + 1];
		sum[0] = 0;
		for (int i = 1; i <= n; i++) {
			sum[i] = sum[i - 1] + (array[i - 1].charAt(0) >> 6 & 1) * 2 - 1;
		}
		int begin = 0, end = 0;
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0; i <= n; i++) {
			int j = map.getOrDefault(sum[i], -1);
			if (j < 0) {
				map.put(sum[i], i);
			} else if (i - j > end - begin) {
				begin = j;
				end = i;
			}
		}
		String[] sub = new String[end - begin];
		System.arraycopy(array, begin, sub, 0, end - begin);
		return sub;
	}

	// 替换子串得到平衡字符串
	public int balancedString(String s) {
		int[] cnt = new int[4];
		String t = "QWER";
		int n = s.length();
		for (int i = 0; i < n; i++) {
			cnt[t.indexOf(s.charAt(i))]++;
		}
		int m = n / 4;
		if (cnt[0] == m && cnt[1] == m && cnt[2] == m && cnt[3] == m) {
			return 0;
		}
		int ans = n;
		for (int i = 0, j = 0; j < n; j++) {
			cnt[t.indexOf(s.charAt(j))]--;
			while (i <= j && cnt[0] <= m && cnt[1] <= m && cnt[2] <= m && cnt[3] <= m) {
				ans = Math.min(ans, j - i + 1);
				cnt[t.indexOf(s.charAt(i++))]++;
			}
		}
		return ans;
	}

	// 字母异位词分组
	public List<List<String>> groupAnagrams(String[] strs) {
		return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> {
			char[] chars = str.toCharArray();
			Arrays.sort(chars);
			return new String(chars);
		})).values()
		);
	}

	// 有效的字母异位词
	public boolean isAnagram(String s, String t) {
		if (s.length() != t.length()) {
			return false;
		}
		int[] alpha = new int[26];
		for (int i = 0; i < s.length(); i++) {
			alpha[s.charAt(i) - 'a']++;
			alpha[t.charAt(i) - 'a']--;
		}
		for (int i = 0; i < 26; i++) {
			if (alpha[i] != 0) {
				return false;
			}
		}
		return true;
	}

	// 找到字符串中所有字母异位词
	public List<Integer> findAnagrams(String s, String p) {
		int n = s.length(), m = p.length();
		List<Integer> ans = new ArrayList<>();
		if (n < m) {
			return ans;
		}
		int[] pCnt = new int[26];
		int[] sCnt = new int[26];
		for (int i = 0; i < m; i++) {
			pCnt[p.charAt(i) - 'a']++;
		}
		int l = 0;
		for (int r = 0; r < n; r++) {
			int curR = s.charAt(r) - 'a';
			sCnt[curR]++;
			while (sCnt[curR] > pCnt[curR]) {
				int curL = s.charAt(l) - 'a';
				sCnt[curL]--;
				l++;
			}
			if (r - l + 1 == m) {
				ans.add(l);
			}
		}
		return ans;
	}

	// 最小时间差
	public int findMinDifference(List<String> timePoints) {
		int n = timePoints.size();
		if (n > 1440) {
			return 0;
		}
		int[] cnts = new int[1440 * 2 + 10];
		for (String s : timePoints) {
			String[] split = s.split(":");
			int h = Integer.parseInt(split[0]), m = Integer.parseInt(split[1]);
			cnts[h * 60 + m]++;
			cnts[h * 60 + m + 1440]++;
		}
		int ans = 1440, last = -1;
		for (int i = 0; i < 1440 * 2 && ans != 0; i++) {
			if (cnts[i] == 0) {
				continue;
			}
			if (cnts[i] > 1) {
				ans = 0;
			} else if (last != -1) {
				ans = Math.min(ans, i - last);
			}
			last = i;
		}
		return ans;
	}

	// 最大网络秩
	public int maximalNetworkRank(int n, int[][] roads) {
		int[] cnt = new int[n];
		int[][] g = new int[n][n];
		for (int[] road : roads) {
			int a = road[0], b = road[1];
			g[a][b] = 1;
			g[b][a] = 1;
			cnt[a]++;
			cnt[b]++;
		}
		int ans = 0;
		for (int a = 0; a < n; a++) {
			for (int b = a + 1; b < n; b++) {
				ans = Math.max(ans, cnt[a] + cnt[b] - g[a][b]);
			}
		}
		return ans;
	}

	// 全排列二
	public List<List<Integer>> permuteUnique(int[] nums) {
		int len = nums.length;
		List<List<Integer>> res = new ArrayList<>();
		if (len == 0) {
			return res;
		}
		Arrays.sort(nums);
		boolean[] used = new boolean[len];
		Deque<Integer> path = new ArrayDeque<>();
		dfs(nums, len, 0, used, path, res);
		return res;
	}

	private void dfs(int[] nums, int len, int depth, boolean[] used, Deque<Integer> path, List<List<Integer>> res) {
		if (depth == len) {
			res.add(new ArrayList<>(path));
			return;
		}
		for (int i = 0; i < len; i++) {
			if (used[i]) {
				continue;
			}
			if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
				continue;
			}
			path.addLast(nums[i]);
			used[i] = true;
			dfs(nums, len, depth + 1, used, path, res);
			used[i] = false;
			path.removeLast();
		}
	}

	// 水位上升的泳池中游泳
	int waterN;
	int[] p;

	void union(int a, int b) {
		p[find(a)] = p[find(b)];
	}

	boolean query(int a, int b) {
		return find(a) == find(b);
	}

	int find(int x) {
		if (p[x] != x) {
			p[x] = find(p[x]);
		}
		return p[x];
	}

	public int swimInWater(int[][] grid) {
		waterN = grid.length;
		p = new int[waterN * waterN];
		for (int i = 0; i < waterN * waterN; i++) {
			p[i] = i;
		}
		List<int[]> edges = new ArrayList<>();
		for (int i = 0; i < waterN; i++) {
			for (int j = 0; j < waterN; j++) {
				int idx = getIndex(i, j);
				p[idx] = idx;
				if (i + 1 < waterN) {
					int a = idx, b = getIndex(i + 1, j);
					int w = Math.max(grid[i][j], grid[i + 1][j]);
					edges.add(new int[]{a, b, w});
				}
				if (j + 1 < waterN) {
					int a = idx, b = getIndex(i, j + 1);
					int w = Math.max(grid[i][j], grid[i][j + 1]);
					edges.add(new int[]{a, b, w});
				}
			}
		}
		Collections.sort(edges, (a, b) -> a[2] - b[2]);
		int start = getIndex(0, 0), end = getIndex(n - 1, n - 1);
		for (int[] edge : edges) {
			int a = edge[0], b = edge[1], w = edge[2];
			union(a, b);
			if (query(start, end)) {
				return w;
			}
		}
		return 0;
	}

	private int getIndex(int i, int j) {
		return i * waterN + j;
	}

	// 和有限的最长子序列
	public int[] answerQueries(int[] nums, int[] queries) {
		Arrays.sort(nums);
		for (int i = 1; i < nums.length; i++) {
			nums[i] += nums[i - 1];
		}
		int m = queries.length;
		int[] ans = new int[m];
		for (int i = 0; i < m; i++) {
			ans[i] = search(nums, queries[i]);
		}
		return ans;
	}

	private int search(int[] nums, int x) {
		int l = 0, r = nums.length;
		while (l < r) {
			int mid = (l + r) >> 1;
			if (nums[mid] > x) {
				r = mid;
			} else {
				l = mid + 1;
			}
		}
		return l;
	}

	// 执行操作后字典序最小的字符串
	public String findLexSmallestString(String s, int a, int b) {
		Deque<String> deque = new ArrayDeque<>();
		deque.offer(s);
		Set<String> vis = new HashSet<>();
		vis.add(s);
		String ans = s;
		int n = s.length();
		while (!deque.isEmpty()) {
			s = deque.poll();
			if (ans.compareTo(s) > 0) {
				ans = s;
			}
			char[] cs = s.toCharArray();
			for (int i = 0; i < n; i += 2) {
				cs[i] = (char) (((cs[i] - '0' + a) % 10) + '0');
			}
			String s1 = String.valueOf(cs);
			String s2 = s.substring(n - b) + s.substring(0, n - b);
			if (vis.add(s1)) {
				deque.offer(s1);
			}
			if (vis.add(s2)) {
				deque.offer(s2);
			}
		}
		return ans;
	}

	// 132	模式
	public boolean find132pattern(int[] nums) {
		int n = nums.length;
		Deque<Integer> deque = new ArrayDeque<>();
		int k = Integer.MIN_VALUE;
		for (int i = n - 1; i >= 0; i--) {
			if (nums[i] < k) {
				return true;
			}
			while (!deque.isEmpty() && deque.peekLast() < nums[i]) {
				k = Math.max(k, deque.pollLast());
			}
			deque.offer(nums[i]);
		}
		return false;
	}

	// 最大二叉树
	public TreeNode constructMaximumBinaryTree(int[] nums) {
		Deque<TreeNode> deque = new ArrayDeque<>();
		for (int i = 0; i < nums.length; i++) {
			TreeNode node = new TreeNode(nums[i]);
			while (!deque.isEmpty()) {
				TreeNode top = deque.peekLast();
				if (top.val > node.val) {
					deque.addLast(node);
					top.right = node;
					break;
				} else {
					deque.removeLast();
					node.left = top;
				}
			}
			if (deque.isEmpty()) {
				deque.addLast(node);
			}
		}
		return deque.peekLast();
	}
}

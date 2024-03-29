package swordFingerProvided;


import java.util.*;

/**
 * @author Kezi
 * @date 2022年06月28日 0:06
 */
public class Solution {

	// 青蛙跳台阶问题
	public int numWays(int n) {
		int a = 1, b = 1, sum;
		for (int i = 0; i < n; i++) {
			sum = (a + b) % 1000000007;
			a = b;
			b = sum;
		}
		return a;
	}

	// 股票的最大利润
	public int maxProfit(int[] prices) {
		int cost = Integer.MAX_VALUE, profit = 0;
		for (int price : prices) {
			cost = Math.min(cost, price);
			profit = Math.max(profit, price - cost);
		}
		return profit;
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

	/**
	 * 礼物的最大价值
	 * 根据题目说明，易得某单元格只可能从上边单元格或左边单元格到达
	 */
	public int maxValue(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		// 初始化第一行、第一列,用于优化
		for (int j = 1; j < n; j++) {
			grid[0][j] += grid[0][j - 1];
		}
		for (int i = 1; i < m; i++) {
			grid[i][0] += grid[i - 1][0];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				grid[i][j] = Math.max(grid[i][j - 1], grid[i - 1][j]);
			}
		}
		return grid[m - 1][n - 1];
	}

	/**
	 * 数组组成最小的数
	 * 若拼接字符串x+y>y+x ，则 x “大于” y ；
	 * 反之，若 x + y < y + x ，则 x “小于” y ；
	 */
	public String minNumber(int[] nums) {
		String[] strs = new String[nums.length];
		for (int i = 0; i < nums.length; i++) {
			strs[i] = String.valueOf(nums[i]);
		}
		Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
		StringBuilder res = new StringBuilder();
		for (String s : strs) {
			res.append(s);
		}
		return res.toString();
	}

	/**
	 * 求1+2+3+。。。+n
	 * 要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
	 */
	public int sumNums(int n) {
		boolean x = n > 1 && (n += sumNums((n - 1))) > 0;
		return n;
	}

	// 链表中倒数第K个节点
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

	/**
	 * 把字符串转换为整数
	 * border=2147483647(Integer.MAX_VALUE)//10=214748364
	 */
	public int strToInt(String str) {
		int res = 0, border = Integer.MAX_VALUE / 10;
		int i = 0, sign = 1, length = str.length();
		if (length == 0) {
			return 0;
		}
		while (str.charAt(i) == ' ') {
			if (++i == length) {
				return 0;
			}
		}
		if (str.charAt(i) == '-') {
			sign = -1;
		}
		if (str.charAt(i) == '-' || str.charAt(i) == '+') {
			i++;
		}
		for (int j = i; j < length; j++) {
			if (str.charAt(j) < '0' || str.charAt(j) > '9') {
				break;
			}
			if (res > border || res == border && str.charAt(j) > '7') {
				return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
			}
			res = res * 10 + (str.charAt(j) - '0');
		}
		return sign * res;
	}

	// 二进制加法
	public String addBinary(String a, String b) {
		StringBuilder res = new StringBuilder();
		int i = a.length() - 1;
		int j = b.length() - 1;
		int carry = 0;
		while (i >= 0 || j >= 0) {
			int digitA = i >= 0 ? a.charAt(i--) - '0' : 0;
			int digitB = j >= 0 ? b.charAt(j--) - '0' : 0;
			int sum = digitA + digitB + carry;
			carry = sum >= 2 ? 1 : 0;
			sum = sum >= 2 ? sum - 2 : sum;
			res.append(sum);
		}
		if (carry == 1) {
			res.append(1);
		}
		return res.reverse().toString();
	}

	// 只出现一次的数字
	public int singleNumber(int[] nums) {
		Map<Integer, Integer> freq = new HashMap<Integer, Integer>();
		for (int num : nums) {
			freq.put(num, freq.getOrDefault(num, 0) + 1);
		}
		int ans = 0;
		for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
			int num = entry.getKey(), occ = entry.getValue();
			if (occ == 1) {
				ans = num;
				break;
			}
		}
		return ans;
	}

	/**
	 * 单词长度的最大乘积
	 * 位运算 + 预计算
	 * 时间复杂度：O((m + n)* n)
	 * 空间复杂度：O(n)
	 */
	public int maxProduct(String[] words) {
		Map<Integer, Integer> map = new HashMap<>();
		int n = words.length;
		for (int i = 0; i < n; i++) {
			int bitMask = 0;
			for (char c : words[i].toCharArray()) {
				bitMask |= (1 << c - 'a');
			}
			map.put(bitMask, Math.max(map.getOrDefault(bitMask, 0), words[i].length()));
		}
		int ans = 0;
		for (int x : map.keySet()) {
			for (int y : map.keySet()) {
				if ((x & y) == 0) {
					ans = Math.max(ans, map.get(x) * map.get(y));
				}
			}
		}
		return ans;
	}

	/**
	 * 左右两边子数组的和相等
	 * 总和为total,左侧元素之和为sum，则 2 * sum + nums[i] = total
	 */
	public int pivotIndex(int[] nums) {
		int total = Arrays.stream(nums).sum();
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			if (2 * sum + nums[i] == total) {
				return i;
			}
			sum += nums[i];
		}
		return -1;
	}

	// 和为k的子数组
	public int subarraySum(int[] nums, int k) {
		int pre_sum = 0;
		int res = 0;
		HashMap<Integer, Integer> map = new HashMap<>(16);
		map.put(0, 1);
		for (int i : nums) {
			pre_sum += i;
			res += map.getOrDefault(pre_sum - k, 0);
			map.put(pre_sum, map.getOrDefault(pre_sum, 0) + 1);
		}
		return res;
	}

	// 翻转链表
	public ListNode reverseList(ListNode head) {
		ListNode prev = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = prev;
			prev = cur;
			cur = next;
		}
		return prev;
	}

	// 链表中的两数相加
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		Deque<Integer> stack1 = new ArrayDeque<>();
		Deque<Integer> stack2 = new ArrayDeque<>();
		while (l1 != null) {
			stack1.push(l1.val);
			l1 = l1.next;
		}
		while (l2 != null) {
			stack2.push(l2.val);
			l2 = l2.next;
		}
		int carry = 0;
		ListNode ans = null;
		while (!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
			int a = stack1.isEmpty() ? 0 : stack1.pop();
			int b = stack2.isEmpty() ? 0 : stack2.pop();
			int cur = a + b + carry;
			carry = cur / 10;
			cur %= 10;
			ListNode curNode = new ListNode(cur);
			curNode.next = ans;
			ans = curNode;
		}
		return ans;
	}

	// 删除链表的倒数第n个节点
	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode slow = head;
		ListNode fast = head;
		for (int i = 0; i < n; i++) {
			fast = fast.next;
		}
		if (fast == null) {
			return head;
		}
		while (fast.next != null) {
			slow = slow.next;
			fast = fast.next;
		}
		slow.next = slow.next.next;
		return head;
	}

	// 有效的变位词
	public boolean isAnagram(String s, String t) {
		if (s.length() != t.length() || s.equals(t)) {
			return false;
		}
		int[] table = new int[26];
		for (int i = 0; i < s.length(); i++) {
			table[s.charAt(i) - 'a']++;
		}
		for (int i = 0; i < t.length(); i++) {
			table[t.charAt(i) - 'a']--;
			if (table[t.charAt(i) - 'a'] < 0) {
				return false;
			}
		}
		return true;
	}

	// 变位词组
	public List<List<String>> groupAnagrams(String[] strs) {
		HashMap<String, ArrayList<String>> map = new HashMap<>();
		for (String str : strs) {
			char[] chars = str.toCharArray();
			Arrays.sort(chars);
			String key = new String(chars);
			ArrayList<String> tmp = map.getOrDefault(key, new ArrayList<>());
			tmp.add(str);
			map.put(key, tmp);
		}
		return new ArrayList<>(map.values());
	}

	// 每日温度
	public int[] dailyTemperatures(int[] temperatures) {
		Deque<Integer> deque = new ArrayDeque<>();
		int[] res = new int[temperatures.length];
		for (int i = 0; i < temperatures.length; i++) {
			while (!deque.isEmpty() && temperatures[deque.peekLast()] < temperatures[i]) {
				int index = deque.pollLast();
				res[index] = i - index;
			}
			deque.addLast(i);
		}
		return res;
	}

	// 回文链表
	public boolean isPalindrome(ListNode head) {
		List<Integer> vals = new ArrayList<>();
		// 将链表复制到数组
		ListNode current = head;
		while (current != null) {
			vals.add(current.val);
			current = current.next;
		}
		// 双指针判断
		int front = 0;
		int back = vals.size() - 1;
		while (front < back) {
			if (!vals.get(front).equals(vals.get(back))) {
				return false;
			}
			front++;
			back--;
		}
		return true;
	}

	// 二叉树中最底层最左边的值
	public int findBottomLeftValue(TreeNode root) {
		Queue<TreeNode> queue = new ArrayDeque<>();
		queue.add(root);
		int ret = root.val;
		while (!queue.isEmpty()) {
			int lg = queue.size();
			for (int i = 0; i < lg; i++) {
				TreeNode q = queue.poll();
				if (i == 0) {
					ret = q.val;
				}
				if (q.left != null) {
					queue.add(q.left);
				}
				if (q.right != null) {
					queue.add(q.right);
				}
			}
		}
		return ret;
	}

	// 两个链表的第一个重合点
	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		if (headA == null || headB == null) {
			return null;
		}
		ListNode pa = headA, pb = headB;
		while (pa != pb) {
			pa = pa == null ? headB : pa.next;
			pb = pb == null ? headA : pb.next;
		}
		return pa;
	}


	// 二叉树每层的最大值
	public List<Integer> largestValues(TreeNode root) {
		Queue<TreeNode> queue = new ArrayDeque<>();
		List<Integer> ret = new ArrayList<>();
		if (root != null) {
			queue.add(root);
		}
		while (!queue.isEmpty()) {
			int num = Integer.MIN_VALUE;
			int lg = queue.size();
			for (int i = 0; i < lg; i++) {
				TreeNode p = queue.poll();
				num = Math.max(num, p.val);
				if (p.left != null) {
					queue.add(p.left);
				}
				if (p.right != null) {
					queue.add(p.right);
				}
			}
			ret.add(num);
		}
		return ret;
	}

	// 和大于等于target鞥最短子数组
	public int minSubArrayLen(int target, int[] nums) {
		int left = 0;
		int total = 0;
		int ret = Integer.MAX_VALUE;
		for (int right = 0; right < nums.length; right++) {
			total += nums[right];
			while (total >= target) {
				ret = Math.min(ret, right - left + 1);
				total -= nums[left++];
			}
		}
		return ret > nums.length ? 0 : ret;
	}

	// 求平方根
	public int mySqrt(int x) {
		int l = 0, r = x, ans = -1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if ((long) mid * mid <= x) {
				ans = mid;
				l = mid + 1;
			} else {
				r = mid - 1;
			}
		}
		return ans;
	}

	// 狒狒吃香蕉
	public int minEatingSpeed(int[] piles, int h) {
		int low = 1;
		int high = 0;
		for (int pile : piles) {
			high = Math.max(high, pile);
		}
		int k = high;
		while (low < high) {
			int speed = (high - low) / 2 + low;
			long time = getTime(piles, speed);
			if (time <= h) {
				k = speed;
				high = speed;
			} else {
				low = speed + 1;
			}
		}
		return k;
	}

	public long getTime(int[] piles, int speed) {
		long time = 0;
		for (int pile : piles) {
			int curTime = (pile + speed - 1) / speed;
			time += curTime;
		}
		return time;
	}

	/**
	 * 合并区间
	 * 如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的
	 */
	public int[][] merge(int[][] intervals) {
		if (intervals.length == 0) {
			return new int[0][2];
		}
		Arrays.sort(intervals, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[0] - o2[0];
			}
		});
		List<int[]> merged = new ArrayList<>();
		for (int i = 0; i < intervals.length; i++) {
			int L = intervals[i][0], R = intervals[i][1];
			if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
				merged.add(new int[]{L, R});
			} else {
				merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
			}
		}
		return merged.toArray(new int[merged.size()][]);
	}

	/**
	 * 数组相对排序
	 * 方法返回值大于0的话就是前一个数和后一个数交换，如果b在map里面，a不在就换一下
	 */
	public int[] relativeSortArray(int[] arr1, int[] arr2) {
		Map<Integer, Integer> map = new HashMap<>();
		int len = arr2.length;
		for (int i = 0; i < len; i++) {
			map.put(arr2[i], i);
		}
		return Arrays.stream(arr1).boxed().sorted((i1, i2) -> {
			if (map.containsKey(i1) && map.containsKey(i2)) {
				return map.get(i1) - map.get(i2);
			} else if (map.containsKey(i1)) {
				return -1;
			} else if (map.containsKey(i2)) {
				return 1;
			} else {
				return i1 - i2;
			}
		}).mapToInt(Integer::valueOf).toArray();
	}

	// 数组中的第K大的数字
	public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> pq = new PriorityQueue<>();
		for (int i = 0; i < nums.length; i++) {
			pq.offer(nums[i]);
			if (pq.size() > k) {
				pq.poll();
			}
		}
		return pq.poll();
	}

	// 所有子集
	List<List<Integer>> res = new ArrayList<>();
	List<Integer> tmp = new ArrayList<>();

	public List<List<Integer>> subsets(int[] nums) {
		dfs(0, nums);
		return res;
	}

	private void dfs(int i, int[] nums) {
		if (i == nums.length) {
			res.add(new ArrayList<>(tmp));
			return;
		}
		tmp.add(nums[i]);
		dfs(i + 1, nums);
		tmp.remove(tmp.size() - 1);
		dfs(i + 1, nums);
	}

	// 爬楼梯的最少成本
	public int minCostClimbingStairs(int[] cost) {
		int n = cost.length;
		int pre = 0, cur = 0;
		for (int i = 2; i <= n; i++) {
			int next = Math.min(cur + cost[i - 1], pre + cost[i - 2]);
			pre = cur;
			cur = next;
		}
		return cur;
	}

	// 三角形中最小路径之和
	public int minimumTotal(List<List<Integer>> triangle) {
		int m = triangle.size();
		int[] dp = new int[m + 1];
		// 从底部开始
		for (int i = m - 1; i >= 0; i--) {
			// 行内顺序更新
			for (int j = 0; j <= i; j++) {
				dp[j] = triangle.get(i).get(j) + Math.min(dp[j], dp[j + 1]);
			}
		}
		return dp[0];
	}

	// 分割等和子串
	public boolean canPartition(int[] nums) {
		int n = nums.length;
		if (n < 2) {
			return false;
		}
		int sum = 0, maxNum = 0;
		for (int num : nums) {
			sum += num;
			maxNum = Math.max(maxNum, num);
		}
		if (sum % 2 != 0) {
			return false;
		}
		int target = sum / 2;
		if (maxNum > target) {
			return false;
		}
		boolean[] dp = new boolean[target + 1];
		dp[0] = true;
		for (int i = 0; i < n; i++) {
			int num = nums[i];
			for (int j = target; j >= num; j--) {
				dp[j] |= dp[j - num];
			}
		}
		return dp[target];
	}

	// 岛屿的最大面积
	public int maxAreaOfIsland(int[][] grid) {
		int ans = 0;
		for (int i = 0; i < grid.length; ++i) {
			for (int j = 0; j < grid[0].length; ++j) {
				ans = Math.max(ans, dfs(grid, i, j));
			}
		}
		return ans;
	}

	private int dfs(int[][] grid, int i, int j) {
		if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] != 1) {
			return 0;
		}
		grid[i][j] = 0;
		int[] di = {0, 0, -1, 1};
		int[] dj = {1, -1, 0, 0};
		int ans = 1;
		for (int index = 0; index < 4; ++index) {
			int next_i = i + di[index], next_j = j + dj[index];
			ans += dfs(grid, next_i, next_j);
		}
		return ans;
	}

	// 粉刷房子--算easy题，要牢记
	public int minCost(int[][] costs) {
		int n = costs.length;
		int a = costs[0][0], b = costs[0][1], c = costs[0][2];
		for (int i = 1; i < n; i++) {
			int d = Math.min(b, c) + costs[i][0];
			int e = Math.min(a, c) + costs[i][1];
			int f = Math.min(a, b) + costs[i][2];
			a = d;
			b = e;
			c = f;
		}
		return Math.min(a, Math.min(b, c));
	}

	// 重排链表
	public void reorderList(ListNode head) {
		if (head == null) {
			return;
		}
		List<ListNode> list = new ArrayList<>();
		ListNode node = head;
		while (node != null) {
			list.add(node);
			node = node.next;
		}
		int i = 0, j = list.size() - 1;
		while (i < j){
			list.get(i).next = list.get(j);
			i++;
			if (i == j){
				break;
			}
			list.get(j).next = list.get(i);
			j--;
		}
		list.get(i).next = null;
	}
}
